from pathlib import Path
import csv
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave

# This must be set before importing PyTorch. Long generative audio outputs can fragment
# the CUDA allocator, so expandable segments reduce avoidable OOM failures without
# changing the model or generation settings.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json5
import numpy
import torch
import whisper
from huggingface_hub.utils import disable_progress_bars
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

# Keep the experiment anchored to the repository rather than the caller's working
# directory. This lets the same command work from either the project root or audio/.
script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
levels_path = project_directory / "src" / "levels.ts"
report_path = script_directory / "alignment_verification_report.csv"

# The lesson data in levels.ts is the authoritative source. This experiment must test
# the same Japanese strings and chunk boundaries used by the frontend rather than
# maintaining a second, easily diverged copy of the curriculum.
levels_start_marker = "/* AUDIO_LEVELS_START */"
levels_end_marker = "/* AUDIO_LEVELS_END */"
qwen_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
speaker_name = "Ono_Anna"
# MFA accepts its published Japanese model by this registry name. The aligner is used
# only to locate known text inside known speech; it is not being asked to transcribe
# or judge pronunciation quality.
mfa_model_name = "japanese_mfa"
whisper_model_name = "large-v3"
whisper_language = "ja"
whisper_beam_size = 5
sentence_sample_count = 100
selection_seed = 0
tts_seed = 0
gpu_index = 0
whisper_sample_rate = 16000
# MFA is CPU-heavy even though Qwen and Whisper run on the GPU. Capping its worker
# count keeps this one-off test from monopolizing the machine while still allowing
# the corpus alignment to run in parallel.
mfa_num_jobs = max(1, min(4, os.cpu_count() or 1))

alignment_punctuation = " \t\n\r\"'`“”‘’「」『』（）()［］[]【】。、，．,.！？!?・:：;；…"

# The CSV intentionally records raw evidence rather than a programmatic pass/fail
# score. Expected text, alignment metadata, audio measurements, and Whisper output
# are retained so a later language-aware review can distinguish TTS, alignment,
# and ASR failures instead of collapsing them into one number.
report_fields = [
    "report_row_number",
    "sentence_sample_number",
    "row_type",
    "level_id",
    "sentence_id",
    "sentence_index",
    "chunk_index",
    "full_sentence_text",
    "expected_text",
    "alignment_token",
    "alignment_label",
    "alignment_valid",
    "alignment_error",
    "alignment_labels_json",
    "expected_chunk_count",
    "aligned_word_count",
    "previous_chunk",
    "next_chunk",
    "alignment_start_seconds",
    "alignment_end_seconds",
    "clip_duration_seconds",
    "source_sentence_duration_seconds",
    "whisper_text",
    "whisper_model",
    "whisper_language",
    "whisper_segment_count",
    "whisper_average_log_probability",
    "whisper_max_no_speech_probability",
    "whisper_max_compression_ratio",
    "audio_sample_rate",
    "audio_rms",
    "audio_peak",
    "whisper_segments_json",
    "tts_model",
    "tts_speaker",
    "alignment_model",
    "alignment_tool_version",
    "selection_seed",
    "tts_seed",
]


def command_text(command):
    return " ".join(str(part) for part in command)


def command_output(result):
    output_parts = []
    stdout = result.stdout.strip() if result.stdout else ""
    stderr = result.stderr.strip() if result.stderr else ""

    if stdout:
        output_parts.append(f"stdout:\n{stdout}")

    if stderr:
        output_parts.append(f"stderr:\n{stderr}")

    if not output_parts:
        return "出力はありません。"

    return "\n\n".join(output_parts)


# Native-tool failures previously looked like generic "command not found" errors.
# Preserve stdout, stderr, the exact command, and its exit code so the next failure
# points to the real dependency or argument problem instead of starting whack-a-mole.
def run_text_command(command, description, environment=None):
    result = subprocess.run(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )

    if result.returncode != 0:
        sys.exit(
            f"{description}\n"
            f"コマンド: {command_text(command)}\n"
            f"終了コード: {result.returncode}\n"
            f"{command_output(result)}"
        )

    return result


def executable_path(path):
    if not path:
        return None

    candidate = Path(path).expanduser()

    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.resolve()

    return None


def find_ffmpeg_executable():
    ffmpeg_path = executable_path(shutil.which("ffmpeg"))

    if ffmpeg_path is None:
        sys.exit("FFmpeg が見つかりません。PATH に ffmpeg があることを確認してください。")

    return ffmpeg_path


# MFA carries native Kaldi dependencies and therefore lives in its own Conda
# environment. Discover its executable directly so normal use requires only the
# project .venv; users should not have to stack or repeatedly activate environments.
def find_mfa_executable():
    candidates = []
    environment_path = os.environ.get("MFA_EXECUTABLE")

    if environment_path:
        candidates.append(Path(environment_path).expanduser())

    path_command = shutil.which("mfa")

    if path_command:
        candidates.append(Path(path_command))

    conda_prefix = os.environ.get("CONDA_PREFIX")

    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mfa")

    home_directory = Path.home()

    for conda_directory_name in [
        "miniforge3",
        "mambaforge",
        "miniconda3",
        "anaconda3",
    ]:
        candidates.append(
            home_directory
            / conda_directory_name
            / "envs"
            / "mfa"
            / "bin"
            / "mfa"
        )

    checked_paths = []
    seen_paths = set()

    for candidate in candidates:
        candidate = candidate.expanduser()
        candidate_key = str(candidate)

        if candidate_key in seen_paths:
            continue

        seen_paths.add(candidate_key)
        checked_paths.append(candidate)
        resolved_candidate = executable_path(candidate)

        if resolved_candidate is not None:
            return resolved_candidate

    checked_text = "\n".join(f"- {path}" for path in checked_paths)
    sys.exit(
        "Montreal Forced Aligner の mfa 実行ファイルが見つかりません。\n"
        "MFA を mfa という名前の Conda 環境にインストールするか、"
        "MFA_EXECUTABLE に実行ファイルの絶対パスを設定してください。\n"
        f"確認した場所:\n{checked_text}"
    )


# Calling the MFA binary by absolute path is not sufficient if it inherits Python or
# virtual-environment variables from the project .venv. Build a minimal process
# environment that points MFA at its own binaries and native libraries while leaving
# the parent Python process untouched.
def create_mfa_environment(mfa_executable):
    environment = os.environ.copy()
    mfa_bin_directory = mfa_executable.parent
    mfa_environment_directory = mfa_bin_directory.parent
    current_path = environment.get("PATH", "")

    environment["PATH"] = f"{mfa_bin_directory}{os.pathsep}{current_path}"
    environment["CONDA_PREFIX"] = str(mfa_environment_directory)
    environment["CONDA_DEFAULT_ENV"] = mfa_environment_directory.name

    for variable_name in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"]:
        environment.pop(variable_name, None)

    return environment


# Validate every external command before loading multi-gigabyte models. In particular,
# checking the actual align_hf subcommand catches an incompatible MFA installation
# before any expensive TTS work begins.
def check_tools():
    ffmpeg_executable = find_ffmpeg_executable()
    ffmpeg_result = run_text_command(
        [ffmpeg_executable, "-version"],
        "FFmpeg の起動確認に失敗しました。",
    )
    ffmpeg_version = ffmpeg_result.stdout.splitlines()[0].strip()

    mfa_executable = find_mfa_executable()
    mfa_environment = create_mfa_environment(mfa_executable)
    mfa_result = run_text_command(
        [mfa_executable, "version"],
        "Montreal Forced Aligner の起動確認に失敗しました。",
        mfa_environment,
    )
    run_text_command(
        [mfa_executable, "align_hf", "--help"],
        "Montreal Forced Aligner に align_hf コマンドがありません。",
        mfa_environment,
    )

    mfa_version = (mfa_result.stdout or mfa_result.stderr).strip()

    if not mfa_version:
        mfa_version = "unknown"

    print(f"FFmpeg: {ffmpeg_version}")
    print(f"MFA: {mfa_version}")
    print(f"MFA実行ファイル: {mfa_executable}")
    print(f"MFA並列ジョブ数: {mfa_num_jobs}")

    return ffmpeg_executable, mfa_executable, mfa_environment, mfa_version


# Require exactly one marker pair. Silently choosing the first of several matching
# blocks could test stale or unrelated lesson data while still producing a believable
# report.
def load_levels():
    if not levels_path.is_file():
        sys.exit(f"levels.ts が見つかりません: {levels_path}")

    levels_source = levels_path.read_text(encoding="utf-8")
    start_count = levels_source.count(levels_start_marker)
    end_count = levels_source.count(levels_end_marker)

    if start_count != 1 or end_count != 1:
        sys.exit(
            "levels.ts の音声データ用マーカーが一意ではありません。"
            f" 開始={start_count}件、終了={end_count}件"
        )

    start_position = levels_source.index(levels_start_marker)
    end_position = levels_source.index(levels_end_marker)

    if end_position <= start_position:
        sys.exit("levels.ts の音声データ用マーカーの順序が正しくありません。")

    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    levels = json5.loads(levels_text)

    if not isinstance(levels, list) or not levels:
        sys.exit("levels.ts の音声データが空か、配列ではありません。")

    return levels


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text))


# Punctuation remains in the natural sentence sent to Qwen because it can affect
# phrasing. It is removed only from the space-delimited MFA token, where punctuation
# is not a spoken chunk and would otherwise create an artificial alignment target.
def get_alignment_token(text):
    token = normalize_text(text).strip(alignment_punctuation)

    if token:
        return token

    return normalize_text(text).strip()


# Qwen receives the complete sentence for linguistic context. The existing curriculum
# chunks are supplied separately to MFA so the resulting short clips inherit the
# sentence-level reading instead of asking TTS to pronounce isolated particles.
def collect_sentence_jobs(levels):
    jobs = []
    level_ids = set()
    sentence_keys = set()

    for level in levels:
        level_id = str(level.get("id", "")).strip()
        sentences = level.get("sentences")

        if not level_id:
            sys.exit("IDのないレベルが見つかりました。")

        if level_id in level_ids:
            sys.exit(f"重複したレベルIDがあります: {level_id}")

        if not isinstance(sentences, list) or not sentences:
            sys.exit(f"文がないレベルがあります: {level_id}")

        level_ids.add(level_id)

        for sentence_index, sentence in enumerate(sentences):
            sentence_id = str(sentence.get("id", "")).strip()
            chunk_objects = sentence.get("chunks")
            sentence_key = (level_id, sentence_id)

            if not sentence_id:
                sys.exit(f"IDのない文があります: {level_id} / {sentence_index}")

            if sentence_key in sentence_keys:
                sys.exit(f"重複した文IDがあります: {level_id} / {sentence_id}")

            if not isinstance(chunk_objects, list) or not chunk_objects:
                sys.exit(f"チャンクがない文があります: {level_id} / {sentence_id}")

            sentence_keys.add(sentence_key)
            chunks = []

            for chunk_index, chunk in enumerate(chunk_objects):
                japanese = normalize_text(chunk.get("japanese", ""))

                if not japanese:
                    sys.exit(
                        "空の日本語チャンクがあります: "
                        f"{level_id} / {sentence_id} / {chunk_index}"
                    )

                chunks.append(japanese)

            alignment_tokens = [get_alignment_token(chunk) for chunk in chunks]
            full_sentence_text = "".join(chunks)

            for chunk_index, token in enumerate(alignment_tokens):
                if not token:
                    sys.exit(
                        "整列できない空のチャンクがあります: "
                        f"{level_id} / {sentence_id} / {chunk_index}"
                    )

                if any(character.isspace() for character in token):
                    sys.exit(
                        "整列用チャンクに空白があります: "
                        f"{level_id} / {sentence_id} / {chunk_index} / {token}"
                    )

            jobs.append({
                "level_id": level_id,
                "sentence_id": sentence_id,
                "sentence_index": sentence_index,
                "full_sentence_text": full_sentence_text,
                "chunks": chunks,
                "alignment_tokens": alignment_tokens,
            })

    return jobs


# Selection is deterministic and spread across levels. Reusing the same sample makes
# before/after reports comparable and avoids a random easy or difficult subset hiding
# a pipeline regression.
def select_sentence_jobs(jobs):
    if sentence_sample_count <= 0:
        sys.exit("sentence_sample_count は1以上である必要があります。")

    if sentence_sample_count >= len(jobs):
        selected_jobs = list(jobs)
    else:
        random_generator = random.Random(selection_seed)
        jobs_by_level = {}

        for job in jobs:
            jobs_by_level.setdefault(job["level_id"], []).append(job)

        selected_jobs = []
        selected_keys = set()
        level_ids = sorted(jobs_by_level)
        base_count = sentence_sample_count // len(level_ids)
        extra_count = sentence_sample_count % len(level_ids)

        for level_index, level_id in enumerate(level_ids):
            level_count = base_count + (1 if level_index < extra_count else 0)

            if level_count == 0:
                continue

            level_jobs = list(jobs_by_level[level_id])
            random_generator.shuffle(level_jobs)

            for job in level_jobs[:level_count]:
                selected_jobs.append(job)
                selected_keys.add((job["level_id"], job["sentence_id"]))

        if len(selected_jobs) < sentence_sample_count:
            remaining_jobs = [
                job
                for job in jobs
                if (job["level_id"], job["sentence_id"]) not in selected_keys
            ]
            random_generator.shuffle(remaining_jobs)
            selected_jobs.extend(
                remaining_jobs[:sentence_sample_count - len(selected_jobs)]
            )

    selected_jobs = sorted(
        selected_jobs,
        key=lambda job: (job["level_id"], job["sentence_index"]),
    )

    for sentence_sample_number, job in enumerate(selected_jobs, start=1):
        job["sentence_sample_number"] = sentence_sample_number
        job["file_stem"] = f"sample-{sentence_sample_number:04d}"

    return selected_jobs


def validate_gpu():
    if not torch.cuda.is_available():
        sys.exit("CUDA対応GPUが見つかりません。")

    device_count = torch.cuda.device_count()

    if gpu_index < 0 or gpu_index >= device_count:
        sys.exit(
            f"gpu_index が範囲外です: {gpu_index} "
            f"(利用可能GPU数: {device_count})"
        )

    torch.cuda.set_device(gpu_index)


# Qwen and Whisper are intentionally loaded in separate stages rather than competing
# for VRAM. Garbage collection, synchronization, and cache release are all needed
# before the second model is loaded; deleting only the local variable is insufficient.
def clear_cuda_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu_index)
        torch.cuda.empty_cache()


def load_qwen_model():
    validate_gpu()
    print(f"GPU: {torch.cuda.get_device_name(gpu_index)}")
    print(f"TTSモデル: {qwen_model_name}")

    return Qwen3TTSModel.from_pretrained(
        qwen_model_name,
        device_map=f"cuda:{gpu_index}",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


# Reject unexpected channel layouts, invalid sample rates, and non-finite samples at
# the model boundary. Flattening arbitrary shapes or forwarding NaNs could create
# corrupt WAV files that fail much later and are harder to attribute to Qwen.
def normalize_generated_waveform(waveform, sample_rate, text):
    waveform = numpy.asarray(waveform)

    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    elif waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    elif waveform.ndim != 1:
        sys.exit(
            f"生成音声の形状がモノラルではありません: {text} / {waveform.shape}"
        )

    waveform = numpy.asarray(waveform, dtype=numpy.float32).copy()
    sample_rate = int(sample_rate)

    if sample_rate <= 0:
        sys.exit(f"生成音声のサンプルレートが正しくありません: {text} / {sample_rate}")

    if waveform.size == 0:
        sys.exit(f"空の音声が生成されました: {text}")

    if not numpy.isfinite(waveform).all():
        sys.exit(f"生成音声にNaNまたは無限大が含まれています: {text}")

    return waveform, sample_rate


def generate_sentence_audio(model, text):
    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(
            text=[text],
            language=["Japanese"],
            speaker=[speaker_name],
            max_new_tokens=2048,
        )

    if len(wavs) != 1:
        sys.exit(f"生成された音声数が一致しません: {text}")

    return normalize_generated_waveform(wavs[0], sample_rate, text)


def write_wav(ffmpeg_executable, path, waveform, sample_rate):
    audio_bytes = waveform.astype("<f4", copy=False).tobytes()
    result = subprocess.run(
        [
            str(ffmpeg_executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(
            f"WAVファイルの作成に失敗しました: {path}\n"
            f"終了コード: {result.returncode}\n{error_message}"
        )

    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"WAVファイルの作成に失敗しました: {path}")

    with wave.open(str(path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        wav_sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()

    if channel_count != 1:
        sys.exit(f"MFA用WAVがモノラルではありません: {path} / {channel_count}ch")

    if wav_sample_rate != sample_rate:
        sys.exit(
            f"MFA用WAVのサンプルレートが一致しません: "
            f"{path} / expected={sample_rate} actual={wav_sample_rate}"
        )

    if sample_width != 2:
        sys.exit(f"MFA用WAVが16-bit PCMではありません: {path}")

    if frame_count != waveform.size:
        sys.exit(
            f"MFA用WAVのサンプル数が一致しません: "
            f"{path} / expected={waveform.size} actual={frame_count}"
        )


# MFA requires ordinary audio and transcript files, but they are implementation
# details of this experiment. They are written only inside TemporaryDirectory so the
# CSV remains the sole persistent output.
def write_alignment_corpus(ffmpeg_executable, corpus_directory, selected_jobs, model):
    generated_sentences = []

    for job in tqdm(
        selected_jobs,
        desc="全文音声を生成しています",
        unit="文",
        dynamic_ncols=True,
    ):
        waveform, sample_rate = generate_sentence_audio(
            model,
            job["full_sentence_text"],
        )
        wav_path = corpus_directory / f"{job['file_stem']}.wav"
        lab_path = corpus_directory / f"{job['file_stem']}.lab"

        write_wav(ffmpeg_executable, wav_path, waveform, sample_rate)
        lab_path.write_text(
            " ".join(job["alignment_tokens"]) + "\n",
            encoding="utf-8",
        )

        generated_sentences.append({
            "job": job,
            "waveform": waveform,
            "sample_rate": sample_rate,
        })

    return generated_sentences


# Align the selected sentences as one corpus. Repeated align_one_hf calls would reload
# MFA and its acoustic resources for every sentence, greatly increasing runtime and
# creating more opportunities for partial or stale outputs.
def run_mfa_corpus_alignment(
    mfa_executable,
    mfa_environment,
    corpus_directory,
    aligned_directory,
    mfa_temporary_directory,
):
    # All generated speech uses the same Qwen speaker, so MFA can safely treat this as
    # a single-speaker corpus. The explicit curriculum spaces are preserved by disabling
    # automatic tokenization, and G2P covers words absent from the bundled dictionary.
    command = [
        mfa_executable,
        "align_hf",
        "--output_format",
        "json",
        "--use_g2p",
        "--no_tokenization",
        "--temporary_directory",
        mfa_temporary_directory,
        "--num_jobs",
        str(mfa_num_jobs),
        "--clean",
        "--final_clean",
        "--overwrite",
        "--single_speaker",
        corpus_directory,
        mfa_model_name,
        aligned_directory,
    ]

    print("MFAで全文音声を一括整列しています。")
    run_text_command(
        command,
        "MFAの一括整列に失敗しました。",
        mfa_environment,
    )


# Require a one-to-one filename match. Falling back to whichever JSON happens to be
# present can silently pair one sentence with another sentence's timestamps, which is
# more dangerous than a clean failure because the resulting CSV still looks valid.
def find_alignment_path(aligned_directory, file_stem):
    matching_paths = sorted(aligned_directory.rglob(f"{file_stem}.json"))

    if len(matching_paths) != 1:
        matches_text = "\n".join(str(path) for path in matching_paths)
        sys.exit(
            f"MFA整列結果を一意に特定できません: {file_stem}\n"
            f"一致数: {len(matching_paths)}\n{matches_text}"
        )

    return matching_paths[0]


def parse_word_entries(alignment_path):
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    tiers = alignment.get("tiers")

    if not isinstance(tiers, dict):
        sys.exit(f"MFA整列結果に tiers がありません: {alignment_path}")

    word_tiers = []

    for tier_name, tier_data in tiers.items():
        if tier_name == "words" or tier_name.endswith(" - words"):
            word_tiers.append((tier_name, tier_data))

    if len(word_tiers) != 1:
        tier_names = ", ".join(name for name, _ in word_tiers)
        sys.exit(
            f"MFA整列結果の words tier を一意に特定できません: "
            f"{alignment_path} / {tier_names}"
        )

    tier_name, tier_data = word_tiers[0]
    entries = tier_data.get("entries")

    if not isinstance(entries, list):
        sys.exit(f"MFA整列結果の entries が配列ではありません: {alignment_path}")

    word_entries = []

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 3:
            sys.exit(
                f"MFA整列結果の項目形式が正しくありません: "
                f"{alignment_path} / {tier_name} / {entry_index}"
            )

        start_seconds = float(entry[0])
        end_seconds = float(entry[1])
        label = normalize_text(entry[2]).strip()

        if not label:
            continue

        word_entries.append({
            "start": start_seconds,
            "end": end_seconds,
            "label": label,
        })

    return word_entries


# Alignment is all-or-nothing for chunk extraction. Japanese sentences often repeat
# particles such as に, の, and は, so a missing token cannot be safely repaired by
# searching for matching text; every following positional boundary could be shifted.
def validate_alignment(job, word_entries, waveform, sample_rate):
    expected_tokens = job["alignment_tokens"]
    actual_labels = [entry["label"] for entry in word_entries]
    errors = []

    if len(word_entries) != len(expected_tokens):
        errors.append(
            "chunk_count_mismatch:"
            f" expected={len(expected_tokens)} actual={len(word_entries)}"
        )

    normalized_expected = [normalize_text(token) for token in expected_tokens]
    normalized_actual = [normalize_text(label) for label in actual_labels]

    if normalized_actual != normalized_expected:
        errors.append(
            "label_sequence_mismatch:"
            f" expected={json.dumps(expected_tokens, ensure_ascii=False)}"
            f" actual={json.dumps(actual_labels, ensure_ascii=False)}"
        )

    previous_end_sample = 0

    for entry_index, entry in enumerate(word_entries):
        start_seconds = entry["start"]
        end_seconds = entry["end"]

        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            errors.append(f"non_finite_timestamp: index={entry_index}")
            continue

        start_sample = round(start_seconds * sample_rate)
        end_sample = round(end_seconds * sample_rate)

        if start_seconds < 0 or start_sample < 0:
            errors.append(
                f"negative_start: index={entry_index} value={start_seconds}"
            )

        if end_seconds <= start_seconds or end_sample <= start_sample:
            errors.append(
                f"invalid_range: index={entry_index} "
                f"start={start_seconds} end={end_seconds}"
            )

        if end_sample > waveform.size:
            errors.append(
                f"range_exceeds_audio: index={entry_index} "
                f"end_sample={end_sample} audio_samples={waveform.size}"
            )

        if start_sample < previous_end_sample:
            errors.append(
                f"overlapping_ranges: index={entry_index} "
                f"start_sample={start_sample} previous_end_sample={previous_end_sample}"
            )

        previous_end_sample = max(previous_end_sample, end_sample)

    return "; ".join(errors), actual_labels


# Do not clamp malformed timestamps into the source waveform. Clamping would turn an
# invalid alignment into plausible but incorrectly labelled audio and contaminate the
# report used to judge TTS quality.
def extract_clip(waveform, sample_rate, start_seconds, end_seconds):
    start_sample = round(start_seconds * sample_rate)
    end_sample = round(end_seconds * sample_rate)

    if start_sample < 0 or end_sample <= start_sample or end_sample > waveform.size:
        sys.exit(
            "検証済みの整列範囲から音声を抽出できません: "
            f"start={start_seconds} end={end_seconds} "
            f"sample_rate={sample_rate} audio_samples={waveform.size}"
        )

    return waveform[start_sample:end_sample].copy()


# Full-sentence rows remain useful even when alignment fails. Chunk waveforms are only
# attached after the complete alignment passes validation, preventing Whisper output
# from being attributed to the wrong expected chunk. This is report-integrity logic,
# not frontend playback suppression.
def create_audio_record(
    generated_sentence,
    row_type,
    chunk_index,
    word_entry,
    alignment_valid,
    alignment_error,
    alignment_labels,
):
    job = generated_sentence["job"]
    waveform = generated_sentence["waveform"]
    sample_rate = generated_sentence["sample_rate"]
    chunks = job["chunks"]
    alignment_tokens = job["alignment_tokens"]
    source_duration = waveform.size / sample_rate

    if row_type == "sentence":
        expected_text = job["full_sentence_text"]
        clip_waveform = waveform.copy()
        alignment_token = ""
        alignment_label = ""
        alignment_start = ""
        alignment_end = ""
        previous_chunk = ""
        next_chunk = ""
    else:
        expected_text = chunks[chunk_index]
        alignment_token = alignment_tokens[chunk_index]
        alignment_label = word_entry["label"] if word_entry else ""
        alignment_start = word_entry["start"] if word_entry else ""
        alignment_end = word_entry["end"] if word_entry else ""
        previous_chunk = chunks[chunk_index - 1] if chunk_index > 0 else ""
        next_chunk = chunks[chunk_index + 1] if chunk_index + 1 < len(chunks) else ""

        if alignment_valid and word_entry is not None:
            clip_waveform = extract_clip(
                waveform,
                sample_rate,
                word_entry["start"],
                word_entry["end"],
            )
        else:
            clip_waveform = numpy.zeros(0, dtype=numpy.float32)

    return {
        "sentence_sample_number": job["sentence_sample_number"],
        "row_type": row_type,
        "level_id": job["level_id"],
        "sentence_id": job["sentence_id"],
        "sentence_index": job["sentence_index"],
        "chunk_index": chunk_index if row_type == "chunk" else "",
        "full_sentence_text": job["full_sentence_text"],
        "expected_text": expected_text,
        "alignment_token": alignment_token,
        "alignment_label": alignment_label,
        "alignment_valid": alignment_valid,
        "alignment_error": alignment_error,
        "alignment_labels": alignment_labels,
        "expected_chunk_count": len(alignment_tokens),
        "aligned_word_count": len(alignment_labels),
        "previous_chunk": previous_chunk,
        "next_chunk": next_chunk,
        "alignment_start_seconds": alignment_start,
        "alignment_end_seconds": alignment_end,
        "clip_duration_seconds": clip_waveform.size / sample_rate if clip_waveform.size else 0,
        "source_sentence_duration_seconds": source_duration,
        "audio_sample_rate": sample_rate,
        "waveform": clip_waveform,
    }


# The temporary workspace contains every intermediate WAV, LAB, MFA database, and JSON
# result. Leaving the scope removes all of it automatically after the in-memory records
# have been built.
def create_alignment_records(
    model,
    selected_jobs,
    ffmpeg_executable,
    mfa_executable,
    mfa_environment,
):
    records = []

    with tempfile.TemporaryDirectory(prefix="ja_alignment_test_") as temporary_name:
        temporary_directory = Path(temporary_name)
        corpus_directory = temporary_directory / "corpus"
        aligned_directory = temporary_directory / "aligned"
        mfa_temporary_directory = temporary_directory / "mfa"
        corpus_directory.mkdir()
        aligned_directory.mkdir()
        mfa_temporary_directory.mkdir()

        generated_sentences = write_alignment_corpus(
            ffmpeg_executable,
            corpus_directory,
            selected_jobs,
            model,
        )
        run_mfa_corpus_alignment(
            mfa_executable,
            mfa_environment,
            corpus_directory,
            aligned_directory,
            mfa_temporary_directory,
        )

        for generated_sentence in tqdm(
            generated_sentences,
            desc="整列結果を検証しています",
            unit="文",
            dynamic_ncols=True,
        ):
            job = generated_sentence["job"]
            alignment_path = find_alignment_path(
                aligned_directory,
                job["file_stem"],
            )
            word_entries = parse_word_entries(alignment_path)
            alignment_error, alignment_labels = validate_alignment(
                job,
                word_entries,
                generated_sentence["waveform"],
                generated_sentence["sample_rate"],
            )
            alignment_valid = not alignment_error

            records.append(
                create_audio_record(
                    generated_sentence,
                    "sentence",
                    "",
                    None,
                    alignment_valid,
                    alignment_error,
                    alignment_labels,
                )
            )

            for chunk_index in range(len(job["chunks"])):
                word_entry = (
                    word_entries[chunk_index]
                    if chunk_index < len(word_entries)
                    else None
                )
                records.append(
                    create_audio_record(
                        generated_sentence,
                        "chunk",
                        chunk_index,
                        word_entry,
                        alignment_valid,
                        alignment_error,
                        alignment_labels,
                    )
                )

    return records


def resample_for_whisper(ffmpeg_executable, waveform, sample_rate):
    if sample_rate == whisper_sample_rate:
        whisper_waveform = waveform.astype(numpy.float32, copy=False)
    else:
        audio_bytes = waveform.astype("<f4", copy=False).tobytes()
        result = subprocess.run(
            [
                str(ffmpeg_executable),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-ar",
                str(whisper_sample_rate),
                "-ac",
                "1",
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            error_message = result.stderr.decode("utf-8", errors="replace").strip()
            sys.exit(
                "Whisper用のリサンプリングに失敗しました。\n"
                f"終了コード: {result.returncode}\n{error_message}"
            )

        whisper_waveform = numpy.frombuffer(result.stdout, dtype="<f4").copy()

    if whisper_waveform.size == 0:
        sys.exit("Whisper用のリサンプリング結果が空です。")

    if not numpy.isfinite(whisper_waveform).all():
        sys.exit("Whisper用の音声にNaNまたは無限大が含まれています。")

    return whisper_waveform


def load_whisper_model():
    validate_gpu()
    print(f"Whisperモデル: {whisper_model_name}")
    return whisper.load_model(
        whisper_model_name,
        device=f"cuda:{gpu_index}",
    )


def segment_float(segment, key):
    value = segment.get(key)

    if value is None:
        return 0.0

    return float(value)


# Whisper is deliberately used as a neutral measurement layer: Japanese transcription,
# no translation, and no previous-clip conditioning. The usual suppression thresholds
# remain disabled to preserve raw output for tiny clips and to keep this report directly
# comparable with the earlier verification run; the script itself does not grade it.
def transcribe_clip(model, ffmpeg_executable, waveform, sample_rate):
    if waveform.size == 0:
        return "", []

    whisper_waveform = resample_for_whisper(
        ffmpeg_executable,
        waveform,
        sample_rate,
    )
    result = model.transcribe(
        whisper_waveform,
        language=whisper_language,
        task="transcribe",
        verbose=None,
        temperature=0.0,
        beam_size=whisper_beam_size,
        condition_on_previous_text=False,
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        fp16=True,
    )

    segments = result.get("segments", [])
    segment_rows = []

    for segment in segments:
        segment_rows.append({
            "start": round(segment_float(segment, "start"), 4),
            "end": round(segment_float(segment, "end"), 4),
            "text": segment.get("text", ""),
            "avg_logprob": round(segment_float(segment, "avg_logprob"), 6),
            "no_speech_prob": round(segment_float(segment, "no_speech_prob"), 6),
            "compression_ratio": round(segment_float(segment, "compression_ratio"), 6),
            "temperature": round(segment_float(segment, "temperature"), 6),
        })

    return result.get("text", "").strip(), segment_rows


def get_segment_average(segment_rows, key):
    if not segment_rows:
        return ""

    return round(
        sum(segment[key] for segment in segment_rows) / len(segment_rows),
        6,
    )


def get_segment_maximum(segment_rows, key):
    if not segment_rows:
        return ""

    return round(max(segment[key] for segment in segment_rows), 6)


def get_audio_rms(waveform):
    if waveform.size == 0:
        return ""

    waveform_float64 = waveform.astype(numpy.float64)
    return round(float(numpy.sqrt(numpy.mean(numpy.square(waveform_float64)))), 8)


def get_audio_peak(waveform):
    if waveform.size == 0:
        return ""

    return round(float(numpy.max(numpy.abs(waveform))), 8)


def create_report_row(
    report_row_number,
    record,
    whisper_text,
    segment_rows,
    mfa_version,
):
    return {
        "report_row_number": report_row_number,
        "sentence_sample_number": record["sentence_sample_number"],
        "row_type": record["row_type"],
        "level_id": record["level_id"],
        "sentence_id": record["sentence_id"],
        "sentence_index": record["sentence_index"],
        "chunk_index": record["chunk_index"],
        "full_sentence_text": record["full_sentence_text"],
        "expected_text": record["expected_text"],
        "alignment_token": record["alignment_token"],
        "alignment_label": record["alignment_label"],
        "alignment_valid": "yes" if record["alignment_valid"] else "no",
        "alignment_error": record["alignment_error"],
        "alignment_labels_json": json.dumps(
            record["alignment_labels"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "expected_chunk_count": record["expected_chunk_count"],
        "aligned_word_count": record["aligned_word_count"],
        "previous_chunk": record["previous_chunk"],
        "next_chunk": record["next_chunk"],
        "alignment_start_seconds": record["alignment_start_seconds"],
        "alignment_end_seconds": record["alignment_end_seconds"],
        "clip_duration_seconds": round(record["clip_duration_seconds"], 8),
        "source_sentence_duration_seconds": round(
            record["source_sentence_duration_seconds"],
            8,
        ),
        "whisper_text": whisper_text,
        "whisper_model": whisper_model_name,
        "whisper_language": whisper_language,
        "whisper_segment_count": len(segment_rows),
        "whisper_average_log_probability": get_segment_average(
            segment_rows,
            "avg_logprob",
        ),
        "whisper_max_no_speech_probability": get_segment_maximum(
            segment_rows,
            "no_speech_prob",
        ),
        "whisper_max_compression_ratio": get_segment_maximum(
            segment_rows,
            "compression_ratio",
        ),
        "audio_sample_rate": record["audio_sample_rate"],
        "audio_rms": get_audio_rms(record["waveform"]),
        "audio_peak": get_audio_peak(record["waveform"]),
        "whisper_segments_json": json.dumps(
            segment_rows,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "tts_model": qwen_model_name,
        "tts_speaker": speaker_name,
        "alignment_model": mfa_model_name,
        "alignment_tool_version": mfa_version,
        "selection_seed": selection_seed,
        "tts_seed": tts_seed,
    }


# Build the report beside the destination and replace it only after every row succeeds.
# A failed run therefore cannot leave a half-written CSV that might be mistaken for a
# completed quality report.
def write_report(records, model, ffmpeg_executable, mfa_version):
    with tempfile.TemporaryDirectory(
        prefix=".alignment_report_",
        dir=script_directory,
    ) as temporary_report_directory_name:
        temporary_report_path = (
            Path(temporary_report_directory_name)
            / report_path.name
        )

        with temporary_report_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as report_file:
            writer = csv.DictWriter(report_file, fieldnames=report_fields)
            writer.writeheader()

            for report_row_number, record in enumerate(
                tqdm(
                    records,
                    desc="Whisperで文字起こししています",
                    unit="件",
                    dynamic_ncols=True,
                ),
                start=1,
            ):
                whisper_text, segment_rows = transcribe_clip(
                    model,
                    ffmpeg_executable,
                    record["waveform"],
                    record["audio_sample_rate"],
                )
                writer.writerow(
                    create_report_row(
                        report_row_number,
                        record,
                        whisper_text,
                        segment_rows,
                        mfa_version,
                    )
                )
                report_file.flush()

        temporary_report_path.replace(report_path)


# The pipeline is staged deliberately: validate dependencies, generate contextual
# sentence audio, align and extract chunks, release Qwen, then load Whisper and write
# the single final CSV. Keeping those phases separate protects both VRAM and provenance.
def main():
    ffmpeg_executable, mfa_executable, mfa_environment, mfa_version = check_tools()
    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    levels = load_levels()
    jobs = collect_sentence_jobs(levels)
    selected_jobs = select_sentence_jobs(jobs)

    if not selected_jobs:
        sys.exit("検証する文がありません。")

    total_chunk_count = sum(len(job["chunks"]) for job in selected_jobs)
    total_report_rows = len(selected_jobs) + total_chunk_count
    print(
        f"検証対象: {len(selected_jobs)}文 / "
        f"{total_chunk_count}チャンク / {total_report_rows}レポート行"
    )

    torch.manual_seed(tts_seed)
    torch.cuda.manual_seed_all(tts_seed)

    qwen_model = load_qwen_model()
    records = create_alignment_records(
        qwen_model,
        selected_jobs,
        ffmpeg_executable,
        mfa_executable,
        mfa_environment,
    )

    del qwen_model
    clear_cuda_memory()

    whisper_model = load_whisper_model()
    write_report(
        records,
        whisper_model,
        ffmpeg_executable,
        mfa_version,
    )

    del whisper_model
    clear_cuda_memory()

    valid_chunk_count = sum(
        1
        for record in records
        if record["row_type"] == "chunk" and record["alignment_valid"]
    )
    invalid_sentence_count = sum(
        1
        for record in records
        if record["row_type"] == "sentence" and not record["alignment_valid"]
    )

    print(
        f"完了: {len(selected_jobs)}文から{len(records)}件の音声を検証しました。"
    )
    print(
        f"有効な整列チャンク: {valid_chunk_count}件 / "
        f"整列エラーのある文: {invalid_sentence_count}件"
    )
    print(f"CSVレポート: {report_path}")


if __name__ == "__main__":
    main()
