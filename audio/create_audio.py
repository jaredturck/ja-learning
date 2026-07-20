from pathlib import Path
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import unicodedata

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import flash_attn
import json5
import numpy
import torch
from accelerate import Accelerator
from huggingface_hub.utils import disable_progress_bars
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
levels_path = project_directory / "src" / "levels.ts"
public_directory = project_directory / "public"
output_directory = public_directory / "audio"
staging_output_directory = public_directory / ".audio_staging"
backup_output_directory = public_directory / ".audio_backup"
work_directory = Path("/tmp/ja_learning_create_audio_work")

ffmpeg_executable = Path("/usr/bin/ffmpeg")
mfa_executable = Path("/home/jared/miniforge3/envs/mfa/bin/mfa")
mfa_environment_directory = mfa_executable.parent.parent

levels_start_marker = "/* AUDIO_LEVELS_START */"
levels_end_marker = "/* AUDIO_LEVELS_END */"
model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
speaker_name = "Ono_Anna"
voice_instruction = "入力文そのまま"
mfa_model_name = "japanese_mfa"
tts_sample_rate = 24000
silence_seconds = 0.2
opus_bitrate = "96k"
max_batch_size = 24
max_batch_cost = 384
mfa_num_jobs = 4
alignment_attempt_count = 3
alignment_punctuation = " \t\n\r\"'`“”‘’「」『』（）()［］[]【】。、，．,.！？!?・:：;；…"

# accelerate launch --multi_gpu --num_processes 2 audio/create_audio.py


def load_levels():
    levels_source = levels_path.read_text(encoding="utf-8")

    if levels_start_marker not in levels_source or levels_end_marker not in levels_source:
        sys.exit("levels.ts に音声データ用の開始・終了マーカーがありません。")

    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    return json5.loads(levels_text)


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text))


def get_alignment_token(text):
    token = normalize_text(text).strip(alignment_punctuation)

    if not token:
        sys.exit(f"MFA用の空チャンクが見つかりました: {text}")

    if any(character.isspace() for character in token):
        sys.exit(f"MFA用チャンクに空白があります: {text}")

    return token


def add_unique_text(texts, seen_texts, text):
    text = normalize_text(text)

    if not text:
        sys.exit("空の日本語テキストが見つかりました。")

    if text in seen_texts:
        return

    seen_texts.add(text)
    texts.append(text)


def create_alignment_job(job_id, level_id, sentence_id, sentence_index, chunks):
    normalized_chunks = [normalize_text(chunk) for chunk in chunks]

    return {
        "job_id": job_id,
        "file_stem": job_id,
        "level_id": str(level_id),
        "sentence_id": str(sentence_id),
        "sentence_index": sentence_index,
        "full_sentence_text": "".join(normalized_chunks),
        "chunks": normalized_chunks,
        "alignment_tokens": [
            get_alignment_token(chunk)
            for chunk in normalized_chunks
        ],
    }


def collect_audio_plan(levels):
    level_texts = {}
    sentence_jobs = []
    sentence_texts = set()
    real_chunk_texts = set()
    distractor_sources = {}

    for level in levels:
        level_id = str(level["id"])
        texts = []
        seen_texts = set()

        for sentence_index, sentence in enumerate(level["sentences"]):
            chunks = [
                normalize_text(chunk["japanese"])
                for chunk in sentence["chunks"]
            ]
            sentence_job = create_alignment_job(
                f"sentence-{len(sentence_jobs) + 1:04d}",
                level_id,
                sentence["id"],
                sentence_index,
                chunks,
            )
            sentence_jobs.append(sentence_job)
            sentence_texts.add(sentence_job["full_sentence_text"])
            add_unique_text(
                texts,
                seen_texts,
                sentence_job["full_sentence_text"],
            )

            for chunk_index, chunk in enumerate(sentence["chunks"]):
                chunk_text = chunks[chunk_index]
                real_chunk_texts.add(chunk_text)
                add_unique_text(texts, seen_texts, chunk_text)

                for distractor in chunk["distractors"]:
                    distractor_text = normalize_text(distractor)
                    add_unique_text(texts, seen_texts, distractor_text)

                    if distractor_text not in distractor_sources:
                        distractor_sources[distractor_text] = {
                            "source_job": sentence_job,
                            "chunk_index": chunk_index,
                        }

        level_texts[level_id] = texts

    reusable_texts = sentence_texts | real_chunk_texts
    distractor_jobs = []

    for distractor_text, source in distractor_sources.items():
        if distractor_text in reusable_texts:
            continue

        source_job = source["source_job"]
        target_chunk_index = source["chunk_index"]
        synthetic_chunks = list(source_job["chunks"])
        synthetic_chunks[target_chunk_index] = distractor_text
        distractor_job = create_alignment_job(
            f"distractor-{len(distractor_jobs) + 1:04d}",
            source_job["level_id"],
            source_job["sentence_id"],
            source_job["sentence_index"],
            synthetic_chunks,
        )
        distractor_job["target_text"] = distractor_text
        distractor_job["target_chunk_index"] = target_chunk_index
        distractor_jobs.append(distractor_job)

    required_texts = {
        text
        for texts in level_texts.values()
        for text in texts
    }

    return level_texts, sentence_jobs, distractor_jobs, required_texts


def load_model(accelerator):
    if not torch.cuda.is_available():
        sys.exit("CUDA対応GPUが見つかりません。")

    torch.cuda.set_device(accelerator.device)

    if accelerator.is_main_process:
        print(f"使用GPU数: {accelerator.num_processes}")

        for gpu_index in range(accelerator.num_processes):
            print(f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")

        print(f"FlashAttention: {flash_attn.__version__}")
        print(f"最大バッチサイズ: {max_batch_size}")
        print(f"モデルを読み込んでいます: {model_name}")

    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=str(accelerator.device),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def normalize_generated_waveform(waveform, sample_rate, text):
    waveform = numpy.asarray(waveform).squeeze()

    if waveform.ndim != 1:
        sys.exit(f"生成音声がモノラルではありません: {text} / {waveform.shape}")

    waveform = numpy.asarray(waveform, dtype=numpy.float32).copy()
    sample_rate = int(sample_rate)

    if sample_rate != tts_sample_rate:
        sys.exit(
            f"生成音声のサンプルレートが一致しません: "
            f"{text} / expected={tts_sample_rate} actual={sample_rate}"
        )

    if waveform.size == 0 or not numpy.isfinite(waveform).all():
        sys.exit(f"生成音声が無効です: {text}")

    return waveform


def generate_audio_batch(model, jobs):
    texts = [job["full_sentence_text"] for job in jobs]

    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(
            text=texts,
            language=["Japanese"] * len(texts),
            speaker=[speaker_name] * len(texts),
            instruct=[voice_instruction] * len(texts),
            max_new_tokens=2048,
        )

    if len(wavs) != len(jobs):
        sys.exit("入力した日本語文数と生成された音声数が一致しません。")

    normalized_wavs = [
        normalize_generated_waveform(waveform, sample_rate, text)
        for text, waveform in zip(texts, wavs, strict=True)
    ]

    return normalized_wavs


def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def try_generate_audio_batch(model, jobs):
    try:
        return generate_audio_batch(model, jobs)
    except torch.OutOfMemoryError:
        return None


def generate_audio_batch_with_retry(model, jobs):
    wavs = try_generate_audio_batch(model, jobs)

    if wavs is not None:
        return [(jobs, wavs)]

    clear_cuda_memory()

    if len(jobs) == 1:
        sys.exit(
            "1文の音声生成でもGPUメモリが不足しました: "
            f"{jobs[0]['full_sentence_text']}"
        )

    middle = len(jobs) // 2
    left_results = generate_audio_batch_with_retry(model, jobs[:middle])
    clear_cuda_memory()
    right_results = generate_audio_batch_with_retry(model, jobs[middle:])
    clear_cuda_memory()

    return [*left_results, *right_results]


def create_audio_batches(jobs, force_single):
    sorted_jobs = sorted(
        jobs,
        key=lambda job: (
            len(job["full_sentence_text"]),
            job["job_id"],
        ),
    )

    if force_single:
        return [[job] for job in sorted_jobs]

    batches = []
    current_batch = []
    current_max_length = 0

    for job in sorted_jobs:
        text_length = len(job["full_sentence_text"])
        next_batch_size = len(current_batch) + 1
        next_max_length = max(current_max_length, text_length)
        next_batch_cost = next_batch_size * next_max_length

        if current_batch and (
            next_batch_size > max_batch_size
            or next_batch_cost > max_batch_cost
        ):
            batches.append(current_batch)
            current_batch = []
            current_max_length = 0

        current_batch.append(job)
        current_max_length = max(current_max_length, text_length)

    if current_batch:
        batches.append(current_batch)

    return batches


def write_wav(path, waveform):
    subprocess.run(
        [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(tts_sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_s16le",
            path,
        ],
        input=waveform.astype("<f4", copy=False).tobytes(),
        stdout=subprocess.DEVNULL,
        check=True,
    )


def write_generated_job(job, waveform, corpus_directory, data_directory):
    file_stem = job["file_stem"]
    numpy.save(data_directory / f"{file_stem}.npy", waveform)
    write_wav(corpus_directory / f"{file_stem}.wav", waveform)
    (corpus_directory / f"{file_stem}.lab").write_text(
        " ".join(job["alignment_tokens"]) + "\n",
        encoding="utf-8",
    )


def generate_local_job_files(
    model,
    jobs,
    attempt_directory,
    accelerator,
    force_single,
):
    batches = create_audio_batches(jobs, force_single)
    local_batches = batches[
        accelerator.process_index::accelerator.num_processes
    ]
    corpus_directory = attempt_directory / "corpus"
    data_directory = attempt_directory / "data"

    for batch_jobs in tqdm(
        local_batches,
        desc=f"GPU {accelerator.local_process_index}",
        unit="バッチ",
        position=accelerator.local_process_index,
        dynamic_ncols=True,
    ):
        batch_results = generate_audio_batch_with_retry(model, batch_jobs)

        for result_jobs, wavs in batch_results:
            for job, waveform in zip(result_jobs, wavs, strict=True):
                write_generated_job(
                    job,
                    waveform,
                    corpus_directory,
                    data_directory,
                )


def reset_directory(path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)


def prepare_attempt_directory(attempt_directory):
    reset_directory(attempt_directory)
    (attempt_directory / "corpus").mkdir()
    (attempt_directory / "data").mkdir()
    (attempt_directory / "aligned").mkdir()
    (attempt_directory / "mfa").mkdir()


def create_mfa_environment():
    environment = os.environ.copy()
    mfa_bin_directory = mfa_environment_directory / "bin"
    current_path = environment.get("PATH", "")

    environment["PATH"] = f"{mfa_bin_directory}{os.pathsep}{current_path}"
    environment["CONDA_PREFIX"] = str(mfa_environment_directory)
    environment["CONDA_DEFAULT_ENV"] = mfa_environment_directory.name

    for variable_name in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"]:
        environment.pop(variable_name, None)

    return environment


def run_mfa_alignment(attempt_directory):
    result = subprocess.run(
        [
            mfa_executable,
            "align_hf",
            "--output_format",
            "json",
            "--use_g2p",
            "--no_tokenization",
            "--temporary_directory",
            attempt_directory / "mfa",
            "--num_jobs",
            str(mfa_num_jobs),
            "--clean",
            "--final_clean",
            "--overwrite",
            "--single_speaker",
            attempt_directory / "corpus",
            mfa_model_name,
            attempt_directory / "aligned",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=create_mfa_environment(),
    )
    status = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (attempt_directory / "mfa_status.json").write_text(
        json.dumps(status, ensure_ascii=False),
        encoding="utf-8",
    )


def check_mfa_status(attempt_directory):
    status = json.loads(
        (attempt_directory / "mfa_status.json").read_text(encoding="utf-8")
    )

    if status["returncode"] != 0:
        sys.exit(
            "MFA整列に失敗しました。\n"
            f"終了コード: {status['returncode']}\n"
            f"stdout:\n{status['stdout']}\n"
            f"stderr:\n{status['stderr']}"
        )


def find_alignment_path(aligned_directory, file_stem):
    matching_paths = list(aligned_directory.rglob(f"{file_stem}.json"))

    if len(matching_paths) != 1:
        return None

    return matching_paths[0]


def parse_word_entries(alignment_path):
    tiers = json.loads(alignment_path.read_text(encoding="utf-8"))["tiers"]
    word_tiers = [
        tier_data
        for tier_name, tier_data in tiers.items()
        if tier_name == "words" or tier_name.endswith(" - words")
    ]

    if len(word_tiers) != 1:
        return []

    word_entries = []

    for entry in word_tiers[0]["entries"]:
        if not isinstance(entry, list) or len(entry) != 3:
            return []

        label = normalize_text(entry[2]).strip()

        if label:
            word_entries.append({
                "start": float(entry[0]),
                "end": float(entry[1]),
                "label": label,
            })

    return word_entries


def validate_alignment(job, word_entries, waveform):
    expected_tokens = job["alignment_tokens"]
    actual_labels = [entry["label"] for entry in word_entries]
    errors = []

    if len(word_entries) != len(expected_tokens):
        errors.append(
            "chunk_count_mismatch:"
            f" expected={len(expected_tokens)} actual={len(word_entries)}"
        )

    if actual_labels != expected_tokens:
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

        start_sample = round(start_seconds * tts_sample_rate)
        end_sample = round(end_seconds * tts_sample_rate)

        if start_sample < 0:
            errors.append(
                f"negative_start: index={entry_index} value={start_seconds}"
            )

        if end_sample <= start_sample:
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
                f"start_sample={start_sample} "
                f"previous_end_sample={previous_end_sample}"
            )

        previous_end_sample = max(previous_end_sample, end_sample)

    return "; ".join(errors)


def resolve_alignment_attempt(
    jobs,
    attempt_directory,
    resolved_data_directory,
    resolved_alignment_directory,
):
    unresolved_job_ids = []
    alignment_errors = {}

    for job in jobs:
        file_stem = job["file_stem"]
        waveform_path = attempt_directory / "data" / f"{file_stem}.npy"
        alignment_path = find_alignment_path(
            attempt_directory / "aligned",
            file_stem,
        )

        if not waveform_path.is_file():
            unresolved_job_ids.append(job["job_id"])
            alignment_errors[job["job_id"]] = "generated_waveform_missing"
            continue

        if alignment_path is None:
            unresolved_job_ids.append(job["job_id"])
            alignment_errors[job["job_id"]] = "alignment_json_missing_or_ambiguous"
            continue

        waveform = numpy.load(waveform_path)
        word_entries = parse_word_entries(alignment_path)
        alignment_error = validate_alignment(job, word_entries, waveform)

        if alignment_error:
            unresolved_job_ids.append(job["job_id"])
            alignment_errors[job["job_id"]] = alignment_error
            continue

        shutil.copy2(
            waveform_path,
            resolved_data_directory / f"{file_stem}.npy",
        )
        shutil.copy2(
            alignment_path,
            resolved_alignment_directory / f"{file_stem}.json",
        )

    (attempt_directory / "unresolved.json").write_text(
        json.dumps(
            {
                "job_ids": unresolved_job_ids,
                "errors": alignment_errors,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def align_jobs_with_retries(jobs, phase_name, model, accelerator):
    phase_directory = work_directory / phase_name
    resolved_data_directory = phase_directory / "resolved_data"
    resolved_alignment_directory = phase_directory / "resolved_alignment"

    if accelerator.is_main_process:
        reset_directory(phase_directory)
        resolved_data_directory.mkdir()
        resolved_alignment_directory.mkdir()

    accelerator.wait_for_everyone()

    unresolved_jobs = list(jobs)
    latest_errors = {}

    for attempt_number in range(1, alignment_attempt_count + 1):
        if not unresolved_jobs:
            break

        attempt_directory = phase_directory / f"attempt-{attempt_number}"

        if accelerator.is_main_process:
            prepare_attempt_directory(attempt_directory)

        accelerator.wait_for_everyone()
        generate_local_job_files(
            model,
            unresolved_jobs,
            attempt_directory,
            accelerator,
            force_single=attempt_number > 1,
        )
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            run_mfa_alignment(attempt_directory)

        accelerator.wait_for_everyone()
        check_mfa_status(attempt_directory)

        if accelerator.is_main_process:
            resolve_alignment_attempt(
                unresolved_jobs,
                attempt_directory,
                resolved_data_directory,
                resolved_alignment_directory,
            )

        accelerator.wait_for_everyone()
        unresolved_result = json.loads(
            (attempt_directory / "unresolved.json").read_text(encoding="utf-8")
        )
        unresolved_job_ids = set(unresolved_result["job_ids"])
        latest_errors = unresolved_result["errors"]
        unresolved_jobs = [
            job
            for job in unresolved_jobs
            if job["job_id"] in unresolved_job_ids
        ]

        if accelerator.is_main_process:
            print(
                f"{phase_name}: 整列試行 {attempt_number} / "
                f"未解決 {len(unresolved_jobs)}件"
            )

    if unresolved_jobs:
        error_lines = [
            f"- {job['job_id']}: {latest_errors.get(job['job_id'], 'unknown')}"
            for job in unresolved_jobs
        ]
        sys.exit(
            f"{phase_name} の整列に{alignment_attempt_count}回失敗しました。\n"
            + "\n".join(error_lines)
        )

    return phase_directory


def extract_clip(waveform, start_seconds, end_seconds):
    start_sample = round(start_seconds * tts_sample_rate)
    end_sample = round(end_seconds * tts_sample_rate)

    if start_sample < 0 or end_sample <= start_sample or end_sample > waveform.size:
        sys.exit(
            "検証済みの整列範囲から音声を抽出できません: "
            f"start={start_seconds} end={end_seconds}"
        )

    return waveform[start_sample:end_sample].copy()


def add_audio_candidate(candidates, text, waveform, order):
    candidates.setdefault(text, []).append({
        "order": order,
        "waveform": waveform,
    })


def choose_median_duration_candidate(candidates):
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["waveform"].size,
            candidate["order"],
        ),
    )
    middle = (len(ordered_candidates) - 1) / 2
    median_size = (
        ordered_candidates[math.floor(middle)]["waveform"].size
        + ordered_candidates[math.ceil(middle)]["waveform"].size
    ) / 2

    selected = min(
        ordered_candidates,
        key=lambda candidate: (
            abs(candidate["waveform"].size - median_size),
            candidate["order"],
        ),
    )

    return selected["waveform"]


def load_resolved_job(phase_directory, job):
    file_stem = job["file_stem"]
    waveform = numpy.load(
        phase_directory / "resolved_data" / f"{file_stem}.npy"
    )
    alignment_path = (
        phase_directory
        / "resolved_alignment"
        / f"{file_stem}.json"
    )
    word_entries = parse_word_entries(alignment_path)
    alignment_error = validate_alignment(job, word_entries, waveform)

    if alignment_error:
        sys.exit(
            f"保存済み整列結果が無効です: {job['job_id']} / {alignment_error}"
        )

    return waveform, word_entries


def build_generated_audio(
    sentence_jobs,
    sentence_phase_directory,
    distractor_jobs,
    distractor_phase_directory,
    required_texts,
):
    sentence_candidates = {}
    real_chunk_candidates = {}
    distractor_candidates = {}
    candidate_order = 0

    for job in sentence_jobs:
        waveform, word_entries = load_resolved_job(
            sentence_phase_directory,
            job,
        )
        add_audio_candidate(
            sentence_candidates,
            job["full_sentence_text"],
            waveform,
            candidate_order,
        )
        candidate_order += 1

        for chunk_text, word_entry in zip(
            job["chunks"],
            word_entries,
            strict=True,
        ):
            add_audio_candidate(
                real_chunk_candidates,
                chunk_text,
                extract_clip(
                    waveform,
                    word_entry["start"],
                    word_entry["end"],
                ),
                candidate_order,
            )
            candidate_order += 1

    for job in distractor_jobs:
        waveform, word_entries = load_resolved_job(
            distractor_phase_directory,
            job,
        )
        target_entry = word_entries[job["target_chunk_index"]]
        add_audio_candidate(
            distractor_candidates,
            job["target_text"],
            extract_clip(
                waveform,
                target_entry["start"],
                target_entry["end"],
            ),
            candidate_order,
        )
        candidate_order += 1

    generated_audio = {}

    for text in sorted(required_texts):
        if text in real_chunk_candidates:
            candidates = real_chunk_candidates[text]
        elif text in sentence_candidates:
            candidates = sentence_candidates[text]
        elif text in distractor_candidates:
            candidates = distractor_candidates[text]
        else:
            sys.exit(f"生成音声候補がありません: {text}")

        generated_audio[text] = choose_median_duration_candidate(candidates)

    return generated_audio


def encode_opus(waveform, output_path):
    subprocess.run(
        [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(tts_sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-application",
            "audio",
            "-b:a",
            opus_bitrate,
            "-vbr",
            "on",
            "-compression_level",
            "10",
            output_path,
        ],
        input=waveform.astype("<f4", copy=False).tobytes(),
        check=True,
    )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        sys.exit(f"Opusファイルの作成に失敗しました: {output_path}")


def build_level_audio(level_id, texts, generated_audio, destination_directory):
    silence = numpy.zeros(
        round(tts_sample_rate * silence_seconds),
        dtype=numpy.float32,
    )
    audio_parts = [silence]
    current_sample = silence.size
    clips = {}

    for text in texts:
        waveform = generated_audio[text]
        start_seconds = current_sample / tts_sample_rate
        duration_seconds = waveform.size / tts_sample_rate

        clips[text] = [round(start_seconds, 8), round(duration_seconds, 8)]
        audio_parts.extend([waveform, silence])
        current_sample += waveform.size + silence.size

    output_path = destination_directory / f"{level_id}.opus"
    encode_opus(numpy.concatenate(audio_parts), output_path)

    if set(clips) != set(texts):
        sys.exit(f"レベル音声のインデックス作成に失敗しました: {level_id}")

    return {
        "file": output_path.name,
        "clips": clips,
    }


def write_index(levels, level_texts, generated_audio, destination_directory):
    index = {
        "version": 1,
        "model": model_name,
        "speaker": speaker_name,
        "levels": {},
    }

    for level in levels:
        level_id = str(level["id"])
        index["levels"][level_id] = build_level_audio(
            level_id,
            level_texts[level_id],
            generated_audio,
            destination_directory,
        )

    index_path = destination_directory / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    expected_files = {
        "index.json",
        *{f"{level['id']}.opus" for level in levels},
    }
    actual_files = {path.name for path in destination_directory.iterdir()}

    if actual_files != expected_files:
        sys.exit(
            "出力ファイル一覧が想定と一致しません。"
            f" expected={sorted(expected_files)} actual={sorted(actual_files)}"
        )


def prepare_output_staging():
    public_directory.mkdir(parents=True, exist_ok=True)

    if backup_output_directory.exists() and not output_directory.exists():
        backup_output_directory.rename(output_directory)

    if staging_output_directory.exists():
        shutil.rmtree(staging_output_directory)

    if backup_output_directory.exists():
        shutil.rmtree(backup_output_directory)

    staging_output_directory.mkdir()


def publish_output():
    if output_directory.exists():
        output_directory.rename(backup_output_directory)

    try:
        staging_output_directory.rename(output_directory)
    except OSError:
        if backup_output_directory.exists() and not output_directory.exists():
            backup_output_directory.rename(output_directory)
        raise

    if backup_output_directory.exists():
        shutil.rmtree(backup_output_directory)


def main():
    accelerator = Accelerator()
    levels = load_levels()
    (
        level_texts,
        sentence_jobs,
        distractor_jobs,
        required_texts,
    ) = collect_audio_plan(levels)

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    if accelerator.is_main_process:
        reset_directory(work_directory)
        prepare_output_staging()
        print(
            f"生成対象: {len(sentence_jobs)}文 / "
            f"{len(distractor_jobs)}件の追加コンテキスト"
        )

    accelerator.wait_for_everyone()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = load_model(accelerator)
    sentence_phase_directory = align_jobs_with_retries(
        sentence_jobs,
        "sentences",
        model,
        accelerator,
    )
    distractor_phase_directory = align_jobs_with_retries(
        distractor_jobs,
        "distractors",
        model,
        accelerator,
    )

    del model
    clear_cuda_memory()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        generated_audio = build_generated_audio(
            sentence_jobs,
            sentence_phase_directory,
            distractor_jobs,
            distractor_phase_directory,
            required_texts,
        )

        if set(generated_audio) != required_texts:
            sys.exit("生成済み音声の一覧が必要な日本語テキストと一致しません。")

        write_index(
            levels,
            level_texts,
            generated_audio,
            staging_output_directory,
        )
        publish_output()
        shutil.rmtree(work_directory)
        print(
            f"完了: {len(generated_audio)}件の音声を"
            f"{len(levels)}個のレベル音声ファイルに書き出しました。"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
