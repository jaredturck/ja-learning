from pathlib import Path
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json5
import numpy
import torch
import whisper
from huggingface_hub.utils import disable_progress_bars
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
levels_path = project_directory / "src" / "levels.ts"
report_path = script_directory / "alignment_verification_report.csv"

levels_start_marker = "/* AUDIO_LEVELS_START */"
levels_end_marker = "/* AUDIO_LEVELS_END */"
qwen_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
speaker_name = "Ono_Anna"
mfa_model_name = "japanese_mfa"
whisper_model_name = "large-v3"
whisper_language = "ja"
whisper_beam_size = 5
sample_count = 100
sample_seed = 0
gpu_index = 0
whisper_sample_rate = 16000

alignment_punctuation = " \t\n\r\"'`“”‘’「」『』（）()［］[]【】。、，．,.！？!?・:：;；…"

report_fields = [
    "sample_number",
    "row_type",
    "level_id",
    "sentence_id",
    "sentence_index",
    "chunk_index",
    "full_sentence_text",
    "expected_text",
    "alignment_token",
    "alignment_label",
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
    "alignment_error",
    "tts_model",
    "tts_speaker",
    "alignment_model",
    "sample_seed",
]


def check_command(command, message):
    result = subprocess.run(
        [command, "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        sys.exit(message)


def check_tools():
    check_command("ffmpeg", "FFmpeg が見つかりません。")
    check_command("mfa", "Montreal Forced Aligner の mfa コマンドが見つかりません。")


def load_levels():
    levels_source = levels_path.read_text(encoding="utf-8")

    if levels_start_marker not in levels_source or levels_end_marker not in levels_source:
        sys.exit("levels.ts に音声データ用の開始・終了マーカーがありません。")

    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    return json5.loads(levels_text)


def get_alignment_token(text):
    token = text.strip(alignment_punctuation)

    if token:
        return token

    return text.strip()


def collect_sentence_jobs(levels):
    jobs = []

    for level in levels:
        for sentence_index, sentence in enumerate(level["sentences"]):
            chunks = [chunk["japanese"] for chunk in sentence["chunks"]]
            alignment_tokens = [get_alignment_token(chunk) for chunk in chunks]
            full_sentence_text = "".join(chunks)

            if not full_sentence_text:
                sys.exit(f"空の日本語文が見つかりました: {level['id']} / {sentence['id']}")

            if any(not token for token in alignment_tokens):
                sys.exit(f"整列できない空のチャンクがあります: {level['id']} / {sentence['id']}")

            jobs.append({
                "level_id": level["id"],
                "sentence_id": sentence["id"],
                "sentence_index": sentence_index,
                "full_sentence_text": full_sentence_text,
                "chunks": chunks,
                "alignment_tokens": alignment_tokens,
            })

    return jobs


def select_sentence_jobs(jobs):
    if sample_count >= len(jobs):
        return list(jobs)

    random_generator = random.Random(sample_seed)
    jobs_by_level = {}

    for job in jobs:
        jobs_by_level.setdefault(job["level_id"], []).append(job)

    selected_jobs = []
    selected_keys = set()
    level_ids = sorted(jobs_by_level)
    base_count = sample_count // len(level_ids)
    extra_count = sample_count % len(level_ids)

    for level_index, level_id in enumerate(level_ids):
        level_count = base_count + (1 if level_index < extra_count else 0)

        if level_count == 0:
            continue

        level_jobs = list(jobs_by_level[level_id])
        random_generator.shuffle(level_jobs)

        for job in level_jobs[:level_count]:
            selected_jobs.append(job)
            selected_keys.add((job["level_id"], job["sentence_id"]))

    if len(selected_jobs) < sample_count:
        remaining_jobs = [
            job
            for job in jobs
            if (job["level_id"], job["sentence_id"]) not in selected_keys
        ]
        random_generator.shuffle(remaining_jobs)
        selected_jobs.extend(remaining_jobs[:sample_count - len(selected_jobs)])

    return sorted(
        selected_jobs,
        key=lambda job: (job["level_id"], job["sentence_index"]),
    )


def clear_cuda_memory():
    torch.cuda.empty_cache()


def load_qwen_model():
    if not torch.cuda.is_available():
        sys.exit("CUDA対応GPUが見つかりません。")

    torch.cuda.set_device(gpu_index)
    print(f"GPU: {torch.cuda.get_device_name(gpu_index)}")
    print(f"TTSモデル: {qwen_model_name}")

    return Qwen3TTSModel.from_pretrained(
        qwen_model_name,
        device_map=f"cuda:{gpu_index}",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def generate_sentence_audio(model, text):
    wavs, sample_rate = model.generate_custom_voice(
        text=[text],
        language=["Japanese"],
        speaker=[speaker_name],
        max_new_tokens=2048,
    )

    if len(wavs) != 1:
        sys.exit(f"生成された音声数が一致しません: {text}")

    waveform = numpy.asarray(wavs[0], dtype=numpy.float32).reshape(-1)

    if waveform.size == 0:
        sys.exit(f"空の音声が生成されました: {text}")

    return waveform, sample_rate


def write_wav(path, waveform, sample_rate):
    audio_bytes = waveform.astype("<f4", copy=False).tobytes()

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
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
        sys.exit(f"WAVファイルの作成に失敗しました: {path}\n{error_message}")

    if not path.exists() or path.stat().st_size == 0:
        sys.exit(f"WAVファイルの作成に失敗しました: {path}")


def run_mfa_alignment(wav_path, lab_path, output_path, temporary_directory):
    result = subprocess.run(
        [
            "mfa",
            "align_one_hf",
            str(wav_path),
            str(lab_path),
            mfa_model_name,
            str(output_path),
            "--output_format",
            "json",
            "--use_g2p",
            "--no_tokenization",
            "--temporary_directory",
            str(temporary_directory),
            "--num_jobs",
            "1",
            "--clean",
            "--final_clean",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(f"MFA整列に失敗しました: {wav_path}\n{error_message}")

    if output_path.exists():
        return output_path

    json_outputs = sorted(output_path.parent.rglob("*.json"))

    if not json_outputs:
        sys.exit(f"MFA整列結果が見つかりません: {wav_path}")

    return json_outputs[0]


def parse_word_entries(alignment_path):
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    tiers = alignment.get("tiers", {})
    entries = None

    for tier_name, tier_data in tiers.items():
        if tier_name == "words" or tier_name.endswith(" - words"):
            entries = tier_data.get("entries", [])
            break

    if entries is None:
        sys.exit(f"MFA整列結果に words tier がありません: {alignment_path}")

    word_entries = []

    for entry in entries:
        if len(entry) != 3:
            continue

        label = str(entry[2]).strip()

        if not label:
            continue

        word_entries.append({
            "start": float(entry[0]),
            "end": float(entry[1]),
            "label": label,
        })

    return word_entries


def extract_clip(waveform, sample_rate, start_seconds, end_seconds):
    start_sample = round(start_seconds * sample_rate)
    end_sample = round(end_seconds * sample_rate)

    if start_sample < 0:
        start_sample = 0

    if end_sample > waveform.size:
        end_sample = waveform.size

    if end_sample <= start_sample:
        return numpy.zeros(0, dtype=numpy.float32)

    return waveform[start_sample:end_sample].copy()


def create_audio_record(job, waveform, sample_rate, row_type, chunk_index, word_entry, alignment_error):
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
        alignment_label = word_entry.get("label", "") if word_entry else ""
        alignment_start = word_entry.get("start", "") if word_entry else ""
        alignment_end = word_entry.get("end", "") if word_entry else ""
        previous_chunk = chunks[chunk_index - 1] if chunk_index > 0 else ""
        next_chunk = chunks[chunk_index + 1] if chunk_index + 1 < len(chunks) else ""

        if word_entry:
            clip_waveform = extract_clip(
                waveform,
                sample_rate,
                word_entry["start"],
                word_entry["end"],
            )
        else:
            clip_waveform = numpy.zeros(0, dtype=numpy.float32)

    return {
        "row_type": row_type,
        "level_id": job["level_id"],
        "sentence_id": job["sentence_id"],
        "sentence_index": job["sentence_index"],
        "chunk_index": chunk_index if row_type == "chunk" else "",
        "full_sentence_text": job["full_sentence_text"],
        "expected_text": expected_text,
        "alignment_token": alignment_token,
        "alignment_label": alignment_label,
        "previous_chunk": previous_chunk,
        "next_chunk": next_chunk,
        "alignment_start_seconds": alignment_start,
        "alignment_end_seconds": alignment_end,
        "clip_duration_seconds": clip_waveform.size / sample_rate if clip_waveform.size else 0,
        "source_sentence_duration_seconds": source_duration,
        "audio_sample_rate": sample_rate,
        "waveform": clip_waveform,
        "alignment_error": alignment_error,
    }


def create_alignment_records(model, selected_jobs):
    records = []

    with tempfile.TemporaryDirectory(prefix="ja_alignment_test_") as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        corpus_directory = temporary_directory / "corpus"
        aligned_directory = temporary_directory / "aligned"
        mfa_temporary_directory = temporary_directory / "mfa"
        corpus_directory.mkdir()
        aligned_directory.mkdir()
        mfa_temporary_directory.mkdir()

        for sample_number, job in enumerate(
            tqdm(
                selected_jobs,
                desc="生成・整列しています",
                unit="文",
                dynamic_ncols=True,
            ),
            start=1,
        ):
            waveform, sample_rate = generate_sentence_audio(
                model,
                job["full_sentence_text"],
            )
            file_stem = f"sample-{sample_number:04d}"
            wav_path = corpus_directory / f"{file_stem}.wav"
            lab_path = corpus_directory / f"{file_stem}.lab"
            output_path = aligned_directory / f"{file_stem}.json"

            write_wav(wav_path, waveform, sample_rate)
            lab_path.write_text(
                " ".join(job["alignment_tokens"]),
                encoding="utf-8",
            )
            alignment_path = run_mfa_alignment(
                wav_path,
                lab_path,
                output_path,
                mfa_temporary_directory,
            )
            word_entries = parse_word_entries(alignment_path)
            alignment_error = ""

            if len(word_entries) != len(job["alignment_tokens"]):
                alignment_error = (
                    "chunk_count_mismatch:"
                    f" expected={len(job['alignment_tokens'])}"
                    f" actual={len(word_entries)}"
                )

            records.append(
                create_audio_record(
                    job,
                    waveform,
                    sample_rate,
                    "sentence",
                    "",
                    None,
                    alignment_error,
                )
            )

            for chunk_index, alignment_token in enumerate(job["alignment_tokens"]):
                word_entry = word_entries[chunk_index] if chunk_index < len(word_entries) else None
                chunk_error = alignment_error

                if word_entry and word_entry["label"] != alignment_token:
                    chunk_error = (
                        f"label_mismatch: expected={alignment_token}"
                        f" actual={word_entry['label']}"
                    )

                records.append(
                    create_audio_record(
                        job,
                        waveform,
                        sample_rate,
                        "chunk",
                        chunk_index,
                        word_entry,
                        chunk_error,
                    )
                )

    return records


def resample_for_whisper(waveform, sample_rate):
    if sample_rate == whisper_sample_rate:
        return waveform.astype(numpy.float32, copy=False)

    audio_bytes = waveform.astype("<f4", copy=False).tobytes()
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
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
        sys.exit(f"Whisper用のリサンプリングに失敗しました。\n{error_message}")

    return numpy.frombuffer(result.stdout, dtype="<f4").copy()


def load_whisper_model():
    if not torch.cuda.is_available():
        sys.exit("CUDA対応GPUが見つかりません。")

    torch.cuda.set_device(gpu_index)
    print(f"Whisperモデル: {whisper_model_name}")
    return whisper.load_model(whisper_model_name, device=f"cuda:{gpu_index}")


def transcribe_clip(model, waveform, sample_rate):
    if waveform.size == 0:
        return "", []

    whisper_waveform = resample_for_whisper(waveform, sample_rate)
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
            "start": round(float(segment.get("start", 0)), 4),
            "end": round(float(segment.get("end", 0)), 4),
            "text": segment.get("text", ""),
            "avg_logprob": round(float(segment.get("avg_logprob", 0)), 6),
            "no_speech_prob": round(float(segment.get("no_speech_prob", 0)), 6),
            "compression_ratio": round(float(segment.get("compression_ratio", 0)), 6),
            "temperature": round(float(segment.get("temperature", 0)), 6),
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

    return round(float(numpy.sqrt(numpy.mean(numpy.square(waveform)))), 8)


def get_audio_peak(waveform):
    if waveform.size == 0:
        return ""

    return round(float(numpy.max(numpy.abs(waveform))), 8)


def create_report_row(sample_number, record, whisper_text, segment_rows):
    return {
        "sample_number": sample_number,
        "row_type": record["row_type"],
        "level_id": record["level_id"],
        "sentence_id": record["sentence_id"],
        "sentence_index": record["sentence_index"],
        "chunk_index": record["chunk_index"],
        "full_sentence_text": record["full_sentence_text"],
        "expected_text": record["expected_text"],
        "alignment_token": record["alignment_token"],
        "alignment_label": record["alignment_label"],
        "previous_chunk": record["previous_chunk"],
        "next_chunk": record["next_chunk"],
        "alignment_start_seconds": record["alignment_start_seconds"],
        "alignment_end_seconds": record["alignment_end_seconds"],
        "clip_duration_seconds": round(record["clip_duration_seconds"], 8),
        "source_sentence_duration_seconds": round(record["source_sentence_duration_seconds"], 8),
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
        "alignment_error": record["alignment_error"],
        "tts_model": qwen_model_name,
        "tts_speaker": speaker_name,
        "alignment_model": mfa_model_name,
        "sample_seed": sample_seed,
    }


def write_report(records, model):
    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=report_fields)
        writer.writeheader()

        for sample_number, record in enumerate(
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
                record["waveform"],
                record["audio_sample_rate"],
            )
            writer.writerow(create_report_row(sample_number, record, whisper_text, segment_rows))
            report_file.flush()


def main():
    check_tools()
    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    levels = load_levels()
    jobs = collect_sentence_jobs(levels)
    selected_jobs = select_sentence_jobs(jobs)

    if not selected_jobs:
        sys.exit("検証する文がありません。")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    qwen_model = load_qwen_model()
    records = create_alignment_records(qwen_model, selected_jobs)

    del qwen_model
    clear_cuda_memory()

    whisper_model = load_whisper_model()
    write_report(records, whisper_model)

    print(f"完了: {len(selected_jobs)}文から{len(records)}件の音声を検証しました。")
    print(f"CSVレポート: {report_path}")


if __name__ == "__main__":
    main()
