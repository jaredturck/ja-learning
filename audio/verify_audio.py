from pathlib import Path
import csv
import json
import random
import subprocess
import sys

import numpy
import torch
import whisper
from tqdm import tqdm

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
index_path = project_directory / "public" / "audio" / "index.json"
report_path = script_directory / "verification_report.csv"

sample_count = 200
sample_seed = 0
whisper_model_name = "large-v3"
whisper_language = "ja"
whisper_beam_size = 5
gpu_index = 0
decode_sample_rate = 16000

report_fields = [
    "sample_number",
    "level_id",
    "audio_file",
    "start_seconds",
    "indexed_duration_seconds",
    "decoded_duration_seconds",
    "expected_text",
    "expected_character_count",
    "whisper_text",
    "whisper_model",
    "whisper_language",
    "whisper_segment_count",
    "whisper_average_log_probability",
    "whisper_max_no_speech_probability",
    "whisper_max_compression_ratio",
    "audio_rms",
    "audio_peak",
    "whisper_segments_json",
    "tts_model",
    "tts_speaker",
    "audio_index_version",
    "sample_seed",
]


def check_ffmpeg():
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        sys.exit("FFmpeg が見つかりません。")


def load_audio_index():
    if not index_path.exists():
        sys.exit(f"音声インデックスが見つかりません: {index_path}")

    audio_index = json.loads(index_path.read_text(encoding="utf-8"))

    if "levels" not in audio_index or not audio_index["levels"]:
        sys.exit("音声インデックスにレベル情報がありません。")

    return audio_index


def collect_clip_records(audio_index):
    records_by_level = {}

    for level_id, level_data in sorted(audio_index["levels"].items()):
        audio_file = level_data.get("file")
        clips = level_data.get("clips")

        if not audio_file or not isinstance(clips, dict) or not clips:
            sys.exit(f"音声インデックスの形式が正しくありません: {level_id}")

        level_records = []

        for expected_text, clip_range in clips.items():
            if not isinstance(clip_range, list) or len(clip_range) != 2:
                sys.exit(f"音声範囲の形式が正しくありません: {level_id} / {expected_text}")

            start_seconds = float(clip_range[0])
            duration_seconds = float(clip_range[1])

            if not expected_text or start_seconds < 0 or duration_seconds <= 0:
                sys.exit(f"音声範囲の値が正しくありません: {level_id} / {expected_text}")

            level_records.append({
                "level_id": level_id,
                "audio_file": audio_file,
                "expected_text": expected_text,
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
            })

        records_by_level[level_id] = level_records

    return records_by_level


def select_level_records(records, count, random_generator):
    if count >= len(records):
        return list(records)

    sorted_records = sorted(
        records,
        key=lambda record: (len(record["expected_text"]), record["expected_text"]),
    )
    selected_records = []

    for bucket_index in range(count):
        bucket_start = bucket_index * len(sorted_records) // count
        bucket_end = (bucket_index + 1) * len(sorted_records) // count
        bucket = sorted_records[bucket_start:bucket_end]
        selected_records.append(random_generator.choice(bucket))

    return selected_records


def select_samples(records_by_level):
    all_records = [
        record
        for level_records in records_by_level.values()
        for record in level_records
    ]

    if sample_count >= len(all_records):
        selected_records = all_records
    else:
        random_generator = random.Random(sample_seed)
        level_ids = sorted(records_by_level)
        base_count = sample_count // len(level_ids)
        extra_count = sample_count % len(level_ids)
        selected_records = []
        selected_keys = set()

        for level_index, level_id in enumerate(level_ids):
            level_count = base_count + (1 if level_index < extra_count else 0)

            if level_count == 0:
                continue

            level_selection = select_level_records(
                records_by_level[level_id],
                level_count,
                random_generator,
            )

            for record in level_selection:
                record_key = (record["level_id"], record["expected_text"])
                selected_keys.add(record_key)
                selected_records.append(record)

        if len(selected_records) < sample_count:
            remaining_records = [
                record
                for record in all_records
                if (record["level_id"], record["expected_text"]) not in selected_keys
            ]
            random_generator.shuffle(remaining_records)
            selected_records.extend(
                remaining_records[:sample_count - len(selected_records)]
            )

    selected_records = sorted(
        selected_records,
        key=lambda record: (
            record["level_id"],
            len(record["expected_text"]),
            record["expected_text"],
        ),
    )

    for sample_number, record in enumerate(selected_records, start=1):
        record["sample_number"] = sample_number

    return selected_records


def decode_level_audio(audio_path):
    if not audio_path.exists():
        sys.exit(f"音声ファイルが見つかりません: {audio_path}")

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(decode_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8", errors="replace").strip()
        sys.exit(f"音声ファイルの読み込みに失敗しました: {audio_path}\n{error_message}")

    waveform = numpy.frombuffer(result.stdout, dtype="<f4").copy()

    if waveform.size == 0:
        sys.exit(f"音声ファイルが空です: {audio_path}")

    return waveform


def extract_clip(level_waveform, record):
    start_sample = round(record["start_seconds"] * decode_sample_rate)
    end_sample = round(
        (record["start_seconds"] + record["duration_seconds"])
        * decode_sample_rate
    )

    if start_sample < 0 or end_sample <= start_sample:
        sys.exit(
            "音声インデックスの範囲が正しくありません: "
            f"{record['level_id']} / {record['expected_text']}"
        )

    if end_sample > level_waveform.size:
        sys.exit(
            "音声インデックスが音声ファイルの長さを超えています: "
            f"{record['level_id']} / {record['expected_text']}"
        )

    return level_waveform[start_sample:end_sample].copy()


def load_whisper_model():
    if not torch.cuda.is_available():
        sys.exit("CUDA対応GPUが見つかりません。")

    device = f"cuda:{gpu_index}"
    print(f"GPU: {torch.cuda.get_device_name(gpu_index)}")
    print(f"Whisperモデル: {whisper_model_name}")
    return whisper.load_model(whisper_model_name, device=device)


def transcribe_clip(model, waveform):
    result = model.transcribe(
        waveform,
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


def create_report_row(record, waveform, whisper_text, segment_rows, audio_index):
    audio_rms = float(numpy.sqrt(numpy.mean(numpy.square(waveform))))
    audio_peak = float(numpy.max(numpy.abs(waveform)))

    return {
        "sample_number": record["sample_number"],
        "level_id": record["level_id"],
        "audio_file": record["audio_file"],
        "start_seconds": round(record["start_seconds"], 8),
        "indexed_duration_seconds": round(record["duration_seconds"], 8),
        "decoded_duration_seconds": round(waveform.size / decode_sample_rate, 8),
        "expected_text": record["expected_text"],
        "expected_character_count": len(record["expected_text"]),
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
        "audio_rms": round(audio_rms, 8),
        "audio_peak": round(audio_peak, 8),
        "whisper_segments_json": json.dumps(
            segment_rows,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "tts_model": audio_index.get("model", ""),
        "tts_speaker": audio_index.get("speaker", ""),
        "audio_index_version": audio_index.get("version", ""),
        "sample_seed": sample_seed,
    }


def write_report(audio_index, selected_records, model):
    records_by_level = {}

    for record in selected_records:
        records_by_level.setdefault(record["level_id"], []).append(record)

    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=report_fields)
        writer.writeheader()

        with tqdm(
            total=len(selected_records),
            desc="音声を検証しています",
            unit="件",
            dynamic_ncols=True,
        ) as progress_bar:
            for level_id in sorted(records_by_level):
                level_data = audio_index["levels"][level_id]
                audio_path = index_path.parent / level_data["file"]
                level_waveform = decode_level_audio(audio_path)

                for record in records_by_level[level_id]:
                    waveform = extract_clip(level_waveform, record)
                    whisper_text, segment_rows = transcribe_clip(model, waveform)
                    writer.writerow(
                        create_report_row(
                            record,
                            waveform,
                            whisper_text,
                            segment_rows,
                            audio_index,
                        )
                    )
                    report_file.flush()
                    progress_bar.update(1)


def main():
    check_ffmpeg()
    audio_index = load_audio_index()
    records_by_level = collect_clip_records(audio_index)
    selected_records = select_samples(records_by_level)

    if not selected_records:
        sys.exit("検証する音声がありません。")

    model = load_whisper_model()
    write_report(audio_index, selected_records, model)

    print(f"完了: {len(selected_records)}件の音声を検証しました。")
    print(f"CSVレポート: {report_path}")


if __name__ == "__main__":
    main()
