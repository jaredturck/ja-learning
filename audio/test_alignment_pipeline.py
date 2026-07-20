from pathlib import Path
import csv
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import unicodedata

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json5
import numpy
import torch
import whisper
from qwen_tts import Qwen3TTSModel


project_directory = Path("/home/jared/Dropbox/Documents/Dropbox_Documents/ja-learning")
audio_directory = project_directory / "audio"
levels_path = project_directory / "src" / "levels.ts"
report_path = audio_directory / "alignment_verification_report.csv"
ffmpeg_executable = Path("/usr/bin/ffmpeg")
mfa_executable = Path("/home/jared/miniforge3/envs/mfa/bin/mfa")
mfa_environment_directory = Path("/home/jared/miniforge3/envs/mfa")
qwen_model_path = Path(
    "/home/jared/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/"
    "snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe"
)
whisper_model_path = Path("/home/jared/.cache/whisper/large-v3.pt")

levels_start_marker = "/* AUDIO_LEVELS_START */"
levels_end_marker = "/* AUDIO_LEVELS_END */"
qwen_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
speaker_name = "Ono_Anna"
voice_instruction = "入力文そのまま"
mfa_model_name = "japanese_mfa"
mfa_version = "3.4.1"
whisper_model_name = "large-v3"
whisper_language = "ja"
whisper_beam_size = 5
selection_seed = 0
tts_seed = 0
gpu_index = 0
whisper_sample_rate = 16000
mfa_num_jobs = 4

mfa_environment = os.environ.copy()
mfa_environment["PATH"] = (
    f"{mfa_environment_directory / 'bin'}{os.pathsep}"
    f"{mfa_environment.get('PATH', '')}"
)
mfa_environment["CONDA_PREFIX"] = str(mfa_environment_directory)
mfa_environment["CONDA_DEFAULT_ENV"] = "mfa"
for variable_name in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"]:
    mfa_environment.pop(variable_name, None)

alignment_punctuation = " \t\n\r\"'`“”‘’「」『』（）()［］[]【】。、，．,.！？!?・:：;；…"

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


def load_levels():
    levels_source = levels_path.read_text(encoding="utf-8")
    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    return json5.loads(levels_text)


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text))


def get_alignment_token(text):
    return normalize_text(text).strip(alignment_punctuation)


def collect_sentence_jobs(levels):
    jobs = []

    for level in levels:
        for sentence_index, sentence in enumerate(level["sentences"]):
            chunks = [
                normalize_text(chunk["japanese"])
                for chunk in sentence["chunks"]
            ]
            alignment_tokens = [get_alignment_token(chunk) for chunk in chunks]
            jobs.append({
                "level_id": str(level["id"]),
                "sentence_id": str(sentence["id"]),
                "sentence_index": sentence_index,
                "full_sentence_text": "".join(chunks),
                "chunks": chunks,
                "alignment_tokens": alignment_tokens,
            })

    jobs.sort(key=lambda job: (job["level_id"], job["sentence_index"]))

    for sentence_sample_number, job in enumerate(jobs, start=1):
        job["sentence_sample_number"] = sentence_sample_number
        job["file_stem"] = f"sample-{sentence_sample_number:04d}"

    return jobs


def clear_cuda_memory():
    gc.collect()
    torch.cuda.synchronize(gpu_index)
    torch.cuda.empty_cache()


def load_qwen_model():
    return Qwen3TTSModel.from_pretrained(
        str(qwen_model_path),
        device_map=f"cuda:{gpu_index}",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def normalize_generated_waveform(waveform, sample_rate, text):
    waveform = numpy.asarray(waveform).squeeze()

    if waveform.ndim != 1:
        sys.exit(f"生成音声がモノラルではありません: {text} / {waveform.shape}")

    waveform = numpy.asarray(waveform, dtype=numpy.float32).copy()
    sample_rate = int(sample_rate)

    if sample_rate <= 0 or waveform.size == 0 or not numpy.isfinite(waveform).all():
        sys.exit(f"生成音声が無効です: {text}")

    return waveform, sample_rate


def generate_sentence_audio(model, text):
    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(
            text=[text],
            language=["Japanese"],
            speaker=[speaker_name],
            instruct=[voice_instruction],
            max_new_tokens=2048,
        )

    return normalize_generated_waveform(wavs[0], sample_rate, text)


def write_wav(path, waveform, sample_rate):
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
            str(sample_rate),
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


def write_alignment_corpus(corpus_directory, jobs, model):
    generated_sentences = []

    for job in jobs:
        waveform, sample_rate = generate_sentence_audio(
            model,
            job["full_sentence_text"],
        )
        write_wav(
            corpus_directory / f"{job['file_stem']}.wav",
            waveform,
            sample_rate,
        )
        (corpus_directory / f"{job['file_stem']}.lab").write_text(
            " ".join(job["alignment_tokens"]) + "\n",
            encoding="utf-8",
        )
        generated_sentences.append({
            "job": job,
            "waveform": waveform,
            "sample_rate": sample_rate,
        })

    return generated_sentences


def run_mfa_corpus_alignment(
    corpus_directory,
    aligned_directory,
    mfa_temporary_directory,
):
    subprocess.run(
        [
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
        ],
        env=mfa_environment,
        check=True,
    )


def find_alignment_path(aligned_directory, file_stem):
    matching_paths = list(aligned_directory.rglob(f"{file_stem}.json"))

    if len(matching_paths) != 1:
        sys.exit(
            f"MFA整列結果を一意に特定できません: "
            f"{file_stem} / {len(matching_paths)}"
        )

    return matching_paths[0]


def parse_word_entries(alignment_path):
    tiers = json.loads(alignment_path.read_text(encoding="utf-8"))["tiers"]
    word_tiers = [
        tier_data
        for tier_name, tier_data in tiers.items()
        if tier_name == "words" or tier_name.endswith(" - words")
    ]

    if len(word_tiers) != 1:
        sys.exit(f"MFA words tierを一意に特定できません: {alignment_path}")

    word_entries = []

    for start_seconds, end_seconds, raw_label in word_tiers[0]["entries"]:
        label = normalize_text(raw_label).strip()

        if label:
            word_entries.append({
                "start": float(start_seconds),
                "end": float(end_seconds),
                "label": label,
            })

    return word_entries


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


def extract_clip(waveform, sample_rate, start_seconds, end_seconds):
    start_sample = round(start_seconds * sample_rate)
    end_sample = round(end_seconds * sample_rate)

    if start_sample < 0 or end_sample <= start_sample or end_sample > waveform.size:
        sys.exit(
            "検証済みの整列範囲から音声を抽出できません: "
            f"start={start_seconds} end={end_seconds}"
        )

    return waveform[start_sample:end_sample]


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

    if row_type == "sentence":
        expected_text = job["full_sentence_text"]
        clip_waveform = waveform
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
        clip_waveform = (
            extract_clip(
                waveform,
                sample_rate,
                word_entry["start"],
                word_entry["end"],
            )
            if alignment_valid and word_entry is not None
            else numpy.zeros(0, dtype=numpy.float32)
        )

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
        "source_sentence_duration_seconds": waveform.size / sample_rate,
        "audio_sample_rate": sample_rate,
        "waveform": clip_waveform,
    }


def create_alignment_records(model, jobs):
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
            corpus_directory,
            jobs,
            model,
        )
        run_mfa_corpus_alignment(
            corpus_directory,
            aligned_directory,
            mfa_temporary_directory,
        )

        for generated_sentence in generated_sentences:
            job = generated_sentence["job"]
            word_entries = parse_word_entries(
                find_alignment_path(aligned_directory, job["file_stem"])
            )
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


def resample_for_whisper(waveform, sample_rate):
    result = subprocess.run(
        [
            ffmpeg_executable,
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
        input=waveform.astype("<f4", copy=False).tobytes(),
        stdout=subprocess.PIPE,
        check=True,
    )
    whisper_waveform = numpy.frombuffer(result.stdout, dtype="<f4").copy()

    if whisper_waveform.size == 0 or not numpy.isfinite(whisper_waveform).all():
        sys.exit("Whisper用の音声が無効です。")

    return whisper_waveform


def load_whisper_model():
    return whisper.load_model(
        str(whisper_model_path),
        device=f"cuda:{gpu_index}",
    )


def segment_float(segment, key):
    value = segment.get(key)
    return 0.0 if value is None else float(value)


def transcribe_clip(model, waveform, sample_rate):
    if waveform.size == 0:
        return "", []

    result = model.transcribe(
        resample_for_whisper(waveform, sample_rate),
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
    segment_rows = [
        {
            "start": round(segment_float(segment, "start"), 4),
            "end": round(segment_float(segment, "end"), 4),
            "text": segment.get("text", ""),
            "avg_logprob": round(segment_float(segment, "avg_logprob"), 6),
            "no_speech_prob": round(segment_float(segment, "no_speech_prob"), 6),
            "compression_ratio": round(segment_float(segment, "compression_ratio"), 6),
            "temperature": round(segment_float(segment, "temperature"), 6),
        }
        for segment in result.get("segments", [])
    ]

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


def write_report(records, model):
    with tempfile.TemporaryDirectory(
        prefix=".alignment_report_",
        dir=audio_directory,
    ) as temporary_report_directory_name:
        temporary_report_path = (
            Path(temporary_report_directory_name) / report_path.name
        )

        with temporary_report_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as report_file:
            writer = csv.DictWriter(report_file, fieldnames=report_fields)
            writer.writeheader()

            for report_row_number, record in enumerate(records, start=1):
                whisper_text, segment_rows = transcribe_clip(
                    model,
                    record["waveform"],
                    record["audio_sample_rate"],
                )
                writer.writerow(
                    create_report_row(
                        report_row_number,
                        record,
                        whisper_text,
                        segment_rows,
                    )
                )

        temporary_report_path.replace(report_path)


def main():
    torch.cuda.set_device(gpu_index)
    jobs = collect_sentence_jobs(load_levels())

    torch.manual_seed(tts_seed)
    torch.cuda.manual_seed_all(tts_seed)

    qwen_model = load_qwen_model()
    records = create_alignment_records(qwen_model, jobs)
    del qwen_model
    clear_cuda_memory()

    whisper_model = load_whisper_model()
    write_report(records, whisper_model)
    del whisper_model
    clear_cuda_memory()


if __name__ == "__main__":
    main()
