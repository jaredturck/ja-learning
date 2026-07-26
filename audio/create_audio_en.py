from pathlib import Path
import gc
import json
import shutil
import sys

import numpy
import torch
from accelerate import Accelerator
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

from create_audio import clear_cuda_memory
from create_audio import create_audio_batches
from create_audio import encode_opus
from create_audio import load_levels
from create_audio import load_model
from create_audio import model_name
from create_audio import normalize_generated_waveform
from create_audio import silence_seconds
from create_audio import speaker_name
from create_audio import tts_sample_rate
from create_audio import voice_instruction

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
public_directory = project_directory / "public"
output_directory = public_directory / "audio_en"
staging_output_directory = public_directory / ".audio_en_staging"
backup_output_directory = public_directory / ".audio_en_backup"
work_directory = Path("/tmp/ja_learning_create_audio_en_work")
expected_level_count = 100
expected_sentence_count = 620

# accelerate launch --multi_gpu --num_processes 2 audio/create_audio_en.py


def collect_audio_plan(levels):
    jobs = []
    level_jobs = {}
    sentence_ids = set()

    for level in levels:
        level_id = str(level["id"])
        current_level_jobs = []

        for sentence in level["sentences"]:
            sentence_id = str(sentence["id"])
            english_text = str(sentence["english"]).strip()

            if not english_text:
                sys.exit(f"空の英語文が見つかりました: {sentence_id}")

            if sentence_id in sentence_ids:
                sys.exit(f"重複した文IDが見つかりました: {sentence_id}")

            sentence_ids.add(sentence_id)
            job = {
                "job_id": sentence_id,
                "file_stem": sentence_id,
                "level_id": level_id,
                "sentence_id": sentence_id,
                "full_sentence_text": english_text,
            }
            jobs.append(job)
            current_level_jobs.append(job)

        level_jobs[level_id] = current_level_jobs

    if len(levels) != expected_level_count:
        sys.exit(
            f"レベル数が想定と一致しません: "
            f"expected={expected_level_count} actual={len(levels)}"
        )

    if len(jobs) != expected_sentence_count:
        sys.exit(
            f"英語文数が想定と一致しません: "
            f"expected={expected_sentence_count} actual={len(jobs)}"
        )

    return jobs, level_jobs


def generate_audio_batch(model, jobs):
    texts = [job["full_sentence_text"] for job in jobs]

    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(
            text=texts,
            language=["English"] * len(texts),
            speaker=[speaker_name] * len(texts),
            instruct=[voice_instruction] * len(texts),
            max_new_tokens=2048,
        )

    if len(wavs) != len(jobs):
        sys.exit("入力した英語文数と生成された音声数が一致しません。")

    return [
        normalize_generated_waveform(waveform, sample_rate, text)
        for text, waveform in zip(texts, wavs, strict=True)
    ]


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
            "1文の英語音声生成でもGPUメモリが不足しました: "
            f"{jobs[0]['full_sentence_text']}"
        )

    middle = len(jobs) // 2
    left_results = generate_audio_batch_with_retry(model, jobs[:middle])
    clear_cuda_memory()
    right_results = generate_audio_batch_with_retry(model, jobs[middle:])
    clear_cuda_memory()

    return [*left_results, *right_results]


def reset_directory(path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)


def generate_local_audio(model, jobs, accelerator):
    batches = create_audio_batches(jobs, False)
    local_batches = batches[
        accelerator.process_index::accelerator.num_processes
    ]

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
                numpy.save(work_directory / f"{job['file_stem']}.npy", waveform)


def build_level_audio(level_id, jobs, destination_directory):
    silence = numpy.zeros(
        round(tts_sample_rate * silence_seconds),
        dtype=numpy.float32,
    )
    audio_parts = [silence]
    current_sample = silence.size
    clips = {}

    for job in jobs:
        waveform_path = work_directory / f"{job['file_stem']}.npy"

        if not waveform_path.is_file():
            sys.exit(f"生成済み英語音声が見つかりません: {job['sentence_id']}")

        waveform = numpy.load(waveform_path)
        start_seconds = current_sample / tts_sample_rate
        duration_seconds = waveform.size / tts_sample_rate
        clips[job["sentence_id"]] = [
            round(start_seconds, 8),
            round(duration_seconds, 8),
        ]
        audio_parts.extend([waveform, silence])
        current_sample += waveform.size + silence.size

    output_path = destination_directory / f"{level_id}.opus"
    encode_opus(numpy.concatenate(audio_parts), output_path)

    if set(clips) != {job["sentence_id"] for job in jobs}:
        sys.exit(f"英語音声インデックスの作成に失敗しました: {level_id}")

    return {
        "file": output_path.name,
        "clips": clips,
    }


def write_index(levels, level_jobs, destination_directory):
    index = {
        "version": 1,
        "model": model_name,
        "speaker": speaker_name,
        "language": "English",
        "levels": {},
    }

    for level in levels:
        level_id = str(level["id"])
        index["levels"][level_id] = build_level_audio(
            level_id,
            level_jobs[level_id],
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
            "英語音声の出力ファイル一覧が想定と一致しません。"
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
    jobs, level_jobs = collect_audio_plan(levels)

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    if accelerator.is_main_process:
        reset_directory(work_directory)
        prepare_output_staging()
        print(
            f"英語音声生成対象: {len(jobs)}文 / "
            f"{len(levels)}レベル"
        )

    accelerator.wait_for_everyone()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = load_model(accelerator)
    generate_local_audio(model, jobs, accelerator)

    del model
    gc.collect()
    clear_cuda_memory()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        generated_paths = list(work_directory.glob("*.npy"))

        if len(generated_paths) != len(jobs):
            sys.exit(
                "生成済み英語音声数が想定と一致しません: "
                f"expected={len(jobs)} actual={len(generated_paths)}"
            )

        write_index(levels, level_jobs, staging_output_directory)
        publish_output()
        shutil.rmtree(work_directory)
        print(
            f"完了: {len(jobs)}件の英語音声を"
            f"{len(levels)}個のレベル音声ファイルに書き出しました。"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
