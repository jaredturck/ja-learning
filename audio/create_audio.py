from pathlib import Path
import gc
import json
import os
import subprocess
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import flash_attn
import json5
import numpy
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from huggingface_hub.utils import disable_progress_bars
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent
levels_path = project_directory / "src" / "levels.ts"
public_directory = project_directory / "public"
output_directory = public_directory / "audio"

levels_start_marker = "/* AUDIO_LEVELS_START */"
levels_end_marker = "/* AUDIO_LEVELS_END */"
model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
speaker_name = "Ono_Anna"
silence_seconds = 0.2
opus_bitrate = "96k"
max_batch_size = 24
max_batch_cost = 384

# accelerate launch --multi_gpu --num_processes 2 audio/create_audio.py

def load_levels():
    levels_source = levels_path.read_text(encoding="utf-8")

    if levels_start_marker not in levels_source or levels_end_marker not in levels_source:
        sys.exit("levels.ts に音声データ用の開始・終了マーカーがありません。")

    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    return json5.loads(levels_text)

def add_unique_text(texts, seen_texts, text):
    if not text:
        sys.exit("空の日本語テキストが見つかりました。")

    if text in seen_texts:
        return

    seen_texts.add(text)
    texts.append(text)

def collect_level_texts(levels):
    level_texts = {}

    for level in levels:
        texts = []
        seen_texts = set()

        for sentence in level["sentences"]:
            sentence_text = "".join(chunk["japanese"] for chunk in sentence["chunks"])
            add_unique_text(texts, seen_texts, sentence_text)

            for chunk in sentence["chunks"]:
                add_unique_text(texts, seen_texts, chunk["japanese"])

                for distractor in chunk["distractors"]:
                    add_unique_text(texts, seen_texts, distractor)

        level_texts[level["id"]] = texts

    return level_texts

def check_ffmpeg():
    subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

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

def generate_audio_batch(model, texts):
    wavs, sample_rate = model.generate_custom_voice(
        text=texts,
        language=["Japanese"] * len(texts),
        speaker=[speaker_name] * len(texts),
        max_new_tokens=2048,
    )

    if len(wavs) != len(texts):
        sys.exit("入力した日本語テキスト数と生成された音声数が一致しません。")

    return wavs, sample_rate

def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()

def try_generate_audio_batch(model, texts):
    try:
        return generate_audio_batch(model, texts)
    except torch.OutOfMemoryError:
        return None

def generate_audio_batch_with_retry(model, texts):
    result = try_generate_audio_batch(model, texts)

    if result is not None:
        wavs, sample_rate = result
        return [(texts, wavs, sample_rate)]

    clear_cuda_memory()

    if len(texts) == 1:
        sys.exit(f"1件の音声生成でもGPUメモリが不足しました: {texts[0]}")

    middle = len(texts) // 2
    left_results = generate_audio_batch_with_retry(model, texts[:middle])
    clear_cuda_memory()
    right_results = generate_audio_batch_with_retry(model, texts[middle:])
    clear_cuda_memory()

    return [*left_results, *right_results]

def create_audio_batches(texts):
    sorted_texts = sorted(texts, key=lambda text: (len(text), text))
    batches = []
    current_batch = []
    current_max_length = 0

    for text in sorted_texts:
        next_batch_size = len(current_batch) + 1
        next_max_length = max(current_max_length, len(text))
        next_batch_cost = next_batch_size * next_max_length

        if current_batch and (next_batch_size > max_batch_size or next_batch_cost > max_batch_cost):
            batches.append(current_batch)
            current_batch = []
            current_max_length = 0

        current_batch.append(text)
        current_max_length = max(current_max_length, len(text))

    if current_batch:
        batches.append(current_batch)

    return batches

def get_local_batches(batches, accelerator):
    return batches[accelerator.process_index::accelerator.num_processes]

def generate_local_audio(model, texts, accelerator):
    generated_audio = {}
    sample_rate = None
    batches = create_audio_batches(texts)
    local_batches = get_local_batches(batches, accelerator)
    local_texts = {
        text
        for batch_texts in local_batches
        for text in batch_texts
    }

    for batch_texts in tqdm(
        local_batches,
        desc=f"GPU {accelerator.local_process_index}",
        unit="バッチ",
        position=accelerator.local_process_index,
        dynamic_ncols=True,
    ):
        batch_results = generate_audio_batch_with_retry(model, batch_texts)

        for result_texts, wavs, current_sample_rate in batch_results:
            if sample_rate is None:
                sample_rate = current_sample_rate

            if current_sample_rate != sample_rate:
                sys.exit("生成された音声のサンプルレートが一致しません。")

            for text, waveform in zip(result_texts, wavs, strict=True):
                waveform = numpy.asarray(waveform, dtype=numpy.float32).reshape(-1)

                if waveform.size == 0:
                    sys.exit(f"空の音声が生成されました: {text}")

                if text in generated_audio:
                    sys.exit(f"同じ日本語テキストの音声が重複して生成されました: {text}")

                generated_audio[text] = waveform

    if set(generated_audio) != local_texts:
        missing_texts = local_texts - set(generated_audio)
        unexpected_texts = set(generated_audio) - local_texts
        sys.exit(
            "ローカル生成音声の確認に失敗しました。"
            f" 未生成: {len(missing_texts)}件、想定外: {len(unexpected_texts)}件"
        )

    if local_batches and sample_rate is None:
        sys.exit("担当した音声が1件も生成されませんでした。")

    return generated_audio, sample_rate

def merge_generated_audio(gathered_results, texts):
    generated_audio = {}
    sample_rate = None

    for process_result in gathered_results:
        current_sample_rate = process_result["sample_rate"]

        if current_sample_rate is not None:
            if sample_rate is None:
                sample_rate = current_sample_rate

            if current_sample_rate != sample_rate:
                sys.exit("GPU間で生成された音声のサンプルレートが一致しません。")

        for text, waveform in process_result["generated_audio"].items():
            if text in generated_audio:
                sys.exit(f"GPU間で同じ日本語テキストの音声が重複しました: {text}")

            generated_audio[text] = waveform

    if set(generated_audio) != set(texts):
        missing_texts = set(texts) - set(generated_audio)
        unexpected_texts = set(generated_audio) - set(texts)
        sys.exit(
            "生成音声の確認に失敗しました。"
            f" 未生成: {len(missing_texts)}件、想定外: {len(unexpected_texts)}件"
        )

    if sample_rate is None:
        sys.exit("音声が1件も生成されませんでした。")

    return generated_audio, sample_rate

def gather_generated_audio(accelerator, local_generated_audio, local_sample_rate, texts):
    local_result = {
        "sample_rate": local_sample_rate,
        "generated_audio": local_generated_audio,
    }
    gathered_results = gather_object([local_result])

    if not accelerator.is_main_process:
        return None, None

    return merge_generated_audio(gathered_results, texts)

def get_output_paths():
    public_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    resolved_public_directory = public_directory.resolve()
    resolved_output_directory = output_directory.resolve()
    expected_output_directory = resolved_public_directory / "audio"

    if resolved_public_directory != project_directory / "public":
        sys.exit("安全確認に失敗しました: public フォルダーが想定された場所にありません。")

    if resolved_output_directory != expected_output_directory:
        sys.exit("安全確認に失敗しました: public/audio 以外は削除できません。")

    output_paths = list(output_directory.iterdir())

    if any(output_path.is_dir() for output_path in output_paths):
        sys.exit("安全確認に失敗しました: public/audio にフォルダーがあります。")

    return output_paths

def reset_output_directory():
    for output_path in get_output_paths():
        output_path.unlink()

def encode_opus(waveform, sample_rate, output_path):
    audio_bytes = waveform.astype("<f4", copy=False).tobytes()

    subprocess.run(
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
            "libopus",
            "-application",
            "audio",
            "-b:a",
            opus_bitrate,
            "-vbr",
            "on",
            "-compression_level",
            "10",
            str(output_path),
        ],
        input=audio_bytes,
        check=True,
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        sys.exit(f"Opusファイルの作成に失敗しました: {output_path}")

def build_level_audio(level_id, texts, generated_audio, sample_rate):
    silence = numpy.zeros(round(sample_rate * silence_seconds), dtype=numpy.float32)
    audio_parts = [silence]
    current_sample = silence.size
    clips = {}

    for text in texts:
        waveform = generated_audio[text]
        start_seconds = current_sample / sample_rate
        duration_seconds = waveform.size / sample_rate

        clips[text] = [round(start_seconds, 8), round(duration_seconds, 8)]
        audio_parts.extend([waveform, silence])
        current_sample += waveform.size + silence.size

    level_waveform = numpy.concatenate(audio_parts)
    output_path = output_directory / f"{level_id}.opus"

    encode_opus(level_waveform, sample_rate, output_path)

    if set(clips) != set(texts):
        sys.exit(f"レベル音声のインデックス作成に失敗しました: {level_id}")

    return {
        "file": output_path.name,
        "clips": clips,
    }

def write_index(levels, level_texts, generated_audio, sample_rate):
    index = {
        "version": 1,
        "model": model_name,
        "speaker": speaker_name,
        "levels": {},
    }

    for level in levels:
        level_id = level["id"]
        index["levels"][level_id] = build_level_audio(
            level_id,
            level_texts[level_id],
            generated_audio,
            sample_rate,
        )

    index_path = output_directory / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not index_path.exists() or index_path.stat().st_size == 0:
        sys.exit("音声インデックスの作成に失敗しました。")

    if set(index["levels"]) != {level["id"] for level in levels}:
        sys.exit("音声インデックスのレベル一覧が levels.ts と一致しません。")

def main():
    accelerator = Accelerator()
    levels = load_levels()
    level_texts = collect_level_texts(levels)
    unique_texts = {
        text
        for texts in level_texts.values()
        for text in texts
    }

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    if accelerator.is_main_process:
        check_ffmpeg()
        get_output_paths()

    accelerator.wait_for_everyone()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = load_model(accelerator)
    local_generated_audio, local_sample_rate = generate_local_audio(
        model,
        unique_texts,
        accelerator,
    )

    del model
    clear_cuda_memory()
    accelerator.wait_for_everyone()

    generated_audio, sample_rate = gather_generated_audio(
        accelerator,
        local_generated_audio,
        local_sample_rate,
        unique_texts,
    )

    if accelerator.is_main_process:
        reset_output_directory()
        write_index(levels, level_texts, generated_audio, sample_rate)
        print(f"完了: {len(unique_texts)}件の音声を{len(levels)}個のレベル音声ファイルに書き出しました。")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
