from pathlib import Path
import json
import subprocess
import sys

import flash_attn
import json5
import numpy
import torch
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
voice_instruction = "日本語学習教材の音声です。入力された日本語だけを、自然で明瞭に、落ち着いた速度で発音してください。"
silence_seconds = 0.2
opus_bitrate = "96k"
batch_size = 32

def load_levels():
    levels_source = levels_path.read_text(encoding="utf-8")
    levels_text = levels_source.split(levels_start_marker, 1)[1]
    levels_text = levels_text.split(levels_end_marker, 1)[0]
    return json5.loads(levels_text)

def add_unique_text(texts, seen_texts, text):
    assert text

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

def load_model():
    assert torch.cuda.is_available()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashAttention: {flash_attn.__version__}")
    print(f"バッチサイズ: {batch_size}")
    print(f"モデルを読み込んでいます: {model_name}")

    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

def generate_audio_batch(model, texts):
    wavs, sample_rate = model.generate_custom_voice(
        text=texts,
        language=["Japanese"] * len(texts),
        speaker=[speaker_name] * len(texts),
        instruct=[voice_instruction] * len(texts),
    )

    assert len(wavs) == len(texts)
    return wavs, sample_rate

def generate_all_audio(model, texts):
    generated_audio = {}
    sample_rate = None
    batch_starts = range(0, len(texts), batch_size)

    for batch_start in tqdm(
        batch_starts,
        desc="音声を生成しています",
        unit="バッチ",
        dynamic_ncols=True,
    ):
        batch_texts = texts[batch_start:batch_start + batch_size]
        wavs, current_sample_rate = generate_audio_batch(model, batch_texts)

        if sample_rate is None:
            sample_rate = current_sample_rate

        assert current_sample_rate == sample_rate

        for text, waveform in zip(batch_texts, wavs, strict=True):
            waveform = numpy.asarray(waveform, dtype=numpy.float32).reshape(-1)
            assert waveform.size > 0
            assert text not in generated_audio
            generated_audio[text] = waveform

    assert set(generated_audio) == set(texts)
    return generated_audio, sample_rate

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

    assert output_path.exists()
    assert output_path.stat().st_size > 0

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

    assert set(clips) == set(texts)

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

    assert index_path.exists()
    assert set(index["levels"]) == {level["id"] for level in levels}

def main():
    levels = load_levels()
    level_texts = collect_level_texts(levels)
    unique_texts = sorted({
        text
        for texts in level_texts.values()
        for text in texts
    })

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    check_ffmpeg()
    get_output_paths()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = load_model()
    generated_audio, sample_rate = generate_all_audio(model, unique_texts)

    reset_output_directory()
    write_index(levels, level_texts, generated_audio, sample_rate)

    print(f"完了: {len(unique_texts)}件の音声を{len(levels)}個のレベル音声ファイルに書き出しました。")


if __name__ == "__main__":
    main()
