# Qwen3-TTS MLX Studio

English | [中文](#中文说明)

A local web demo for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), running fully offline on Apple Silicon via [mlx-audio](https://github.com/Blaizzy/mlx-audio) (MLX, GPU-accelerated). One dark-themed single-page UI, three tabs, three models.

## Features

- **Preset voices** — 9 built-in speakers (incl. Beijing dialect *Dylan*, Sichuan dialect *Eric*, Japanese *Ono_Anna*, Korean *Sohee*), powered by the CustomVoice model
- **Instruction control** — steer emotion / tone / speed with natural language, e.g. "speak with great anger" or "whisper like telling a secret"
- **Voice cloning** — record in the browser or upload a 3–15s reference clip plus its transcript, then speak any text in that voice (Base model, cross-lingual supported)
- **Voice design** — no reference audio at all: describe a voice in words ("a calm, deep male narrator with a slow pace") and the VoiceDesign model invents it
- **Streaming** — toggle on to play audio as it generates, first chunk in ~0.6–2s
- 10 languages + auto detection; adjustable temperature / top_k / top_p / repetition penalty
- Waveform view, duration / generation time / RTF / peak-memory stats, replayable & downloadable history

## Requirements

- Apple Silicon Mac (M1 or later)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- ~9GB disk for the three models (each ~2.9GB, downloaded on demand)

## Quick start

```bash
git clone https://github.com/misakaikato/qwen3-tts-mlx-demo.git
cd qwen3-tts-mlx-demo
uv sync
uv run uvicorn server:app --port 8321
# open http://localhost:8321
```

Models are **not** bundled in this repo. On first launch the CustomVoice model (~2.9GB) is downloaded automatically from Hugging Face; the Base (cloning) and VoiceDesign models download the first time you open their tabs. To pre-download manually:

```bash
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
```

Swap models (e.g. bf16 for higher fidelity) via environment variables: `QWEN3_TTS_MODEL`, `QWEN3_TTS_BASE_MODEL`, `QWEN3_TTS_DESIGN_MODEL`.

## Tips

- **Cloning**: microphone recording needs browser permission; the reference clip should be 3–15s of clean speech, and the transcript must match it exactly. The reference can be in any language — the output can be in any supported language.
- **Performance**: warm RTF is roughly 0.2–0.35 on an M-series GPU (10s of audio in ~2.5s).

## Known model quirks

The model occasionally fails to emit EOS (dialect / instruction combos make it likelier) and pads the output with silence (batch decode) or a constant low hum (streaming decode) until the token limit. The server ships four guards: text-length-scaled `max_tokens`, early termination after 5s of continuous hum, tail trimming, and an automatic perturb-and-retry when the output is implausibly short for the text.

Also note: generation is deterministic for a given input within one process. A few "quiet" voice descriptions (e.g. "low, husky, very slow") can degenerate persistently in some process instances — reword the description or restart the server to reroll. This is upstream model/quantization behaviour, not a demo bug.

---

# 中文说明

[English](#qwen3-tts-mlx-studio) | 中文

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) 的本地 Web demo，通过 [mlx-audio](https://github.com/Blaizzy/mlx-audio)（MLX，GPU 加速）在 Apple Silicon 上完全离线运行。单页深色 UI，三个页签对应三个模型。

## 能力

- **预置音色** — 9 种内置说话人（含北京话 Dylan、四川话 Eric、日语 Ono_Anna、韩语 Sohee），CustomVoice 模型
- **指令控制** — 用自然语言控制情感 / 语气 / 语速，如「用特别愤怒的语气说」「耳语，像在说悄悄话」
- **声音克隆** — 页面内录音或上传 3–15 秒参考音频 + 文字稿，即可用该音色朗读任意文本（Base 模型，支持跨语言）
- **音色设计** — 完全不需要参考音频：用一段文字描述凭空设计声音（如「低沉浑厚的旁白男声，语速缓慢」），VoiceDesign 模型
- **流式生成** — 打开开关边生成边播放，首包约 0.6–2 秒
- 10 种语言 + 自动检测；temperature / top_k / top_p / repetition penalty 可调
- 波形图、时长 / 耗时 / RTF / 峰值内存统计，历史记录可回放下载

## 环境要求

- Apple Silicon Mac（M1 及以上）
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- 约 9GB 磁盘空间（三个模型各约 2.9GB，按需下载）

## 快速开始

```bash
git clone https://github.com/misakaikato/qwen3-tts-mlx-demo.git
cd qwen3-tts-mlx-demo
uv sync
uv run uvicorn server:app --port 8321
# 打开 http://localhost:8321
```

仓库**不包含**模型文件。首次启动自动从 Hugging Face 下载 CustomVoice 模型（约 2.9GB）；首次进入「声音克隆」/「音色设计」页签时分别下载对应模型。也可手动预下载：

```bash
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
uv run hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
```

换模型（如 bf16 更高保真）用环境变量：`QWEN3_TTS_MODEL`、`QWEN3_TTS_BASE_MODEL`、`QWEN3_TTS_DESIGN_MODEL`。

## 使用提示

- **克隆**：录音需要浏览器麦克风权限；参考音频建议 3–15 秒干净人声，文字稿必须与音频内容一致。参考音频可以是任何语言，生成文本可以是任何受支持的语言。
- **性能**：M 系列 GPU 热机后 RTF 约 0.2–0.35（10 秒音频约 2.5 秒生成）。

## 已知模型问题

模型偶发不发 EOS（方言 / 指令组合更易触发），会持续输出静音（批式解码）或恒定低噪（流式解码）直到 token 上限。服务端有四道护栏：按字数动态限制 max_tokens、连续 5 秒嗡声提前终止、返回前裁掉尾部嗡声、输出时长明显低于文本语速下限时自动微扰提示词重试一次。

另外：生成对输入是确定性的（同进程同输入必得同输出），个别「安静系」音色描述（如「低沉沙哑、语速缓慢」）在某些进程实例里会稳定退化成短噪声——换个措辞或重启服务重掷即可，响亮音色不受影响。属上游模型 / 量化行为，非 demo 的 bug。
