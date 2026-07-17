"""Qwen3-TTS demo — MLX (Apple Silicon) + FastAPI 薄后端。

两个模型:CustomVoice(预置音色+指令控制,启动即加载)、Base(声音克隆,按需加载)。
支持整段返回(WAV)与流式返回(raw s16le PCM)。
"""

import io
import os
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

MODEL_REPOS = {
	"custom": os.environ.get(
		"QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
	),
	"base": os.environ.get(
		"QWEN3_TTS_BASE_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
	),
	"design": os.environ.get(
		"QWEN3_TTS_DESIGN_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
	),
}

app = FastAPI(title="Qwen3-TTS MLX Demo")

state = {
	"custom": {"status": "loading", "error": None, "model": None},
	"base": {"status": "idle", "error": None, "model": None},
	"design": {"status": "idle", "error": None, "model": None},
}
meta = {"speakers": [], "languages": []}
gen_lock = threading.Lock()  # MLX 生成不做并发
load_lock = threading.Lock()


def _load(kind: str):
	try:
		from mlx_audio.tts.utils import load_model

		model = load_model(MODEL_REPOS[kind])
		state[kind]["model"] = model
		if kind == "custom":
			meta["speakers"] = model.get_supported_speakers()
			meta["languages"] = model.get_supported_languages()
		state[kind]["status"] = "ready"
	except Exception as e:  # noqa: BLE001
		state[kind]["status"] = "error"
		state[kind]["error"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_load, args=("custom",), daemon=True).start()


class TTSRequest(BaseModel):
	text: str = Field(min_length=1, max_length=2000)
	speaker: str = "Vivian"
	language: str = "auto"
	instruct: str | None = None
	temperature: float = Field(0.9, ge=0.0, le=2.0)
	top_k: int = Field(50, ge=1, le=200)
	top_p: float = Field(1.0, ge=0.0, le=1.0)
	repetition_penalty: float = Field(1.05, ge=0.8, le=2.0)
	stream: bool = False


class DesignRequest(BaseModel):
	text: str = Field(min_length=1, max_length=2000)
	instruct: str = Field(min_length=1, max_length=500)
	language: str = "auto"
	temperature: float = Field(0.9, ge=0.0, le=2.0)
	top_k: int = Field(50, ge=1, le=200)
	top_p: float = Field(1.0, ge=0.0, le=1.0)
	repetition_penalty: float = Field(1.05, ge=0.8, le=2.0)
	stream: bool = False


def _is_hum(a: np.ndarray) -> bool:
	"""跑飞段检测:模型不发 EOS 时,批式解码输出静音、流式解码输出恒定低噪(嗡声)。
	真语音 RMS 高且子窗变异大(实测 cv>0.25),嗡声 RMS~0.02 且 cv<0.06。
	ponytail: 持续的低幅平稳音(如哼鸣)会误杀,demo 够用;误杀了就调低 0.04/0.12 阈值"""
	w = a[: len(a) // 10 * 10].reshape(10, -1)
	if w.size == 0:
		return True
	rms = np.sqrt((w**2).mean(axis=1))
	m = float(rms.mean())
	return m < 2e-3 or (m < 0.04 and float(rms.std()) / m < 0.12)


def _trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
	# 从尾部按 1s 块裁掉嗡声/静音段,再按 100ms 窗裁残余静音
	while len(audio) > sr and _is_hum(audio[-sr:]):
		audio = audio[:-sr]
	win = sr // 10
	n = len(audio) - (len(audio) % win)
	if n > win:
		rms = np.sqrt((audio[:n].reshape(-1, win) ** 2).mean(axis=1))
		loud = np.nonzero(rms > 2e-3)[0]
		if len(loud):
			audio = audio[: min(len(audio), (loud[-1] + 4) * win)]
	return audio


def _chunk_iter(model, gen_kwargs: dict):
	"""统一走流式生成 + 静音早停:模型偶尔不发 EOS 会输出嗡声/静音直到 max_tokens,
	连续 5 个 1s 块嗡声就提前终止;句间自然停顿只有 1-2s,原样保留。"""
	# ponytail: 12Hz codec ≈ 12 token/s;按字数上限防跑飞(留 ~5x 余量),避免生成到 4096 卡几分钟
	gen_kwargs["max_tokens"] = min(4096, 300 + len(gen_kwargs["text"]) * 6)
	silent_run = []
	with gen_lock:
		for r in model.generate(**gen_kwargs, stream=True, streaming_interval=1.0):
			a = np.asarray(r.audio, dtype=np.float32)
			if os.environ.get("TTS_DEBUG"):
				w = a[: len(a) // 10 * 10].reshape(10, -1)
				rms = np.sqrt((w**2).mean(axis=1))
				m = float(rms.mean())
				print(f"[chunk] len={len(a)} rms={m:.4f} cv={float(rms.std()) / m if m > 0 else 0:.2f} hum={_is_hum(a)}", flush=True)
			if _is_hum(a):
				silent_run.append(a)
				if len(silent_run) >= 5:
					return
				continue
			yield from silent_run
			silent_run = []
			yield a


def _synthesize(model, gen_kwargs: dict, stream: bool):
	"""整段:拼接+裁尾部静音后返回 WAV;流式:边生成边吐 s16le PCM。"""
	sr = model.sample_rate

	if stream:
		def pcm_iter():
			try:
				for a in _chunk_iter(model, gen_kwargs):
					yield (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()
			except Exception as e:  # noqa: BLE001
				# 响应头已发出,只能中断流;错误进服务端日志
				print(f"[stream error] {type(e).__name__}: {e}")

		return StreamingResponse(
			pcm_iter(),
			media_type="application/octet-stream",
			headers={"X-Sample-Rate": str(sr)},
		)

	def _gen_once():
		chunks = list(_chunk_iter(model, dict(gen_kwargs)))
		if not chunks:
			return np.zeros(0, dtype=np.float32)
		return _trim_silence(np.concatenate(chunks), sr)

	t0 = time.time()
	try:
		audio = _gen_once()
		# ponytail: 生成对输入是确定性的,偶发退化(输出噪声/念一半转嗡声)在同进程内重试同结果;
		# 微扰一个不发音的位置(instruct 或文本末尾加句号)即可跳出吸引子,重掷一次并保留较长者。
		# 时长下限按语速估:中文 ~4字/s 取 0.15s/字,英文 ~14字符/s 取 0.05s/字符,上限 4s 防长文误判
		text = gen_kwargs["text"]
		cjk = sum("一" <= c <= "鿿" for c in text)
		per_char = 0.15 if cjk > len(text) / 2 else 0.05
		if len(audio) / sr < min(4.0, per_char * len(text)):
			if gen_kwargs.get("instruct"):
				gen_kwargs["instruct"] += "。"
			else:
				gen_kwargs["text"] += "。"
			retry = _gen_once()
			if len(retry) > len(audio):
				audio = retry
	except Exception as e:  # noqa: BLE001
		raise HTTPException(500, f"{type(e).__name__}: {e}") from e
	gen_time = time.time() - t0
	if len(audio) == 0:
		raise HTTPException(500, "no audio generated")
	pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
	buf = io.BytesIO()
	with wave.open(buf, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(sr)
		w.writeframes(pcm.tobytes())

	import mlx.core as mx

	dur = len(pcm) / sr
	return Response(
		content=buf.getvalue(),
		media_type="audio/wav",
		headers={
			"X-Audio-Duration": f"{dur:.2f}",
			"X-Gen-Time": f"{gen_time:.2f}",
			"X-RTF": f"{gen_time / dur:.3f}" if dur > 0 else "0",
			"X-Peak-Memory": f"{mx.get_peak_memory() / 1e9:.2f}",
		},
	)


@app.get("/api/status")
def status():
	return {
		"custom": {k: v for k, v in state["custom"].items() if k != "model"},
		"base": {k: v for k, v in state["base"].items() if k != "model"},
		"design": {k: v for k, v in state["design"].items() if k != "model"},
		"models": MODEL_REPOS,
		"speakers": meta["speakers"],
		"languages": meta["languages"],
	}


@app.post("/api/load-base")
def load_base():
	with load_lock:
		if state["base"]["status"] in ("idle", "error"):
			state["base"]["status"] = "loading"
			state["base"]["error"] = None
			threading.Thread(target=_load, args=("base",), daemon=True).start()
	return {"status": state["base"]["status"]}


@app.post("/api/load-design")
def load_design():
	with load_lock:
		if state["design"]["status"] in ("idle", "error"):
			state["design"]["status"] = "loading"
			state["design"]["error"] = None
			threading.Thread(target=_load, args=("design",), daemon=True).start()
	return {"status": state["design"]["status"]}


@app.post("/api/tts")
def tts(req: TTSRequest):
	if state["custom"]["status"] != "ready":
		raise HTTPException(503, f"model not ready: {state['custom']['status']}")
	return _synthesize(
		state["custom"]["model"],
		dict(
			text=req.text,
			voice=req.speaker,
			lang_code=req.language,
			instruct=req.instruct or None,
			temperature=req.temperature,
			top_k=req.top_k,
			top_p=req.top_p,
			repetition_penalty=req.repetition_penalty,
		),
		req.stream,
	)


@app.post("/api/clone")
def clone(
	ref_audio: UploadFile,
	ref_text: str = Form(min_length=1),
	text: str = Form(min_length=1, max_length=2000),
	language: str = Form("auto"),
	temperature: float = Form(0.9),
	top_k: int = Form(50),
	top_p: float = Form(1.0),
	repetition_penalty: float = Form(1.05),
	stream: bool = Form(False),
):
	if state["base"]["status"] != "ready":
		raise HTTPException(
			503, f"base model not ready: {state['base']['status']} {state['base']['error'] or ''}"
		)
	suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
	with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
		f.write(ref_audio.file.read())
		ref_path = f.name
	try:
		return _synthesize(
			state["base"]["model"],
			dict(
				text=text,
				ref_audio=ref_path,
				ref_text=ref_text,
				lang_code=language,
				temperature=temperature,
				top_k=top_k,
				top_p=top_p,
				repetition_penalty=repetition_penalty,
			),
			stream,
		)
	finally:
		if not stream:
			os.unlink(ref_path)
		# ponytail: 流式时临时文件等一轮后删,避免生成器还没读就被清;泄漏上限=系统tmp清理
		else:
			threading.Timer(600, lambda: os.path.exists(ref_path) and os.unlink(ref_path)).start()


@app.post("/api/design")
def design(req: DesignRequest):
	if state["design"]["status"] != "ready":
		raise HTTPException(
			503, f"design model not ready: {state['design']['status']} {state['design']['error'] or ''}"
		)
	return _synthesize(
		state["design"]["model"],
		dict(
			text=req.text,
			instruct=req.instruct,
			lang_code=req.language,
			temperature=req.temperature,
			top_k=req.top_k,
			top_p=req.top_p,
			repetition_penalty=req.repetition_penalty,
		),
		req.stream,
	)


@app.get("/")
def index():
	return FileResponse(Path(__file__).parent / "index.html")
