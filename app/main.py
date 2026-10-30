from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from app.services import ExhibitMatcher, KnowledgeBase
from app.vision import VESSEL_TYPE_NAMES

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
# Keep model caches beside this project so Windows user-cache permissions do not block downloads.
os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".model_cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".model_cache" / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
IMAGE_DIR = ROOT / os.getenv("IMAGE_DIR", "data/exhibit_gallery")
EXHIBITS_PATH = DATA_DIR / "exhibits.json"
VESSEL_TYPES_PATH = DATA_DIR / "vessel_types.json"
TYPE_THRESHOLD = float(os.getenv("VESSEL_TYPE_THRESHOLD", "0.65"))
EXHIBIT_THRESHOLD = float(os.getenv("EXHIBIT_SIMILARITY_THRESHOLD", "0.82"))
MARGIN_THRESHOLD = float(os.getenv("MATCH_MARGIN_THRESHOLD", "0.06"))
EXHIBIT_CANDIDATE_FLOOR = float(os.getenv("EXHIBIT_CANDIDATE_FLOOR", "0.68"))

app = FastAPI(title="青铜器智能导览系统", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
knowledge = KnowledgeBase(
    EXHIBITS_PATH,
    VESSEL_TYPES_PATH,
    os.getenv("TEXT_EMBEDDING_MODEL_PATH", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
)
matcher = ExhibitMatcher(
    IMAGE_DIR,
    knowledge.exhibits,
    os.getenv("VISION_WEIGHTS_PATH", ""),
    os.getenv("VESSEL_CLASSIFIER_PATH", "models/vessel_classifier/best.pt"),
    os.getenv("GALLERY_SPLIT_PATH", "data/exhibit_gallery_split.json"),
)


class ChatRequest(BaseModel):
    question: str
    exhibit_id: str | None = None
    vessel_type: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


def _client() -> OpenAI:
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="尚未配置千问 API Key。请复制 .env.example 为 .env 并填写 DASHSCOPE_API_KEY。")
    return OpenAI(api_key=key, base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))


def _answer(
    question: str,
    exhibit: dict[str, Any] | None,
    vessel_type: str | None,
    passages: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> str:
    client = _client()
    sources = "\n\n".join(f"[{i + 1}] {p['name']}\n{p['text']}\n来源：{p['source_url']}" for i, p in enumerate(passages))
    exhibit_label = exhibit["name"] if exhibit else VESSEL_TYPE_NAMES.get(vessel_type or "", "尚未确定展品")
    system = (
        "你是中国古代青铜器导览助手。只依据提供的资料回答，不要补造年代、用途或出处。"
        "资料不足时明确说明。回答要完整但不堆砌：先直接回答问题，再结合资料补充形制特征、用途、历史背景、"
        "纹饰或铭文意义中最相关的内容。通常写成2至4个短段落，约220至400个汉字；简单事实问题可适当缩短。"
        "不要把相似度或模型判断写成学术鉴定结论，并在相关句末用[1]等编号标注来源。"
        f"当前识别展品：{exhibit_label}\n检索资料：\n{sources or '没有检索到相关资料。'}"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(m for m in history[-8:] if m.get("role") in {"user", "assistant"} and m.get("content"))
    messages.append({"role": "user", "content": question})
    result = client.chat.completions.create(model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"), messages=messages, temperature=0.2)
    return result.choices[0].message.content or "暂时无法生成回答。"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/exhibits")
def list_exhibits() -> list[dict[str, str]]:
    return [{"id": e["id"], "name": e["name"], "period": e["period"]} for e in knowledge.exhibits]


@app.get("/api/exhibits/{exhibit_id}")
def get_exhibit(exhibit_id: str) -> dict[str, Any]:
    exhibit = knowledge.by_id.get(exhibit_id)
    if not exhibit:
        raise HTTPException(status_code=404, detail="未找到该展品。")
    return exhibit


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "vision_model_loaded": matcher.loaded,
        "vessel_classifier_loaded": matcher.classifier_loaded,
        "vessel_classifier_error": matcher.classifier_error,
        "image_count": matcher.image_count,
        "embedding_model_loaded": knowledge.embedding_loaded,
        "embedding_backend": knowledge.embedding_backend,
        "embedding_model": knowledge.model_path,
        "api_key_configured": bool(os.getenv("DASHSCOPE_API_KEY", "").strip()),
    }


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传图片文件。")
    data = await file.read(12 * 1024 * 1024 + 1)
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片不能超过 12 MB。")
    try:
        result = matcher.match(data, top_k=3)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    vessel_type = result["vessel_type"]
    candidates = result["exhibit_candidates"]
    best = candidates[0] if candidates else None
    margin = best["score"] - candidates[1]["score"] if best and len(candidates) > 1 else (best["score"] if best else 0.0)
    if vessel_type["confidence"] < TYPE_THRESHOLD:
        decision = "retry_with_top3"
        exhibit = None
    elif best and best["score"] >= EXHIBIT_THRESHOLD and margin >= MARGIN_THRESHOLD:
        decision = "specific_exhibit"
        exhibit = knowledge.by_id.get(best["exhibit_id"])
    else:
        decision = "vessel_type_only"
        exhibit = None
    return {
        **result,
        "exhibit": exhibit,
        "candidates": candidates,
        "exhibit_similarity": best["score"] if best else None,
        "match_margin": round(margin, 4),
        "decision": decision,
        "needs_confirmation": decision == "retry_with_top3",
        "allow_type_confirmation": decision == "retry_with_top3",
        "allow_exhibit_confirmation": bool(
            decision == "vessel_type_only" and best and best["score"] >= EXHIBIT_CANDIDATE_FLOOR
        ),
        "thresholds": {
            "vessel_type": TYPE_THRESHOLD,
            "exhibit_similarity": EXHIBIT_THRESHOLD,
            "exhibit_candidate_floor": EXHIBIT_CANDIDATE_FLOOR,
        },
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空。")
    exhibit = knowledge.by_id.get(request.exhibit_id) if request.exhibit_id else None
    passages = knowledge.search(request.question, exhibit_id=request.exhibit_id, vessel_type=request.vessel_type)
    try:
        answer = _answer(request.question, exhibit, request.vessel_type, passages, request.history)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"千问 API 调用失败：{exc}") from exc
    sources = [{"name": p["name"], "url": p["source_url"], "license": p["license"]} for p in passages]
    return {"answer": answer, "sources": sources}


@app.post("/api/guide")
def guide(exhibit_id: str = Form(""), vessel_type: str = Form(""), history: str = Form("[]")) -> dict[str, Any]:
    if not exhibit_id and vessel_type not in VESSEL_TYPE_NAMES:
        raise HTTPException(status_code=422, detail="请先获得可靠的器型识别结果。")
    if exhibit_id and exhibit_id not in knowledge.by_id:
        raise HTTPException(status_code=404, detail="未找到该展品。")
    try:
        messages = json.loads(history)
        if not isinstance(messages, list):
            messages = []
    except json.JSONDecodeError:
        messages = []
    exhibit = knowledge.by_id.get(exhibit_id)
    target_type = vessel_type or (exhibit.get("vessel_type") if exhibit else "")
    label = exhibit["name"] if exhibit else VESSEL_TYPE_NAMES[target_type]
    passages = knowledge.search(label + " 年代 器型 纹饰 用途", exhibit_id=exhibit_id or None, vessel_type=target_type)
    try:
        prompt = (
            "请为参观者生成一段较完整的展品讲解，约250至350个汉字。先概括年代与器类，再介绍最有辨识度的形制、"
            "纹饰或铭文，最后说明其历史价值；分成2至4个短段落。"
            if exhibit
            else "请生成一段较完整的器型讲解，约250至350个汉字。介绍典型形制、原始用途、礼制意义和图像识别要点，"
            "分成2至4个短段落，不要假定它是某件具体藏品。"
        )
        answer = _answer(prompt, exhibit, target_type, passages, messages)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"千问 API 调用失败：{exc}") from exc
    return {"answer": answer, "sources": [{"name": p["name"], "url": p["source_url"], "license": p["license"]} for p in passages]}
