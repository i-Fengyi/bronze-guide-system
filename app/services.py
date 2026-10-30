from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.vision import GalleryEncoder, VesselClassifier


class KnowledgeBase:
    def __init__(self, path: Path, vessel_types_path: Path, model_path: str) -> None:
        self.exhibits: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        self.vessel_types: list[dict[str, Any]] = json.loads(vessel_types_path.read_text(encoding="utf-8"))
        self.records = [*self.exhibits, *self.vessel_types]
        self.by_id = {item["id"]: item for item in self.exhibits}
        self.record_by_id = {item["id"]: item for item in self.records}
        self.model_path = model_path
        self._encoder = None
        self.embedding_backend = "local-hash-ngram"
        self._collection = self._build_collection(path.parent / "chroma")

    @property
    def embedding_loaded(self) -> bool:
        return self._encoder is not None

    def _build_collection(self, persist_dir: Path):
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(self.model_path)
        if model_dir.exists():
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(str(model_dir))
                self.embedding_backend = f"sentence-transformers:{model_dir.name}"
            except Exception:
                self._encoder = None

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(name="bronze_nmc_v1", metadata={"hnsw:space": "cosine"})
        documents = [self._document_text(item) for item in self.records]
        desired_ids = {item["id"] for item in self.records}
        stale_ids = set(collection.get()["ids"]) - desired_ids
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
        collection.upsert(
            ids=[item["id"] for item in self.records],
            documents=documents,
            metadatas=[
                {
                    "record_id": item["id"],
                    "name": item["name"],
                    "source_url": item["source_url"],
                    "vessel_type": item.get("vessel_type") or "unknown",
                    "entity_type": item.get("entity_type", "exhibit"),
                }
                for item in self.records
            ],
            embeddings=self._encode_texts(documents),
        )
        return collection

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is not None:
            vectors = self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vectors.astype("float32").tolist()
        return [self._hash_embedding(text) for text in texts]

    @staticmethod
    def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
        normalized = "".join(text.lower().split())
        vector = np.zeros(dimensions, dtype="float32")
        tokens = [normalized[i : i + size] for size in (1, 2, 3) for i in range(max(0, len(normalized) - size + 1))]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    @staticmethod
    def _document_text(item: dict[str, Any]) -> str:
        parts = [
            f"名称：{item['name']}",
            f"年代：{item.get('period', '资料未注明')}",
            f"类别：{item.get('category', '资料未注明')}",
        ]
        if item.get("dimensions"):
            parts.append(f"尺寸：{item['dimensions']}")
        parts.append(f"资料介绍：{item['description']}")
        if item.get("keywords"):
            parts.append(f"相关关键词：{'、'.join(item['keywords'])}")
        return "\n".join(parts)

    def search(
        self,
        query: str,
        exhibit_id: str | None = None,
        vessel_type: str | None = None,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        if exhibit_id in self.by_id:
            exhibit = self.by_id[exhibit_id]
            selected = [exhibit]
            type_record = self.record_by_id.get(f"type_{exhibit.get('vessel_type', '')}")
            if type_record:
                selected.append(type_record)
            return [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "text": self._document_text(item),
                    "source_url": item["source_url"],
                    "license": item["license"],
                }
                for item in selected
            ]
        if vessel_type:
            where = {"vessel_type": vessel_type}
        else:
            where = None
        matching_count = sum(1 for item in self.records if item.get("vessel_type") == vessel_type) if vessel_type else len(self.records)
        result = self._collection.query(
            query_embeddings=self._encode_texts([query]),
            n_results=min(top_k, max(1, matching_count)),
            where=where,
        )
        selected = [self.record_by_id[item_id] for item_id in result["ids"][0]]
        return [{"id": item["id"], "name": item["name"], "text": self._document_text(item), "source_url": item["source_url"], "license": item["license"]} for item in selected]


class ExhibitMatcher:
    def __init__(
        self,
        image_dir: Path,
        exhibits: list[dict[str, Any]],
        weights_path: str = "",
        classifier_path: str = "models/vessel_classifier/best.pt",
        gallery_split_path: str = "data/exhibit_gallery_split.json",
    ) -> None:
        self.image_dir = image_dir
        self.exhibits = exhibits
        root = Path(__file__).resolve().parents[1]
        feature_path = Path(weights_path) if weights_path else root / "models" / "resnet-18"
        classifier = Path(classifier_path)
        split_path = Path(gallery_split_path)
        self.classifier = VesselClassifier(classifier if classifier.is_absolute() else root / classifier)
        self.encoder = GalleryEncoder(feature_path if feature_path.is_absolute() else root / feature_path)
        self.gallery_split_path = split_path if split_path.is_absolute() else root / split_path
        self._gallery: list[tuple[str, str | None, np.ndarray]] | None = None

    def _reference_paths(self, exhibit_id: str, gallery_dir: Path) -> list[Path]:
        if self.gallery_split_path.exists():
            split = json.loads(self.gallery_split_path.read_text(encoding="utf-8"))
            references = split.get("exhibits", {}).get(exhibit_id, {}).get("reference", [])
            if references:
                return [self.image_dir / relative for relative in references]
        return sorted(gallery_dir.rglob("*"))

    @property
    def loaded(self) -> bool:
        self.classifier._load()
        return self.classifier.loaded

    @property
    def classifier_loaded(self) -> bool:
        self.classifier._load()
        return self.classifier.loaded

    @property
    def classifier_error(self) -> str | None:
        self.classifier._load()
        return self.classifier.load_error

    @property
    def image_count(self) -> int:
        total = 0
        for item in self.exhibits:
            gallery_dir = self.image_dir / item.get("gallery_dir", item["id"])
            total += sum(1 for path in self._reference_paths(item["id"], gallery_dir) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        return total

    def _load_gallery(self) -> list[tuple[str, str | None, np.ndarray]]:
        if self._gallery is not None:
            return self._gallery
        gallery: list[tuple[str, str | None, np.ndarray]] = []
        by_id = {item["id"]: item for item in self.exhibits}
        for item in self.exhibits:
            exhibit_id = item["id"]
            gallery_dir = self.image_dir / item.get("gallery_dir", exhibit_id)
            if not gallery_dir.exists():
                continue
            for path in self._reference_paths(exhibit_id, gallery_dir):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                try:
                    gallery.append((exhibit_id, by_id[exhibit_id].get("vessel_type"), self.encoder.encode(Image.open(path))))
                except Exception:
                    continue
        self._gallery = gallery
        return gallery

    def match(self, data: bytes, top_k: int = 3) -> dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise RuntimeError("无法读取图片，请使用 JPG、PNG 或 WebP 格式。") from exc
        type_candidates = self.classifier.predict(image, top_k=3)
        if not type_candidates:
            raise RuntimeError(
                "器型分类模型尚未训练或无法加载。请先运行 scripts/train_classifier.py，"
                f"详细信息：{self.classifier_error or '未找到权重'}"
            )
        predicted_type = type_candidates[0]["label"]
        query = self.encoder.encode(image)
        gallery = self._load_gallery()
        scores: dict[str, list[float]] = {}
        for exhibit_id, vessel_type, vector in gallery:
            if vessel_type != predicted_type:
                continue
            scores.setdefault(exhibit_id, []).append(float(np.dot(query, vector)))
        ranked = sorted(((max(values), exhibit_id) for exhibit_id, values in scores.items()), reverse=True)[:top_k]
        names = {item["id"]: item["name"] for item in self.exhibits}
        return {
            "vessel_type": type_candidates[0],
            "type_candidates": type_candidates,
            "exhibit_candidates": [
                {"exhibit_id": exhibit_id, "name": names[exhibit_id], "score": round(score, 4)}
                for score, exhibit_id in ranked
            ],
        }
