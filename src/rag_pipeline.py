"""Chunker, vector database creator, and retriever."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader

from src.config import DATA_DIR, DB_DIR, AppConfig

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        RecursiveCharacterTextSplitter = None


class SupportEmbeddingFunction(EmbeddingFunction[Documents]):
    """Gemini embeddings with a deterministic local fallback for demos/tests."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None
        if config.gemini_api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=config.gemini_api_key)
            except Exception:
                self._client = None

    def __call__(self, input: Documents) -> Embeddings:
        if self._client:
            try:
                response = self._client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=list(input),
                )
                return [item.values for item in response.embeddings]
            except Exception:
                pass
        return [_hash_embedding(text) for text in input]


def build_or_load_collection(config: AppConfig, rebuild: bool = False):
    DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if rebuild:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=SupportEmbeddingFunction(config),
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        chunks = load_and_chunk_documents(config)
        if chunks:
            collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
    return collection


def retrieve(query: str, config: AppConfig) -> tuple[list[dict[str, Any]], float]:
    collection = build_or_load_collection(config)
    result = collection.query(query_texts=[query], n_results=config.top_k)

    chunks: list[dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        score = max(0.0, 1.0 - float(distance))
        chunks.append(
            {
                "text": document,
                "source": metadata.get("source", "unknown"),
                "score": score,
            }
        )

    best_score = chunks[0]["score"] if chunks else 0.0
    return chunks, best_score


def load_and_chunk_documents(config: AppConfig) -> list[dict[str, Any]]:
    splitter = _make_splitter(config)
    chunks: list[dict[str, Any]] = []

    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        text = _read_document(path)
        if not text.strip():
            continue
        for index, chunk in enumerate(splitter(text)):
            chunks.append(
                {
                    "id": f"{path.stem}-{index}",
                    "text": chunk,
                    "metadata": {"source": path.name, "chunk": index},
                }
            )
    return chunks


def _make_splitter(config: AppConfig):
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text

    def split_text(text: str) -> list[str]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if len(candidate) <= config.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = paragraph

        if current:
            chunks.append(current)

        if not chunks and text.strip():
            step = max(1, config.chunk_size - config.chunk_overlap)
            return [text[index : index + config.chunk_size] for index in range(0, len(text), step)]

        return chunks

    return split_text


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
