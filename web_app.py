import logging
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from psb_citations import EXPLICIT_POINT, IDENTIFIER, POINT, REFERENCE
from psb_engine import Chunk, Proxy, Retriever, build_chunks, load_docs, safe_lines
from solution import DirectAssistant, _envint


LOG = logging.getLogger("legal_web")
STATIC = Path(__file__).with_name("web")


@dataclass
class EvidenceTrace:
    mapping: dict[str, Chunk] = field(default_factory=dict)
    raw: str = ""


TRACE: ContextVar[EvidenceTrace | None] = ContextVar("evidence_trace", default=None)


class ObservedProxy(Proxy):
    def chat(self, *args, **kwargs):
        raw = super().chat(*args, **kwargs)
        trace = TRACE.get()
        if trace is not None:
            trace.raw = raw
        return raw


class ObservedBundles:
    def __init__(self, bundles):
        self.bundles = bundles

    def pack(self, *args, **kwargs):
        context, mapping = self.bundles.pack(*args, **kwargs)
        trace = TRACE.get()
        if trace is not None:
            trace.mapping = mapping
        return context, mapping


def dialogue_input(question, history):
    normalise = lambda text: " ".join(text.split())
    latest = normalise(question)
    if not latest:
        raise ValueError("Введите вопрос.")
    turns = [latest]
    remaining = 2500 - len(latest)
    for turn in reversed(history):
        previous = normalise(turn)
        if not previous:
            continue
        if len(previous) + 1 > remaining:
            break
        turns.append(previous)
        remaining -= len(previous) + 1
    return "\n".join(f"Пользователь: {turn}" for turn in reversed(turns))


def collect_sources(raw, mapping, documents):
    sources = {}
    for ref in REFERENCE.finditer(raw):
        marker = ref.group(1) or ref.group(2)
        ids = list(IDENTIFIER.finditer(marker))
        for i, identifier in enumerate(ids):
            label = "E" + identifier.group(1).lstrip("0")
            chunk = mapping.get(label)
            if chunk is None:
                continue
            tail = marker[identifier.end():ids[i + 1].start() if i + 1 < len(ids) else len(marker)]
            section = chunk.section
            fragment = chunk.text.strip()
            explicit = EXPLICIT_POINT.search(tail)
            if explicit and chunk.category == "company":
                points = list(POINT.finditer(chunk.text))
                for j, point in enumerate(points):
                    if point.group(1) == explicit.group(1):
                        section = "п. " + point.group(1)
                        end = points[j + 1].start() if j + 1 < len(points) else len(chunk.text)
                        fragment = chunk.text[point.start():end].strip()
                        break
            key = (chunk.idx, section)
            if key in sources:
                continue
            sources[key] = {
                "title": chunk.title,
                "section": section,
                "fragment": fragment if len(fragment) <= 360 else fragment[:359].rstrip() + "…",
                "text": documents.get(chunk.filename, chunk.text),
            }
    return list(sources.values())


class WebRag:
    def __init__(self, proxy):
        data_dir = Path(os.getenv("DATA_DIR") or Path(__file__).with_name("data"))
        cache_dir = Path(os.getenv("WEB_CACHE_DIR") or Path(__file__).with_name(".web-cache"))
        docs = load_docs(data_dir / "knowledge-base.zip")
        chunks = build_chunks(docs, chunk_size=_envint("PSB_CHUNK_SIZE", 1150, 600, 2200))
        self.documents = {name: safe_lines(text).strip() for name, text in docs}
        self.retriever = Retriever(chunks, proxy=proxy, index_path=cache_dir / "kb_vectors.npz")
        self.assistant = DirectAssistant(self.retriever, proxy)
        self.assistant.evidence = ObservedBundles(self.assistant.evidence)
        self.index_lock = Lock()
        self.index_attempted = False

    def prepare_index(self):
        with self.index_lock:
            if self.index_attempted:
                return
            self.index_attempted = True
            if self.retriever.vectors is not None or os.getenv("PSB_SKIP_EMBEDDINGS") in ("1", "true"):
                return
            try:
                self.retriever.path.parent.mkdir(parents=True, exist_ok=True)
                self.retriever.build_index(batch=_envint("PSB_EMBED_BATCH", 16, 4, 32))
            except Exception as exc:
                LOG.warning("Embeddings failed, using lexical retrieval: %s", type(exc).__name__)

    def answer(self, question):
        self.prepare_index()
        trace = EvidenceTrace()
        token = TRACE.set(trace)
        try:
            answer = self.assistant.answer(question)
            return {"answer": answer, "sources": collect_sources(trace.raw, trace.mapping, self.documents)}
        finally:
            TRACE.reset(token)


@asynccontextmanager
async def lifespan(app):
    proxy = ObservedProxy()
    app.state.proxy = proxy
    app.state.rag = None
    try:
        try:
            app.state.rag = WebRag(proxy)
        except Exception as exc:
            LOG.error("Knowledge base could not be loaded: %s", type(exc).__name__)
        yield
    finally:
        proxy.client.close()


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatInput(BaseModel):
    question: Question
    history: list[Question] = Field(default_factory=list, max_length=12)


app = FastAPI(title="PSB Legal RAG", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status(request: Request):
    return {
        "knowledge_base": request.app.state.rag is not None,
        "proxy_configured": request.app.state.proxy.available,
    }


@app.post("/api/chat")
def chat(payload: ChatInput, request: Request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Отправьте вопрос в формате JSON.")
    if request.app.state.rag is None:
        raise HTTPException(503, "Не удалось загрузить базу. Проверьте knowledge-base.zip в DATA_DIR.")
    if not request.app.state.proxy.available:
        raise HTTPException(503, "Прокси не настроен. Проверьте настройки LLM на сервере.")
    try:
        return request.app.state.rag.answer(dialogue_input(payload.question, payload.history))
    except httpx.TimeoutException:
        raise HTTPException(504, "Прокси не ответил вовремя. Повторите попытку.") from None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            message = "Прокси отклонил доступ. Проверьте ключ и права на модель."
        elif exc.response.status_code == 429:
            message = "Прокси ограничил частоту запросов. Повторите попытку позже."
        else:
            message = "Прокси вернул ошибку. Проверьте подключение и название модели."
        raise HTTPException(502, message) from None
    except httpx.RequestError:
        raise HTTPException(502, "Не удалось подключиться к прокси. Повторите попытку.") from None
    except Exception as exc:
        LOG.error("Answer failed: %s", type(exc).__name__)
        raise HTTPException(502, "Не удалось получить ответ. Проверьте журнал сервера.") from None
