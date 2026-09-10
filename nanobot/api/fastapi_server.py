"""FastAPI implementation of the OpenAI-compatible Agent API.

This adapter intentionally shares the same AgentLoop contract as the aiohttp
server: a request is translated into ``agent_loop.process_direct`` and the
result is returned in OpenAI-compatible JSON or SSE chunks.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised by optional install
    raise ImportError(
        "FastAPI support requires: pip install 'nanobot-ai[fastapi]'"
    ) from exc

from nanobot.config.paths import get_media_dir
from nanobot.utils.media_decode import (
    MAX_FILE_SIZE,
    FileSizeExceeded,
    save_base64_data_url,
)
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    session_id: str | None = None


def _error_payload(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": status}},
    )


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


def _completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta} if delta else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_content(message: ChatMessage) -> tuple[str, list[str]]:
    """Parse the single supported user message and persist base64 images."""
    if message.role != "user":
        raise ValueError("Only a single user message is supported")
    media_dir = get_media_dir("api-fastapi")
    media_paths: list[str] = []
    content = message.content
    if isinstance(content, str):
        return content, media_paths

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if url.startswith("data:"):
                saved = save_base64_data_url(url, media_dir)
                if saved:
                    media_paths.append(saved)
            elif url:
                raise ValueError(
                    "Remote image URLs are not supported. Use base64 data URLs."
                )
    return " ".join(text_parts), media_paths


def create_fastapi_app(
    agent_loop,
    model_name: str = "agent",
    request_timeout: float = 120.0,
) -> FastAPI:
    """Create a FastAPI application around an initialized AgentLoop."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        connect = getattr(agent_loop, "_connect_mcp", None)
        if connect is not None:
            await connect()
        try:
            yield
        finally:
            close = getattr(agent_loop, "close_mcp", None)
            if close is not None:
                await close()

    app = FastAPI(title="Agent API", version="1.0", lifespan=lifespan)
    app.state.agent_loop = agent_loop
    app.state.model_name = model_name
    app.state.request_timeout = request_timeout
    app.state.session_locks: dict[str, asyncio.Lock] = {}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": app.state.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "agent",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        if len(payload.messages) != 1:
            return _error_payload(400, "Only a single user message is supported")
        if payload.model and payload.model != app.state.model_name:
            return _error_payload(
                400, f"Only configured model '{app.state.model_name}' is available"
            )
        try:
            text, media_paths = _parse_content(payload.messages[0])
        except ValueError as exc:
            return _error_payload(400, str(exc))
        except FileSizeExceeded as exc:
            return _error_payload(413, str(exc))
        except Exception:
            logger.exception("Error parsing FastAPI request")
            return _error_payload(413, "File too large or invalid upload")

        session_key = f"api:{payload.session_id}" if payload.session_id else API_SESSION_KEY
        lock = app.state.session_locks.setdefault(session_key, asyncio.Lock())
        agent = app.state.agent_loop
        timeout = app.state.request_timeout
        model = app.state.model_name

        if payload.stream:
            async def event_stream():
                queue: asyncio.Queue[str | None] = asyncio.Queue()
                failed = False

                async def on_stream(delta: str) -> None:
                    await queue.put(delta)

                async def on_stream_end(*_args: Any, **_kwargs: Any) -> None:
                    await queue.put(None)

                async def run_agent() -> None:
                    nonlocal failed
                    try:
                        async with lock:
                            await asyncio.wait_for(
                                agent.process_direct(
                                    content=text,
                                    media=media_paths or None,
                                    session_key=session_key,
                                    channel="api",
                                    chat_id=API_CHAT_ID,
                                    on_stream=on_stream,
                                    on_stream_end=on_stream_end,
                                ),
                                timeout=timeout,
                            )
                    except Exception:
                        failed = True
                        logger.exception("FastAPI streaming error for session {}", session_key)
                        await queue.put(None)

                task = asyncio.create_task(run_agent())
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                try:
                    while True:
                        token = await queue.get()
                        if token is None:
                            break
                        yield _sse_chunk(token, model, chunk_id)
                    if not failed:
                        yield _sse_chunk("", model, chunk_id, finish_reason="stop")
                        yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    task.cancel()
                    raise
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        try:
            async with lock:
                response = await asyncio.wait_for(
                    agent.process_direct(
                        content=text,
                        media=media_paths or None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout,
                )
                response_text = _response_text(response)
                if not response_text.strip():
                    response = await asyncio.wait_for(
                        agent.process_direct(
                            content=text,
                            media=media_paths or None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                        ),
                        timeout=timeout,
                    )
                    response_text = _response_text(response) or EMPTY_FINAL_RESPONSE_MESSAGE
        except asyncio.TimeoutError:
            return _error_payload(504, f"Request timed out after {timeout}s", "timeout")
        except Exception:
            logger.exception("FastAPI request failed for session {}", session_key)
            return _error_payload(500, "Internal server error", "server_error")

        return JSONResponse(_completion_response(response_text, model))

    return app


__all__ = ["ChatCompletionRequest", "ChatMessage", "create_fastapi_app", "MAX_FILE_SIZE"]
