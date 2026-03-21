"""Regression tests for stream cancellation bookkeeping in handler proxies."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException
import pytest

from app.config import ModelEntryConfig
from app.core import handler_process as handler_process_module
from app.core.handler_process import HandlerProcessProxy


class _CaptureQueue:
    """Minimal queue stub that records items put into it."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        """Record an item in FIFO order."""

        self.items.append(item)


@pytest.mark.asyncio
async def test_call_stream_does_not_enqueue_cancel_after_normal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful streams should not enqueue a late cancel control message."""

    model_cfg = ModelEntryConfig(
        model_path="dummy-whisper-model",
        model_type="whisper",
        model_id="dummy-whisper-model",
    )
    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg.__dict__.copy(),
        model_type=model_cfg.model_type,
        model_path=model_cfg.model_path,
        model_id=model_cfg.model_id,
    )
    proxy._request_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._control_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._rpc_timeout = 1.0
    proxy._stream_queue_size = 4

    req_id = "req-stream-ok"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    async def _collect_stream() -> list[str]:
        return [chunk async for chunk in proxy._call_stream("generate_text_stream", "hello")]

    stream_task = asyncio.create_task(_collect_stream())
    await asyncio.sleep(0)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})
    await result_queue.put({"type": handler_process_module._STREAM_END})

    assert await stream_task == ["hello"]
    assert proxy._control_queue.items == []


@pytest.mark.asyncio
async def test_call_stream_does_not_enqueue_cancel_after_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal stream errors should not enqueue a late cancel control message."""

    model_cfg = ModelEntryConfig(
        model_path="dummy-whisper-model",
        model_type="whisper",
        model_id="dummy-whisper-model",
    )
    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg.__dict__.copy(),
        model_type=model_cfg.model_type,
        model_path=model_cfg.model_path,
        model_id=model_cfg.model_id,
    )
    proxy._request_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._control_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._rpc_timeout = 1.0
    proxy._stream_queue_size = 4

    req_id = "req-stream-error"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    async def _collect_stream() -> list[str]:
        return [chunk async for chunk in proxy._call_stream("generate_text_stream", "hello")]

    stream_task = asyncio.create_task(_collect_stream())
    await asyncio.sleep(0)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})
    await result_queue.put(
        {
            "type": "error",
            "status_code": 500,
            "detail": {"message": "boom"},
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await stream_task

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == {"message": "boom"}
    assert proxy._control_queue.items == []


@pytest.mark.asyncio
async def test_call_stream_enqueues_cancel_once_after_early_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing a live stream early should forward exactly one cancel signal."""

    model_cfg = ModelEntryConfig(
        model_path="dummy-whisper-model",
        model_type="whisper",
        model_id="dummy-whisper-model",
    )
    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg.__dict__.copy(),
        model_type=model_cfg.model_type,
        model_path=model_cfg.model_path,
        model_id=model_cfg.model_id,
    )
    proxy._request_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._control_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._rpc_timeout = 1.0
    proxy._stream_queue_size = 4

    req_id = "req-stream-disconnect"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    stream = proxy._call_stream("generate_text_stream", "hello")
    first_chunk_task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})

    assert await first_chunk_task == "hello"

    await stream.aclose()

    assert proxy._control_queue.items == [{"id": req_id, "method": handler_process_module._CANCEL}]
