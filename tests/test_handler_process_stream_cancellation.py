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


async def _wait_for_pending_id(proxy: HandlerProcessProxy, req_id: str) -> None:
    """Wait until the proxy has created the pending queue for ``req_id``."""

    for _ in range(20):
        if req_id in proxy._pending:
            return
        await asyncio.sleep(0)
    msg = f"Timed out waiting for pending id {req_id}"
    raise AssertionError(msg)


@pytest.mark.asyncio
async def test_call_stream_enqueues_single_cancel_on_early_consumer_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early stream shutdown should enqueue exactly one cancel control message."""

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
    proxy._lazy_queue_config = {"queue_size": 4, "timeout": 19}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Mark the proxy as started without spawning a real child process."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_id = "req-stream-disconnect"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    stream = proxy._call_stream("generate_text_stream", "hello")
    first_chunk_task = asyncio.create_task(stream.__anext__())
    await _wait_for_pending_id(proxy, req_id)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})

    assert await first_chunk_task == "hello"

    await stream.aclose()

    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]
    assert proxy._control_queue.items == [{"id": req_id, "method": handler_process_module._CANCEL}]


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
    proxy._lazy_queue_config = {"queue_size": 4, "timeout": 19}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Mark the proxy as started without spawning a real child process."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_id = "req-stream-ok"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    async def _collect_stream() -> list[str]:
        return [chunk async for chunk in proxy._call_stream("generate_text_stream", "hello")]

    stream_task = asyncio.create_task(_collect_stream())
    await _wait_for_pending_id(proxy, req_id)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})
    await result_queue.put({"type": handler_process_module._STREAM_END})

    assert await stream_task == ["hello"]
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]
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
    proxy._lazy_queue_config = {"queue_size": 4, "timeout": 19}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Mark the proxy as started without spawning a real child process."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_id = "req-stream-error"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    async def _collect_stream() -> list[str]:
        return [chunk async for chunk in proxy._call_stream("generate_text_stream", "hello")]

    stream_task = asyncio.create_task(_collect_stream())
    await _wait_for_pending_id(proxy, req_id)

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
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]
    assert proxy._control_queue.items == []
