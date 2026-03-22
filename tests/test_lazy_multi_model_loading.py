"""Fail-first tests for lazy multi-model handler startup."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from typing import Any
import uuid

import pytest

from app.config import ModelEntryConfig, MultiModelServerConfig
from app.core import handler_process as handler_process_module
from app.core.handler_process import HandlerProcessProxy
from app.schemas.openai import ChatCompletionRequest, Message


class _CaptureQueue:
    """Minimal queue stub that records items put into it."""

    def __init__(
        self,
        *,
        event_log: list[str] | None = None,
        event_name: str = "enqueue",
    ) -> None:
        """Initialize an empty FIFO capture list."""

        self.items: list[dict[str, Any]] = []
        self._event_log = event_log
        self._event_name = event_name

    def put(self, item: dict[str, Any]) -> None:
        """Record an item in FIFO order."""

        if self._event_log is not None:
            self._event_log.append(self._event_name)
        self.items.append(item)


class _ReadyThenStopQueue:
    """Queue stub that emits one ready message, then stops the reader."""

    def __init__(self, *, on_first_get: Any | None = None) -> None:
        self._first = True
        self._on_first_get = on_first_get

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        """Return a ready message once, then raise to terminate the reader."""

        del timeout
        if self._first:
            self._first = False
            if self._on_first_get is not None:
                self._on_first_get()
            return {"type": "ready", "success": True}
        msg = "stop reader"
        raise RuntimeError(msg)


async def _wait_for_pending_ids(
    proxy: HandlerProcessProxy,
    expected_ids: set[str],
) -> None:
    """Wait until the proxy has registered all expected pending request ids."""

    for _ in range(20):
        if expected_ids.issubset(proxy._pending):
            return
        await asyncio.sleep(0)
    msg = f"Timed out waiting for pending ids {sorted(expected_ids)}"
    raise AssertionError(msg)


async def _wait_for_queue_size(capture_queue: _CaptureQueue, size: int) -> None:
    """Wait until the capture queue contains ``size`` items."""

    for _ in range(20):
        if len(capture_queue.items) >= size:
            return
        await asyncio.sleep(0)
    msg = f"Timed out waiting for capture queue size {size}"
    raise AssertionError(msg)


def _load_endpoints_module() -> Any:
    """Import ``app.api.endpoints`` with lightweight handler stubs."""

    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    fake_vlm_module = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm_module.MLXVLMHandler = object

    module_names = [
        "app.handler.mlx_lm",
        "app.handler.mlx_vlm",
        "app.api.endpoints",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules["app.handler.mlx_vlm"] = fake_vlm_module
        sys.modules.pop("app.api.endpoints", None)
        return importlib.import_module("app.api.endpoints")
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_server_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import ``app.server`` with lightweight runtime and handler stubs."""

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = object
    fake_fastapi.Request = object

    fake_fastapi_cors = types.ModuleType("fastapi.middleware.cors")
    fake_fastapi_cors.CORSMiddleware = object

    fake_fastapi_responses = types.ModuleType("fastapi.responses")
    fake_fastapi_responses.JSONResponse = object

    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.clear_cache = lambda: None

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.Config = object

    fake_endpoints = types.ModuleType("app.api.endpoints")
    fake_endpoints.router = object()

    fake_registry = types.ModuleType("app.core.model_registry")
    fake_registry.ModelRegistry = object

    fake_handler = types.ModuleType("app.handler")
    fake_handler.MLXFluxHandler = object

    fake_lm = types.ModuleType("app.handler.mlx_lm")
    fake_lm.MLXLMHandler = object

    fake_vlm = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm.MLXVLMHandler = object

    fake_embeddings = types.ModuleType("app.handler.mlx_embeddings")
    fake_embeddings.MLXEmbeddingsHandler = object

    fake_whisper = types.ModuleType("app.handler.mlx_whisper")
    fake_whisper.MLXWhisperHandler = object

    module_overrides: dict[str, types.ModuleType] = {
        "fastapi": fake_fastapi,
        "fastapi.middleware.cors": fake_fastapi_cors,
        "fastapi.responses": fake_fastapi_responses,
        "mlx.core": fake_mx_core,
        "uvicorn": fake_uvicorn,
        "app.api.endpoints": fake_endpoints,
        "app.core.model_registry": fake_registry,
        "app.handler": fake_handler,
        "app.handler.mlx_lm": fake_lm,
        "app.handler.mlx_vlm": fake_vlm,
        "app.handler.mlx_embeddings": fake_embeddings,
        "app.handler.mlx_whisper": fake_whisper,
    }

    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in [*module_overrides, "app.server"]
    }

    try:
        for name, module in module_overrides.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.delitem(sys.modules, "app.server", raising=False)
        return importlib.import_module("app.server")
    finally:
        sys.modules.pop("app.server", None)
        for name, module in original_modules.items():
            if name == "app.server":
                continue
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.asyncio
async def test_create_multi_lifespan_registers_models_without_eager_proxy_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should register configured models without eagerly loading each child."""

    server_module = _load_server_module(monkeypatch)
    created_proxies: dict[str, _FakeProxy] = {}

    class _FakeRegistry:
        """Minimal async registry stub for lifespan tests."""

        def __init__(self) -> None:
            """Initialize registry capture state."""

            self.handlers: dict[str, Any] = {}

        async def register_model(
            self,
            model_id: str,
            handler: Any,
            model_type: str,
            context_length: int | None = None,
        ) -> None:
            """Record a handler registration."""

            del model_type, context_length
            self.handlers[model_id] = handler

        def get_handler(self, model_id: str) -> Any:
            """Return a previously registered handler."""

            return self.handlers[model_id]

        def get_model_count(self) -> int:
            """Return the number of configured handlers."""

            return len(self.handlers)

        async def cleanup_all(self) -> None:
            """Satisfy the shutdown interface."""

    class _FakeProxy:
        """Proxy stub that records whether startup warmed the child."""

        def __init__(
            self,
            model_cfg_dict: dict[str, Any],
            model_type: str,
            model_path: str,
            model_id: str,
        ) -> None:
            """Store model identity and track start calls."""

            del model_cfg_dict, model_type, model_path
            self.model_id = model_id
            self.start_calls: list[dict[str, Any]] = []
            created_proxies[model_id] = self

        async def start(self, queue_config: dict[str, Any]) -> None:
            """Record an eager start attempt."""

            self.start_calls.append(queue_config.copy())

        async def cleanup(self) -> None:
            """Satisfy the handler cleanup interface."""

    monkeypatch.setattr(server_module, "ModelRegistry", _FakeRegistry)
    monkeypatch.setattr(server_module, "HandlerProcessProxy", _FakeProxy)

    config = MultiModelServerConfig(
        models=[
            ModelEntryConfig(
                model_path="mlx-community/model-a",
                model_type="lm",
                model_id="model-a",
                queue_timeout=31,
                queue_size=7,
            ),
            ModelEntryConfig(
                model_path="mlx-community/model-b",
                model_type="whisper",
                model_id="model-b",
                queue_timeout=41,
                queue_size=9,
            ),
        ]
    )
    app = types.SimpleNamespace(state=types.SimpleNamespace())

    async with server_module.create_multi_lifespan(config)(app):
        assert sorted(app.state.registry.handlers) == ["model-a", "model-b"]
        assert app.state.handler is created_proxies["model-a"]
        assert created_proxies["model-a"].start_calls == []
        assert created_proxies["model-b"].start_calls == []


@pytest.mark.asyncio
async def test_create_multi_lifespan_preserves_queue_config_for_deferred_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should stash each model's eventual cold-start queue config on the proxy."""

    server_module = _load_server_module(monkeypatch)
    created_proxies: dict[str, _FakeProxy] = {}

    class _FakeRegistry:
        """Minimal registry stub for deferred queue-config tests."""

        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        async def register_model(
            self,
            model_id: str,
            handler: Any,
            model_type: str,
            context_length: int | None = None,
        ) -> None:
            del model_type, context_length
            self.handlers[model_id] = handler

        def get_handler(self, model_id: str) -> Any:
            return self.handlers[model_id]

        def get_model_count(self) -> int:
            return len(self.handlers)

        async def cleanup_all(self) -> None:
            return None

    class _FakeProxy:
        """Proxy stub exposing deferred queue-config storage for assertions."""

        def __init__(
            self,
            model_cfg_dict: dict[str, Any],
            model_type: str,
            model_path: str,
            model_id: str,
        ) -> None:
            del model_cfg_dict, model_type, model_path
            self.model_id = model_id
            self.start_calls: list[dict[str, Any]] = []
            self._lazy_queue_config: dict[str, Any] | None = None
            created_proxies[model_id] = self

        async def start(self, queue_config: dict[str, Any]) -> None:
            self.start_calls.append(queue_config.copy())

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(server_module, "ModelRegistry", _FakeRegistry)
    monkeypatch.setattr(server_module, "HandlerProcessProxy", _FakeProxy)

    config = MultiModelServerConfig(
        models=[
            ModelEntryConfig(
                model_path="mlx-community/model-a",
                model_type="lm",
                model_id="model-a",
                max_concurrency=3,
                queue_timeout=31,
                queue_size=7,
            ),
            ModelEntryConfig(
                model_path="mlx-community/model-b",
                model_type="whisper",
                model_id="model-b",
                max_concurrency=2,
                queue_timeout=41,
                queue_size=9,
            ),
        ]
    )
    app = types.SimpleNamespace(state=types.SimpleNamespace())

    async with server_module.create_multi_lifespan(config)(app):
        assert created_proxies["model-a"]._lazy_queue_config == {
            "max_concurrency": 3,
            "timeout": 31,
            "queue_size": 7,
        }
        assert created_proxies["model-b"]._lazy_queue_config == {
            "max_concurrency": 2,
            "timeout": 41,
            "queue_size": 9,
        }


@pytest.mark.asyncio
async def test_models_endpoint_lists_configured_models_without_warming_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured models should appear in ``/v1/models`` without eager child startup."""

    server_module = _load_server_module(monkeypatch)
    endpoints_module = _load_endpoints_module()
    created_proxies: dict[str, _FakeProxy] = {}

    class _FakeRegistry:
        """Registry stub that tracks configured models for the endpoint contract."""

        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}
            self._models: list[dict[str, Any]] = []

        async def register_model(
            self,
            model_id: str,
            handler: Any,
            model_type: str,
            context_length: int | None = None,
        ) -> None:
            del model_type, context_length
            self.handlers[model_id] = handler
            self._models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "organization-owner",
                }
            )

        def get_handler(self, model_id: str) -> Any:
            return self.handlers[model_id]

        def get_model_count(self) -> int:
            return len(self.handlers)

        def list_models(self) -> list[dict[str, Any]]:
            return list(self._models)

        async def cleanup_all(self) -> None:
            return None

    class _FakeProxy:
        """Proxy stub that records whether the endpoint path warmed a child."""

        def __init__(
            self,
            model_cfg_dict: dict[str, Any],
            model_type: str,
            model_path: str,
            model_id: str,
        ) -> None:
            del model_cfg_dict, model_type, model_path
            self.model_id = model_id
            self.start_calls: list[dict[str, Any]] = []
            created_proxies[model_id] = self

        async def start(self, queue_config: dict[str, Any]) -> None:
            self.start_calls.append(queue_config.copy())

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(server_module, "ModelRegistry", _FakeRegistry)
    monkeypatch.setattr(server_module, "HandlerProcessProxy", _FakeProxy)

    config = MultiModelServerConfig(
        models=[
            ModelEntryConfig(
                model_path="mlx-community/model-a",
                model_type="lm",
                model_id="model-a",
                queue_timeout=31,
                queue_size=7,
            ),
            ModelEntryConfig(
                model_path="mlx-community/model-b",
                model_type="whisper",
                model_id="model-b",
                queue_timeout=41,
                queue_size=9,
            ),
        ]
    )
    app = types.SimpleNamespace(state=types.SimpleNamespace())
    raw_request = types.SimpleNamespace(app=app, state=types.SimpleNamespace(request_id="req-test"))

    async with server_module.create_multi_lifespan(config)(app):
        response = await endpoints_module.models(raw_request)

        assert [model.id for model in response.data] == ["model-a", "model-b"]
        assert created_proxies["model-a"].start_calls == []
        assert created_proxies["model-b"].start_calls == []


@pytest.mark.asyncio
async def test_health_endpoint_reports_registered_models_as_lazy_unvalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health should report lazy-start models as registered, not yet proven loaded."""

    server_module = _load_server_module(monkeypatch)
    endpoints_module = _load_endpoints_module()
    created_proxies: dict[str, _FakeProxy] = {}

    class _FakeRegistry:
        """Registry stub that exposes configured models to ``/health``."""

        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}
            self._models: list[dict[str, Any]] = []

        async def register_model(
            self,
            model_id: str,
            handler: Any,
            model_type: str,
            context_length: int | None = None,
        ) -> None:
            del model_type, context_length
            self.handlers[model_id] = handler
            self._models.append({"id": model_id})

        def get_handler(self, model_id: str) -> Any:
            return self.handlers[model_id]

        def get_model_count(self) -> int:
            return len(self.handlers)

        def list_models(self) -> list[dict[str, Any]]:
            return list(self._models)

        async def cleanup_all(self) -> None:
            return None

    class _FakeProxy:
        """Proxy stub that records whether health warms the child."""

        def __init__(
            self,
            model_cfg_dict: dict[str, Any],
            model_type: str,
            model_path: str,
            model_id: str,
        ) -> None:
            del model_cfg_dict, model_type, model_path
            self.model_id = model_id
            self.start_calls: list[dict[str, Any]] = []
            created_proxies[model_id] = self

        async def start(self, queue_config: dict[str, Any]) -> None:
            self.start_calls.append(queue_config.copy())

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(server_module, "ModelRegistry", _FakeRegistry)
    monkeypatch.setattr(server_module, "HandlerProcessProxy", _FakeProxy)

    config = MultiModelServerConfig(
        models=[
            ModelEntryConfig(
                model_path="mlx-community/model-a",
                model_type="lm",
                model_id="model-a",
                queue_timeout=31,
                queue_size=7,
            ),
            ModelEntryConfig(
                model_path="mlx-community/model-b",
                model_type="whisper",
                model_id="model-b",
                queue_timeout=41,
                queue_size=9,
            ),
        ]
    )
    app = types.SimpleNamespace(state=types.SimpleNamespace())
    raw_request = types.SimpleNamespace(app=app, state=types.SimpleNamespace(request_id="req-test"))

    async with server_module.create_multi_lifespan(config)(app):
        response = await endpoints_module.health(raw_request)

        assert response.status == "ok"
        assert response.model_id == "model-a, model-b"
        assert response.model_status == "registered (2 model(s); lazy-start validation pending)"
        assert created_proxies["model-a"].start_calls == []
        assert created_proxies["model-b"].start_calls == []


@pytest.mark.asyncio
async def test_queue_stats_reports_unloaded_proxy_without_warming_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue stats should expose unloaded-proxy state without forcing a cold start."""

    endpoints_module = _load_endpoints_module()

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
    proxy._rpc_timeout = 1.0
    proxy._lazy_queue_config = {  # type: ignore[attr-defined]
        "max_concurrency": 3,
        "queue_size": 5,
        "timeout": 19,
    }

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Record any unexpected cold-start attempt."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    raw_request = types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(handler=proxy)),
        state=types.SimpleNamespace(request_id="req-test"),
    )

    response = await endpoints_module.queue_stats(raw_request)

    assert response["status"] == "ok"
    assert response["queue_stats"] == {
        "queue_stats": {
            "loaded": False,
            "model_id": "dummy-whisper-model",
            "model_status": "registered",
            "validation_status": "pending_first_request",
            "max_concurrency": 3,
            "queue_size": 5,
            "timeout": 19,
        }
    }
    assert start_calls == []
    assert proxy._request_queue.items == []


@pytest.mark.asyncio
async def test_chat_completions_first_request_to_unloaded_model_starts_exactly_one_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A routed first hit should cold-start exactly one selected proxy."""

    endpoints_module = _load_endpoints_module()

    selected_model_cfg = ModelEntryConfig(
        model_path=".",
        model_type="lm",
        model_id="model-a",
    )
    selected_proxy = HandlerProcessProxy(
        model_cfg_dict=selected_model_cfg.__dict__.copy(),
        model_type=selected_model_cfg.model_type,
        model_path=selected_model_cfg.model_path,
        model_id=selected_model_cfg.model_id,
    )
    selected_proxy._request_queue = _CaptureQueue()  # type: ignore[assignment]
    selected_proxy._rpc_timeout = 1.0
    selected_proxy._lazy_queue_config = {"queue_size": 5, "timeout": 19}  # type: ignore[attr-defined]

    unrelated_model_cfg = ModelEntryConfig(
        model_path="tests",
        model_type="lm",
        model_id="model-b",
    )
    unrelated_proxy = HandlerProcessProxy(
        model_cfg_dict=unrelated_model_cfg.__dict__.copy(),
        model_type=unrelated_model_cfg.model_type,
        model_path=unrelated_model_cfg.model_path,
        model_id=unrelated_model_cfg.model_id,
    )
    unrelated_proxy._request_queue = _CaptureQueue()  # type: ignore[assignment]
    unrelated_proxy._rpc_timeout = 1.0

    selected_start_calls: list[dict[str, Any]] = []
    unrelated_start_calls: list[dict[str, Any]] = []

    async def _selected_start(queue_config: dict[str, Any]) -> None:
        selected_start_calls.append(queue_config.copy())
        selected_proxy._process = object()  # type: ignore[assignment]
        selected_proxy._running = True

    async def _unrelated_start(queue_config: dict[str, Any]) -> None:
        unrelated_start_calls.append(queue_config.copy())
        unrelated_proxy._process = object()  # type: ignore[assignment]
        unrelated_proxy._running = True

    selected_proxy.start = _selected_start  # type: ignore[method-assign]
    unrelated_proxy.start = _unrelated_start  # type: ignore[method-assign]

    class _Registry:
        def __init__(self) -> None:
            self.handlers = {"model-a": selected_proxy, "model-b": unrelated_proxy}

        def get_handler(self, model_id: str) -> Any:
            return self.handlers[model_id]

    fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
    req_id = str(fixed_uuid)
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: fixed_uuid)

    raw_request = types.SimpleNamespace(
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(registry=_Registry(), handler=selected_proxy)
        ),
        state=types.SimpleNamespace(request_id="req-test"),
    )
    request = ChatCompletionRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
    )

    response_task = asyncio.create_task(endpoints_module.chat_completions(request, raw_request))
    await _wait_for_pending_ids(selected_proxy, {req_id})
    await selected_proxy._pending[req_id].put(
        {
            "type": "result",
            "value": {"response": "hello", "usage": None},
        }
    )

    response = await response_task

    assert selected_start_calls == [selected_proxy._lazy_queue_config]  # type: ignore[attr-defined]
    assert unrelated_start_calls == []
    assert selected_proxy._request_queue.items[0]["method"] == "generate_text_response"
    assert unrelated_proxy._request_queue.items == []
    assert json.loads(response.body)["model"] == "model-a"


@pytest.mark.asyncio
async def test_call_lazily_starts_unloaded_proxy_before_enqueuing_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-stream RPCs should cold-start an unloaded proxy exactly once."""

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
    event_log: list[str] = []
    proxy._request_queue = _CaptureQueue(event_log=event_log)  # type: ignore[assignment]
    proxy._rpc_timeout = 1.0
    proxy._lazy_queue_config = {"queue_size": 5, "timeout": 19}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []
    allow_start_to_finish = asyncio.Event()

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Block completion so the test can assert enqueue ordering."""

        event_log.append("start")
        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True
        await allow_start_to_finish.wait()

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_id = "req-lazy-call"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    call_task = asyncio.create_task(proxy._call("generate_text_response", "hello"))
    await asyncio.sleep(0)

    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]
    assert event_log == ["start"]
    assert proxy._request_queue.items == []

    allow_start_to_finish.set()
    await _wait_for_pending_ids(proxy, {req_id})
    await _wait_for_queue_size(proxy._request_queue, 1)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "result", "value": {"ok": True}})

    assert await call_task == {"ok": True}
    assert event_log[:2] == ["start", "enqueue"]
    assert proxy._request_queue.items[0]["method"] == "generate_text_response"


@pytest.mark.asyncio
async def test_call_lazily_starts_only_once_for_concurrent_first_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first hits should coalesce to a single deferred cold start."""

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
    proxy._rpc_timeout = 1.0
    proxy._lazy_queue_config = {"queue_size": 8, "timeout": 29}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []
    allow_start_to_finish = asyncio.Event()

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Record the cold-start config and block to expose duplicate starts."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True
        await allow_start_to_finish.wait()

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_ids = iter(["req-concurrent-a", "req-concurrent-b"])
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: next(req_ids))

    first_task = asyncio.create_task(proxy._call("generate_text_response", "hello-a"))
    second_task = asyncio.create_task(proxy._call("generate_text_response", "hello-b"))
    await asyncio.sleep(0)

    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]

    allow_start_to_finish.set()
    await _wait_for_pending_ids(proxy, {"req-concurrent-a", "req-concurrent-b"})

    await proxy._pending["req-concurrent-a"].put({"type": "result", "value": {"ok": "a"}})
    await proxy._pending["req-concurrent-b"].put({"type": "result", "value": {"ok": "b"}})

    assert await first_task == {"ok": "a"}
    assert await second_task == {"ok": "b"}
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_call_reuses_loaded_proxy_without_second_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat non-stream requests should not cold-start the same proxy twice."""

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
    proxy._rpc_timeout = 1.0
    proxy._lazy_queue_config = {"queue_size": 5, "timeout": 19}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Record each attempted cold start."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_ids = iter(["req-repeat-call-a", "req-repeat-call-b"])
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: next(req_ids))

    first_call_task = asyncio.create_task(proxy._call("generate_text_response", "hello-a"))
    await asyncio.sleep(0)
    await proxy._pending["req-repeat-call-a"].put({"type": "result", "value": {"ok": "a"}})

    assert await first_call_task == {"ok": "a"}
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]

    second_call_task = asyncio.create_task(proxy._call("generate_text_response", "hello-b"))
    await asyncio.sleep(0)
    await proxy._pending["req-repeat-call-b"].put({"type": "result", "value": {"ok": "b"}})

    assert await second_call_task == {"ok": "b"}
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_call_stream_lazily_starts_unloaded_proxy_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming RPCs should cold-start an unloaded proxy exactly once."""

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
    event_log: list[str] = []
    proxy._request_queue = _CaptureQueue(event_log=event_log)  # type: ignore[assignment]
    proxy._control_queue = _CaptureQueue()  # type: ignore[assignment]
    proxy._rpc_timeout = 1.0
    proxy._stream_queue_size = 4
    proxy._lazy_queue_config = {"queue_size": 6, "timeout": 23}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []
    allow_start_to_finish = asyncio.Event()

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Block completion so the test can assert enqueue ordering."""

        event_log.append("start")
        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True
        await allow_start_to_finish.wait()

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_id = "req-lazy-stream"
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: req_id)

    async def _collect_stream() -> list[str]:
        """Consume the async generator into a list for assertion."""

        return [chunk async for chunk in proxy._call_stream("generate_text_stream", "hello")]

    stream_task = asyncio.create_task(_collect_stream())
    await asyncio.sleep(0)

    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]
    assert event_log == ["start"]
    assert proxy._request_queue.items == []

    allow_start_to_finish.set()
    await _wait_for_pending_ids(proxy, {req_id})
    await _wait_for_queue_size(proxy._request_queue, 1)

    result_queue = proxy._pending[req_id]
    await result_queue.put({"type": "chunk", "value": "hello"})
    await result_queue.put({"type": handler_process_module._STREAM_END})

    assert await stream_task == ["hello"]
    assert event_log[:2] == ["start", "enqueue"]
    assert proxy._request_queue.items[0]["method"] == "generate_text_stream"


@pytest.mark.asyncio
async def test_call_stream_reuses_loaded_proxy_without_second_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat stream requests should not cold-start the same proxy twice."""

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
    proxy._lazy_queue_config = {"queue_size": 6, "timeout": 23}  # type: ignore[attr-defined]

    start_calls: list[dict[str, Any]] = []

    async def _fake_start(queue_config: dict[str, Any]) -> None:
        """Record each attempted cold start."""

        start_calls.append(queue_config.copy())
        proxy._process = object()  # type: ignore[assignment]
        proxy._running = True

    proxy.start = _fake_start  # type: ignore[method-assign]

    req_ids = iter(["req-repeat-stream-a", "req-repeat-stream-b"])
    monkeypatch.setattr(handler_process_module.uuid, "uuid4", lambda: next(req_ids))

    async def _collect_stream(prompt: str) -> list[str]:
        """Consume the async stream into a list for assertion."""

        return [chunk async for chunk in proxy._call_stream("generate_text_stream", prompt)]

    first_stream_task = asyncio.create_task(_collect_stream("hello-a"))
    await asyncio.sleep(0)
    await proxy._pending["req-repeat-stream-a"].put({"type": "chunk", "value": "hello-a"})
    await proxy._pending["req-repeat-stream-a"].put({"type": handler_process_module._STREAM_END})

    assert await first_stream_task == ["hello-a"]
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]

    second_stream_task = asyncio.create_task(_collect_stream("hello-b"))
    await asyncio.sleep(0)
    await proxy._pending["req-repeat-stream-b"].put({"type": "chunk", "value": "hello-b"})
    await proxy._pending["req-repeat-stream-b"].put({"type": handler_process_module._STREAM_END})

    assert await second_stream_task == ["hello-b"]
    assert start_calls == [proxy._lazy_queue_config]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_start_registers_ready_queue_before_reader_can_consume_immediate_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate ready signals should not be dropped before ``__ready__`` exists."""

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
    proxy._response_queue = _ReadyThenStopQueue(  # type: ignore[assignment]
        on_first_get=lambda: setattr(proxy, "_running", False)
    )

    class _InlineThread:
        """Thread stub that runs the target synchronously on ``start()``."""

        def __init__(
            self,
            *,
            target: Any,
            daemon: bool,
            name: str,
        ) -> None:
            del daemon, name
            self._target = target

        def start(self) -> None:
            self._target()

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            del timeout

    class _FakeProcess:
        """Process stub satisfying the startup/cleanup surface."""

        def __init__(
            self,
            *,
            target: Any,
            args: tuple[Any, ...],
            name: str,
        ) -> None:
            del target, args, name
            self.pid = 12345

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(handler_process_module.threading, "Thread", _InlineThread)
    proxy._ctx = types.SimpleNamespace(Process=_FakeProcess)  # type: ignore[assignment]

    original_wait_for = handler_process_module.asyncio.wait_for

    async def _short_wait(awaitable: Any, timeout: float) -> Any:
        if timeout == 300:
            timeout = 0.01
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(handler_process_module.asyncio, "wait_for", _short_wait)

    await proxy.start({"queue_size": 5, "timeout": 19})

    assert proxy._started is True
    assert proxy.model_created > 0


@pytest.mark.asyncio
async def test_restart_of_dead_proxy_retires_stale_reader_before_spawning_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart should retire any old live reader thread before starting a new one."""

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

    class _DeadProcess:
        """Process stub representing an unexpectedly exited child."""

        def is_alive(self) -> bool:
            return False

    class _StaleReaderThread:
        """Reader-thread stub that records whether restart joined it."""

        def __init__(self) -> None:
            self.join_calls: list[float | None] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

    started_threads: list[_ReplacementThread] = []

    class _ReplacementThread:
        """Thread stub that records replacement starts without running."""

        def __init__(
            self,
            *,
            target: Any,
            daemon: bool,
            name: str,
        ) -> None:
            del target, daemon, name
            self.started = False
            started_threads.append(self)

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self.started = False

    class _FakeProcess:
        """Replacement child-process stub for the restart path."""

        def __init__(
            self,
            *,
            target: Any,
            args: tuple[Any, ...],
            name: str,
        ) -> None:
            del target, args, name
            self.pid = 67890

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    stale_reader = _StaleReaderThread()
    proxy._process = _DeadProcess()  # type: ignore[assignment]
    proxy._reader_thread = stale_reader  # type: ignore[assignment]
    proxy._started = True
    proxy._running = True

    monkeypatch.setattr(handler_process_module.threading, "Thread", _ReplacementThread)
    proxy._ctx = types.SimpleNamespace(Process=_FakeProcess)  # type: ignore[assignment]

    async def _immediate_ready(awaitable: Any, timeout: float) -> Any:
        del timeout
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        return {"type": "ready", "success": True}

    monkeypatch.setattr(handler_process_module.asyncio, "wait_for", _immediate_ready)

    await proxy.start({"queue_size": 5, "timeout": 19})

    assert stale_reader.join_calls == [2]
    assert len(started_threads) == 1
    assert started_threads[0].started is True
