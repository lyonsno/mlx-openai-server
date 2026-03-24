"""Fail-first tests for generation-config-seeded multi-model defaults."""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import queue
import signal
import sys
import types
from typing import Any

from fastapi.responses import JSONResponse
import pytest

from app import config as config_module
from app.config import load_config_from_yaml
from app.core import handler_process as handler_process_module
from app.core.handler_process import HandlerProcessProxy
from app.schemas.openai import ChatCompletionRequest, Message, ResponsesRequest

GENERATION_CONFIG_DEFAULTS: dict[str, int | float] = {
    "temperature": 0.73,
    "top_p": 0.91,
    "top_k": 27,
    "min_p": 0.17,
    "repetition_penalty": 1.19,
    "max_new_tokens": 4096,
}

GLOBAL_ENV_DEFAULTS: dict[str, str] = {
    "DEFAULT_TEMPERATURE": "0.2",
    "DEFAULT_TOP_P": "0.5",
    "DEFAULT_TOP_K": "99",
    "DEFAULT_MIN_P": "0.9",
    "DEFAULT_REPETITION_PENALTY": "1.5",
    "DEFAULT_MAX_TOKENS": "42",
}

GENERATION_CONFIG_STRING_DEFAULTS: dict[str, str] = {
    "temperature": "0.73",
    "top_p": "0.91",
    "top_k": "27",
    "min_p": "0.17",
    "repetition_penalty": "1.19",
    "max_new_tokens": "4096",
}

LARGE_INTEGER_STRING = "9007199254740993"


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
    """Import ``app.server`` with lightweight handler/runtime stubs."""

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

    class _FakeLMHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

    fake_lm = types.ModuleType("app.handler.mlx_lm")
    fake_lm.MLXLMHandler = _FakeLMHandler

    class _FakeVLMHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

    fake_vlm = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm.MLXVLMHandler = _FakeVLMHandler

    class _FakeEmbeddingsHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

    fake_embeddings = types.ModuleType("app.handler.mlx_embeddings")
    fake_embeddings.MLXEmbeddingsHandler = _FakeEmbeddingsHandler

    class _FakeWhisperHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs

    fake_whisper = types.ModuleType("app.handler.mlx_whisper")
    fake_whisper.MLXWhisperHandler = _FakeWhisperHandler

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


class _FakeRegistry:
    """Minimal registry stub that resolves handlers by model id."""

    def __init__(self, handlers: dict[str, Any]) -> None:
        self._handlers = handlers

    def get_handler(self, model_id: str) -> Any:
        return self._handlers[model_id]


def _make_raw_request(registry: _FakeRegistry, handler: Any | None = None) -> Any:
    """Build a minimal request-like object for endpoint unit tests."""

    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=registry, handler=handler)),
        state=types.SimpleNamespace(request_id="req-test"),
    )


def _write_model_dir(model_dir: Path, generation_config: dict[str, int | float]) -> None:
    """Create a model directory with a ``generation_config.json`` file."""

    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )


def test_load_config_from_yaml_seeds_missing_defaults_from_generation_config(
    tmp_path: Path,
) -> None:
    """Missing per-model defaults should seed from ``generation_config.json``."""

    model_dir = tmp_path / "model-a"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-a\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))
    model_config = config.models[0]

    assert model_config.default_temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert model_config.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert model_config.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert model_config.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert model_config.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert model_config.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_load_config_from_yaml_prefers_explicit_defaults_over_generation_config(
    tmp_path: Path,
) -> None:
    """Explicit per-model YAML defaults should override ``generation_config.json``."""

    model_dir = tmp_path / "model-a"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_dir}\n"
        "    model_type: lm\n"
        "    model_id: model-a\n"
        "    default_temperature: 0.42\n"
        "    default_top_k: 7\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))
    model_config = config.models[0]

    assert model_config.default_temperature == pytest.approx(0.42)
    assert model_config.default_top_k == 7
    assert model_config.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert model_config.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert model_config.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert model_config.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_load_config_from_yaml_ignores_internal_generation_config_bookkeeping_fields(
    tmp_path: Path,
) -> None:
    """Runtime-only seeding flags from YAML should not suppress default seeding."""

    model_dir = tmp_path / "model-bookkeeping-flags"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_dir}\n"
        "    model_type: lm\n"
        "    model_id: model-bookkeeping-flags\n"
        "    generation_config_seed_attempted: true\n"
        "    generation_config_lookup_warning_emitted: true\n",
        encoding="utf-8",
    )

    model_config = load_config_from_yaml(str(config_path)).models[0]

    assert model_config.default_temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert model_config.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert model_config.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert model_config.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert model_config.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert model_config.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]
    assert model_config.generation_config_lookup_warning_emitted is False


def test_load_config_from_yaml_coerces_numeric_generation_config_strings(
    tmp_path: Path,
) -> None:
    """Numeric strings from generation config should become typed runtime defaults."""

    model_dir = tmp_path / "model-typed"
    _write_model_dir(model_dir, GENERATION_CONFIG_STRING_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-typed\n",
        encoding="utf-8",
    )

    model_config = load_config_from_yaml(str(config_path)).models[0]

    assert model_config.default_temperature == pytest.approx(0.73)
    assert isinstance(model_config.default_temperature, float)
    assert model_config.default_top_p == pytest.approx(0.91)
    assert isinstance(model_config.default_top_p, float)
    assert model_config.default_top_k == 27
    assert isinstance(model_config.default_top_k, int)
    assert model_config.default_min_p == pytest.approx(0.17)
    assert isinstance(model_config.default_min_p, float)
    assert model_config.default_repetition_penalty == pytest.approx(1.19)
    assert isinstance(model_config.default_repetition_penalty, float)
    assert model_config.default_max_tokens == 4096
    assert isinstance(model_config.default_max_tokens, int)


def test_resolve_generation_config_model_dir_uses_local_files_only_for_repo_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repo-id generation-config resolution should stay local-cache-only."""

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    captured_kwargs: dict[str, Any] = {}

    def _fake_snapshot_download(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    resolved = config_module.resolve_generation_config_model_dir("mlx-community/model-a-4bit")

    assert resolved == snapshot_dir
    assert captured_kwargs["allow_patterns"] == "generation_config.json"
    assert captured_kwargs["local_files_only"] is True


def test_importing_app_config_does_not_eagerly_import_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing ``app.config`` should not require ``huggingface_hub`` eagerly."""

    app_package = importlib.import_module("app")
    original_modules = {
        name: sys.modules.get(name)
        for name in ("app.config", "huggingface_hub", "huggingface_hub.constants")
    }
    for name in original_modules:
        sys.modules.pop(name, None)

    attempted_imports: list[str] = []
    original_import = builtins.__import__

    def _guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            attempted_imports.append(name)
            msg = "app.config should not import huggingface_hub until repo resolution is needed"
            raise AssertionError(msg)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    fresh_config_module: types.ModuleType | None = None
    try:
        fresh_config_module = importlib.import_module("app.config")
    finally:
        sys.modules.pop("app.config", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if original_modules["app.config"] is None:
            app_package.__dict__.pop("config", None)
        else:
            setattr(app_package, "config", original_modules["app.config"])

    assert fresh_config_module is not None
    assert attempted_imports == []


def test_load_config_from_yaml_rejects_non_integral_generation_config_values(
    tmp_path: Path,
) -> None:
    """Lossy integer coercions from generation config should be ignored with warnings."""

    model_dir = tmp_path / "model-invalid-integers"
    _write_model_dir(
        model_dir,
        {
            "temperature": 0.73,
            "top_k": 1.7,
            "max_new_tokens": True,
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-invalid\n",
        encoding="utf-8",
    )

    warning_messages: list[str] = []
    sink_id = config_module.logger.add(
        lambda message: warning_messages.append(str(message).rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        config = load_config_from_yaml(str(config_path))
    finally:
        config_module.logger.remove(sink_id)

    model_config = config.models[0]

    assert model_config.default_temperature == pytest.approx(0.73)
    assert model_config.default_top_k is None
    assert model_config.default_max_tokens is None
    assert any("top_k=1.7" in message for message in warning_messages)
    assert any("max_new_tokens=True" in message for message in warning_messages)


def test_attempt_generation_config_seeding_retries_missing_defaults_after_semantic_fix(
    tmp_path: Path,
) -> None:
    """Later startup phases should fill fields that were blocked by bad values."""

    model_dir = tmp_path / "model-semantic-retry"
    model_dir.mkdir()
    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text(
        json.dumps(
            {
                "temperature": "bad",
                "top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
            }
        ),
        encoding="utf-8",
    )

    model_cfg = config_module.ModelEntryConfig(
        model_path=str(model_dir),
        model_type="lm",
        model_id="model-semantic-retry",
    )

    config_module.attempt_generation_config_seeding(model_cfg)

    assert model_cfg.default_temperature is None
    assert model_cfg.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert config_module.should_attempt_generation_config_seeding(model_cfg) is True

    generation_config_path.write_text(
        json.dumps(
            {
                "temperature": GENERATION_CONFIG_DEFAULTS["temperature"],
                "top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
            }
        ),
        encoding="utf-8",
    )

    config_module.attempt_generation_config_seeding(model_cfg)

    assert model_cfg.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert model_cfg.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])


def test_load_config_from_yaml_preserves_large_integer_like_generation_config_strings(
    tmp_path: Path,
) -> None:
    """Large integer-like strings should preserve exact values without float rounding."""

    model_dir = tmp_path / "model-large-integers"
    _write_model_dir(
        model_dir,
        {
            "temperature": 0.73,
            "top_k": LARGE_INTEGER_STRING,
            "max_new_tokens": LARGE_INTEGER_STRING,
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-large\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))

    model_config = config.models[0]

    assert model_config.default_temperature == pytest.approx(0.73)
    assert model_config.default_top_k == int(LARGE_INTEGER_STRING)
    assert model_config.default_max_tokens == int(LARGE_INTEGER_STRING)


def test_create_handler_from_config_preserves_seeded_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded defaults should survive the server-side handler attach layer."""

    model_dir = tmp_path / "model-handler"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-handler\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    server_module = _load_server_module(monkeypatch)
    handler = server_module.create_handler_from_config(model_config)

    assert handler._uses_model_sampling_defaults is True
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert handler.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert handler.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert handler.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_create_handler_from_config_preserves_seeded_defaults_for_multimodal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded defaults should also attach to multimodal handlers."""

    model_dir = tmp_path / "model-handler-mm"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_dir}\n"
        "    model_type: multimodal\n"
        "    model_id: model-handler-mm\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    server_module = _load_server_module(monkeypatch)
    handler = server_module.create_handler_from_config(model_config)

    assert handler._uses_model_sampling_defaults is True
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert handler.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert handler.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert handler.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_handler_process_proxy_preserves_seeded_defaults(tmp_path: Path) -> None:
    """Seeded defaults should survive the serialized handler-proxy path."""

    model_dir = tmp_path / "model-proxy"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-proxy\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert proxy._uses_model_sampling_defaults is True
    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert proxy.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert proxy.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert proxy.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_split_model_cfg_for_worker_keeps_runtime_bookkeeping_separate() -> None:
    """Worker handoff should keep runtime seeding state out of serialized config."""

    model_cfg = config_module.ModelEntryConfig(
        model_path="mlx-community/model-split-4bit",
        model_type="lm",
        model_id="model-split",
        default_temperature=GENERATION_CONFIG_DEFAULTS["temperature"],
        generation_config_seed_attempted=True,
        generation_config_lookup_warning_emitted=True,
    )

    model_cfg_dict, startup_state = handler_process_module._split_model_cfg_for_worker(model_cfg)

    assert model_cfg_dict["model_path"] == "mlx-community/model-split-4bit"
    assert model_cfg_dict["default_temperature"] == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    for field_name in config_module.RUNTIME_ONLY_MODEL_ENTRY_FIELDS:
        assert field_name not in model_cfg_dict
    assert startup_state == {
        "generation_config_lookup_warning_emitted": True,
        "generation_config_seed_attempted": True,
    }


def test_handler_worker_preserves_partial_generation_config_attempt_state_after_proxy_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child startup should not re-attempt partial repo seeding after proxy handoff."""

    model_cfg = config_module.ModelEntryConfig(
        model_path="mlx-community/model-partial-worker-4bit",
        model_type="lm",
        model_id="model-partial-worker",
        default_temperature=GENERATION_CONFIG_DEFAULTS["temperature"],
        default_top_p=GENERATION_CONFIG_DEFAULTS["top_p"],
        generation_config_seed_attempted=True,
    )

    captured_model_cfgs: list[config_module.ModelEntryConfig] = []
    initialized_queue_configs: list[dict[str, Any]] = []
    cleanup_calls: list[str] = []

    class _FakeHandler:
        async def initialize(self, queue_config: dict[str, Any]) -> None:
            initialized_queue_configs.append(dict(queue_config))

        async def cleanup(self) -> None:
            cleanup_calls.append("cleanup")

    def _fake_create_handler_from_config(
        worker_model_cfg: config_module.ModelEntryConfig,
    ) -> _FakeHandler:
        captured_model_cfgs.append(worker_model_cfg)
        assert worker_model_cfg.default_temperature == pytest.approx(
            GENERATION_CONFIG_DEFAULTS["temperature"]
        )
        assert worker_model_cfg.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
        assert worker_model_cfg.default_top_k is None
        assert worker_model_cfg.generation_config_seed_attempted is True
        assert config_module.should_attempt_generation_config_seeding(worker_model_cfg) is False
        return _FakeHandler()

    fake_server = types.ModuleType("app.server")
    fake_server.create_handler_from_config = _fake_create_handler_from_config

    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx_core.clear_cache = lambda: None
    fake_mlx.core = fake_mlx_core  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "app.server", fake_server)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    request_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    control_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    request_queue.put({"id": "shutdown-test", "method": handler_process_module._SHUTDOWN})
    model_cfg_dict, startup_state = handler_process_module._split_model_cfg_for_worker(model_cfg)

    handler_process_module._handler_worker(
        model_cfg_dict,
        startup_state,
        {"queue_size": 5, "timeout": 19},
        request_queue,  # type: ignore[arg-type]
        response_queue,  # type: ignore[arg-type]
        control_queue,  # type: ignore[arg-type]
    )

    assert len(captured_model_cfgs) == 1
    assert initialized_queue_configs == [{"queue_size": 5, "timeout": 19}]
    assert cleanup_calls == ["cleanup"]
    assert response_queue.get_nowait() == {"type": "ready", "success": True}
    assert response_queue.get_nowait() == {
        "id": "shutdown-test",
        "type": "shutdown_complete",
    }


def test_handler_worker_uses_explicit_proxy_served_model_name_when_cfg_dict_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit proxy aliases should survive the worker handoff."""

    proxy = HandlerProcessProxy(
        model_cfg_dict={"model_path": "dummy-model", "model_type": "whisper"},
        model_type="whisper",
        model_path="dummy-model",
        served_model_name="served-alias",
    )

    captured_model_cfgs: list[config_module.ModelEntryConfig] = []

    class _FakeHandler:
        async def initialize(self, _queue_config: dict[str, Any]) -> None:
            return None

        async def cleanup(self) -> None:
            return None

    def _fake_create_handler_from_config(
        worker_model_cfg: config_module.ModelEntryConfig,
    ) -> _FakeHandler:
        captured_model_cfgs.append(worker_model_cfg)
        return _FakeHandler()

    fake_server = types.ModuleType("app.server")
    fake_server.create_handler_from_config = _fake_create_handler_from_config

    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx_core.clear_cache = lambda: None
    fake_mlx.core = fake_mlx_core  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "app.server", fake_server)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    request_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    control_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    request_queue.put({"id": "shutdown-test", "method": handler_process_module._SHUTDOWN})

    handler_process_module._handler_worker(
        dict(proxy._model_cfg_dict),
        dict(proxy._startup_state),
        {"queue_size": 5, "timeout": 19},
        request_queue,  # type: ignore[arg-type]
        response_queue,  # type: ignore[arg-type]
        control_queue,  # type: ignore[arg-type]
    )

    assert len(captured_model_cfgs) == 1
    assert captured_model_cfgs[0].served_model_name == "served-alias"
    assert captured_model_cfgs[0].model_id == "served-alias"


def test_handler_process_proxy_coerces_numeric_generation_config_strings_at_runtime_boundary(
    tmp_path: Path,
) -> None:
    """Proxy defaults should be typed numerics even when generation config used strings."""

    model_dir = tmp_path / "model-proxy-typed"
    _write_model_dir(model_dir, GENERATION_CONFIG_STRING_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_dir}\n"
        "    model_type: lm\n"
        "    model_id: model-proxy-typed\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert proxy.default_temperature == pytest.approx(0.73)
    assert isinstance(proxy.default_temperature, float)
    assert proxy.default_top_p == pytest.approx(0.91)
    assert isinstance(proxy.default_top_p, float)
    assert proxy.default_top_k == 27
    assert isinstance(proxy.default_top_k, int)
    assert proxy.default_min_p == pytest.approx(0.17)
    assert isinstance(proxy.default_min_p, float)
    assert proxy.default_repetition_penalty == pytest.approx(1.19)
    assert isinstance(proxy.default_repetition_penalty, float)
    assert proxy.default_max_tokens == 4096
    assert isinstance(proxy.default_max_tokens, int)


def test_handler_process_proxy_seeds_repo_id_models_from_resolved_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unattempted repo-id configs should seed defaults on the proxy path."""

    snapshot_dir = tmp_path / "hf-snapshot-proxy"
    _write_model_dir(snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    model_config = config_module.ModelEntryConfig(
        model_path="mlx-community/model-proxy-4bit",
        model_type="lm",
        model_id="model-proxy",
    )

    monkeypatch.setattr(
        config_module,
        "_resolve_generation_config_model_dir",
        lambda _model_path: snapshot_dir,
        raising=False,
    )

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert proxy._uses_model_sampling_defaults is True
    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert proxy.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert proxy.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert proxy.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_create_handler_from_config_seeds_repo_id_models_from_resolved_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unattempted repo-id configs should seed defaults through handler creation."""

    snapshot_dir = tmp_path / "hf-snapshot"
    _write_model_dir(snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    model_config = config_module.ModelEntryConfig(
        model_path="mlx-community/model-a-4bit",
        model_type="lm",
        model_id="model-a",
    )

    server_module = _load_server_module(monkeypatch)
    monkeypatch.setattr(
        server_module,
        "_resolve_generation_config_model_dir",
        lambda _model_path: snapshot_dir,
        raising=False,
    )

    handler = server_module.create_handler_from_config(model_config)

    assert handler._uses_model_sampling_defaults is True
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert handler.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert handler.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )


def test_load_config_from_yaml_seeds_repo_id_models_from_resolved_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repo-id model paths should seed defaults during YAML load too."""

    snapshot_dir = tmp_path / "hf-snapshot-load"
    _write_model_dir(snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  - model_path: mlx-community/model-load-4bit\n"
        "    model_type: lm\n"
        "    model_id: model-load\n",
        encoding="utf-8",
    )

    resolver_calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        resolver_calls.append(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    model_config = load_config_from_yaml(str(config_path)).models[0]

    assert resolver_calls == [
        {
            "repo_id": "mlx-community/model-load-4bit",
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert model_config.default_temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert model_config.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert model_config.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert model_config.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert model_config.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert model_config.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_handler_process_proxy_does_not_reresolve_partially_seeded_repo_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy construction should not repeat repo resolution after YAML-load seeding."""

    snapshot_dir = tmp_path / "hf-snapshot-partial"
    _write_model_dir(
        snapshot_dir,
        {
            "temperature": GENERATION_CONFIG_DEFAULTS["temperature"],
            "top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  - model_path: mlx-community/model-partial-4bit\n"
        "    model_type: lm\n"
        "    model_id: model-partial\n",
        encoding="utf-8",
    )

    resolver_calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        resolver_calls.append(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    model_config = load_config_from_yaml(str(config_path)).models[0]
    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert resolver_calls == [
        {
            "repo_id": "mlx-community/model-partial-4bit",
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert model_config.default_temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert model_config.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert model_config.default_top_k is None
    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k is None


def test_create_handler_from_config_does_not_reresolve_partially_seeded_repo_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child handler creation should not repeat repo resolution after YAML-load seeding."""

    snapshot_dir = tmp_path / "hf-snapshot-partial-child"
    _write_model_dir(
        snapshot_dir,
        {
            "temperature": GENERATION_CONFIG_DEFAULTS["temperature"],
            "top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  - model_path: mlx-community/model-partial-child-4bit\n"
        "    model_type: lm\n"
        "    model_id: model-partial-child\n",
        encoding="utf-8",
    )

    loader_resolver_calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        loader_resolver_calls.append(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    model_config = load_config_from_yaml(str(config_path)).models[0]
    server_module = _load_server_module(monkeypatch)
    handler_resolver_calls: list[str] = []

    def _unexpected_reresolve(model_path: str) -> Path:
        handler_resolver_calls.append(model_path)
        return snapshot_dir

    monkeypatch.setattr(
        server_module,
        "_resolve_generation_config_model_dir",
        _unexpected_reresolve,
        raising=False,
    )

    handler = server_module.create_handler_from_config(model_config)

    assert loader_resolver_calls == [
        {
            "repo_id": "mlx-community/model-partial-child-4bit",
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert handler_resolver_calls == []
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k is None


def test_repo_snapshot_without_generation_config_can_seed_later_on_proxy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior absent file should not block later proxy-side pickup."""

    initial_snapshot_dir = tmp_path / "hf-snapshot-no-generation-config"
    initial_snapshot_dir.mkdir()
    later_snapshot_dir = tmp_path / "hf-snapshot-no-generation-config-later"
    _write_model_dir(later_snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    model_path = "mlx-community/model-no-generation-config-4bit"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_path}\n"
        "    model_type: lm\n"
        "    model_id: model-no-generation-config\n",
        encoding="utf-8",
    )

    loader_resolver_calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        loader_resolver_calls.append(kwargs)
        return str(initial_snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    model_config = load_config_from_yaml(str(config_path)).models[0]
    assert model_config.default_temperature is None

    proxy_resolver_calls: list[str] = []

    def _proxy_retry_resolver(resolved_model_path: str) -> Path:
        proxy_resolver_calls.append(resolved_model_path)
        return later_snapshot_dir

    monkeypatch.setattr(
        config_module,
        "_resolve_generation_config_model_dir",
        _proxy_retry_resolver,
        raising=False,
    )

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert loader_resolver_calls == [
        {
            "repo_id": model_path,
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert proxy_resolver_calls == [model_path]
    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]


def test_create_handler_from_config_can_seed_later_after_snapshot_without_generation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior absent file should not block later child-side pickup."""

    initial_snapshot_dir = tmp_path / "hf-snapshot-no-generation-config-child"
    initial_snapshot_dir.mkdir()
    later_snapshot_dir = tmp_path / "hf-snapshot-no-generation-config-child-later"
    _write_model_dir(later_snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    model_path = "mlx-community/model-no-generation-config-child-4bit"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_path}\n"
        "    model_type: lm\n"
        "    model_id: model-no-generation-config-child\n",
        encoding="utf-8",
    )

    loader_resolver_calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        loader_resolver_calls.append(kwargs)
        return str(initial_snapshot_dir)

    monkeypatch.setattr(config_module, "snapshot_download", _fake_snapshot_download)

    model_config = load_config_from_yaml(str(config_path)).models[0]
    assert model_config.default_temperature is None
    server_module = _load_server_module(monkeypatch)
    child_resolver_calls: list[str] = []

    def _child_retry_resolver(resolved_model_path: str) -> Path:
        child_resolver_calls.append(resolved_model_path)
        return later_snapshot_dir

    monkeypatch.setattr(
        server_module,
        "_resolve_generation_config_model_dir",
        _child_retry_resolver,
        raising=False,
    )

    handler = server_module.create_handler_from_config(model_config)

    assert loader_resolver_calls == [
        {
            "repo_id": model_path,
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert child_resolver_calls == [model_path]
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]


def test_malformed_generation_config_can_seed_later_on_proxy_path(
    tmp_path: Path,
) -> None:
    """An early parse failure should not block later proxy-side pickup."""

    model_dir = tmp_path / "model-malformed-proxy"
    model_dir.mkdir()
    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text("{not-json", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-malformed-proxy\n",
        encoding="utf-8",
    )

    model_config = load_config_from_yaml(str(config_path)).models[0]
    assert model_config.default_temperature is None

    generation_config_path.write_text(
        json.dumps(GENERATION_CONFIG_DEFAULTS),
        encoding="utf-8",
    )

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_config.__dict__.copy(),
        model_type=model_config.model_type,
        model_path=model_config.model_path,
        model_id=model_config.model_id,
    )

    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]


def test_create_handler_from_config_can_seed_later_after_malformed_generation_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An early parse failure should not block later child-side pickup."""

    model_dir = tmp_path / "model-malformed-child"
    model_dir.mkdir()
    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text("{not-json", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-malformed-child\n",
        encoding="utf-8",
    )

    model_config = load_config_from_yaml(str(config_path)).models[0]
    assert model_config.default_temperature is None

    generation_config_path.write_text(
        json.dumps(GENERATION_CONFIG_DEFAULTS),
        encoding="utf-8",
    )

    server_module = _load_server_module(monkeypatch)
    handler = server_module.create_handler_from_config(model_config)

    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]


def test_cold_cache_repo_id_can_retry_later_startup_phases_but_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-cache misses may retry later, but startup should not spam warnings."""

    model_path = "mlx-community/model-cold-cache-4bit"
    later_snapshot_dir = tmp_path / "hf-snapshot-cold-cache-later"
    _write_model_dir(later_snapshot_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_path}\n"
        "    model_type: lm\n"
        "    model_id: model-cold-cache\n",
        encoding="utf-8",
    )

    resolver_calls: list[dict[str, Any]] = []

    def _cold_cache_snapshot_download(**kwargs: Any) -> str:
        resolver_calls.append(kwargs)
        msg = (
            "Cannot find an appropriate cached snapshot folder for the specified "
            "revision on the local disk and outgoing traffic has been disabled."
        )
        raise FileNotFoundError(msg)

    monkeypatch.setattr(config_module, "snapshot_download", _cold_cache_snapshot_download)

    warning_messages: list[str] = []
    sink_id = config_module.logger.add(
        lambda message: warning_messages.append(str(message).rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        model_config = load_config_from_yaml(str(config_path)).models[0]
        assert model_config.default_temperature is None

        proxy_resolver_calls: list[str] = []

        def _proxy_retry_resolver(resolved_model_path: str) -> Path:
            proxy_resolver_calls.append(resolved_model_path)
            return later_snapshot_dir

        monkeypatch.setattr(
            config_module,
            "_resolve_generation_config_model_dir",
            _proxy_retry_resolver,
            raising=False,
        )

        proxy = HandlerProcessProxy(
            model_cfg_dict=model_config.__dict__.copy(),
            model_type=model_config.model_type,
            model_path=model_config.model_path,
            model_id=model_config.model_id,
        )
        server_module = _load_server_module(monkeypatch)
        child_resolver_calls: list[str] = []

        def _child_retry_resolver(resolved_model_path: str) -> Path:
            child_resolver_calls.append(resolved_model_path)
            return later_snapshot_dir

        monkeypatch.setattr(
            server_module,
            "_resolve_generation_config_model_dir",
            _child_retry_resolver,
            raising=False,
        )
        handler = server_module.create_handler_from_config(model_config)
    finally:
        config_module.logger.remove(sink_id)

    assert resolver_calls == [
        {
            "repo_id": model_path,
            "allow_patterns": "generation_config.json",
            "local_files_only": True,
        }
    ]
    assert proxy_resolver_calls == [model_path]
    assert child_resolver_calls == [model_path]
    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    model_warning_messages = [message for message in warning_messages if model_path in message]
    assert len(model_warning_messages) == 1


def test_generation_config_cache_miss_warning_scopes_retry_to_preinitialize_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-cache warnings should not imply post-initialize pickup this startup."""

    model_cfg = config_module.ModelEntryConfig(
        model_path="mlx-community/model-warning-scope-4bit",
        model_type="lm",
        model_id="model-warning-scope",
    )

    def _cold_cache_snapshot_download(**_kwargs: Any) -> str:
        msg = (
            "Cannot find an appropriate cached snapshot folder for the specified "
            "revision on the local disk and outgoing traffic has been disabled."
        )
        raise FileNotFoundError(msg)

    monkeypatch.setattr(config_module, "snapshot_download", _cold_cache_snapshot_download)

    warning_messages: list[str] = []
    sink_id = config_module.logger.add(
        lambda message: warning_messages.append(str(message).rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        config_module.attempt_generation_config_seeding(model_cfg)
    finally:
        config_module.logger.remove(sink_id)

    model_warning_messages = [
        message for message in warning_messages if model_cfg.model_path in message
    ]
    assert len(model_warning_messages) == 1
    assert "before handler initialization" in model_warning_messages[0]
    assert "later in startup if a source becomes available" not in model_warning_messages[0]


def test_handler_process_proxy_skips_repo_resolution_when_defaults_already_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy construction should not resolve repo snapshots once defaults exist."""

    model_cfg_dict = {
        "model_path": "mlx-community/model-proxy-skip-4bit",
        "model_type": "lm",
        "model_id": "model-proxy-skip",
        "default_temperature": GENERATION_CONFIG_DEFAULTS["temperature"],
        "default_top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
        "default_top_k": GENERATION_CONFIG_DEFAULTS["top_k"],
        "default_min_p": GENERATION_CONFIG_DEFAULTS["min_p"],
        "default_repetition_penalty": GENERATION_CONFIG_DEFAULTS["repetition_penalty"],
        "default_max_tokens": GENERATION_CONFIG_DEFAULTS["max_new_tokens"],
    }

    def _unexpected_resolver(_model_path: str) -> Path:
        msg = "repo resolution should be skipped once proxy defaults are already seeded"
        raise AssertionError(msg)

    monkeypatch.setattr(
        config_module,
        "_resolve_generation_config_model_dir",
        _unexpected_resolver,
        raising=False,
    )

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path=model_cfg_dict["model_path"],
        model_id=model_cfg_dict["model_id"],
    )

    assert proxy.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert proxy.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert proxy.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert proxy.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert proxy.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert proxy.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


def test_handler_process_proxy_preserves_debug_flag() -> None:
    """Proxy construction should preserve the model ``debug`` flag."""

    model_cfg_dict = {
        "model_path": "mlx-community/model-debug-4bit",
        "model_type": "lm",
        "model_id": "model-debug",
        "debug": True,
    }

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path=model_cfg_dict["model_path"],
        model_id=model_cfg_dict["model_id"],
    )

    assert getattr(proxy, "debug", None) is True


@pytest.mark.asyncio
async def test_chat_completions_logs_debug_requests_for_proxy_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proxy-backed debug requests should emit chat request logging."""

    endpoints_module = _load_endpoints_module()
    proxy = HandlerProcessProxy(
        model_cfg_dict={
            "model_path": "mlx-community/model-debug-4bit",
            "model_type": "lm",
            "model_id": "model-debug",
            "debug": True,
            "default_temperature": GENERATION_CONFIG_DEFAULTS["temperature"],
            "default_top_p": GENERATION_CONFIG_DEFAULTS["top_p"],
            "default_top_k": GENERATION_CONFIG_DEFAULTS["top_k"],
            "default_min_p": GENERATION_CONFIG_DEFAULTS["min_p"],
            "default_repetition_penalty": GENERATION_CONFIG_DEFAULTS["repetition_penalty"],
            "default_max_tokens": GENERATION_CONFIG_DEFAULTS["max_new_tokens"],
        },
        model_type="lm",
        model_path="mlx-community/model-debug-4bit",
        model_id="model-debug",
    )
    raw_request = _make_raw_request(_FakeRegistry({"model-debug": proxy}), handler=proxy)

    logged_requests: list[dict[str, Any]] = []

    def _fake_log_debug_server_request(**kwargs: Any) -> None:
        logged_requests.append(kwargs)

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        return JSONResponse(content={"model": request.model})

    monkeypatch.setattr(
        endpoints_module, "log_debug_server_request", _fake_log_debug_server_request
    )
    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)

    response = await endpoints_module.chat_completions(
        ChatCompletionRequest(
            model="model-debug",
            messages=[Message(role="user", content="hello from debug logging")],
        ),
        raw_request,
    )

    assert isinstance(response, JSONResponse)
    assert len(logged_requests) == 1
    assert logged_requests[0]["route"] == "/v1/chat/completions"
    assert logged_requests[0]["request_id"] == "req-test"
    assert logged_requests[0]["request_payload"]["model"] == "model-debug"
    assert logged_requests[0]["request_payload"]["messages"][0]["content"] == (
        "hello from debug logging"
    )


def test_create_handler_from_config_skips_repo_resolution_for_non_text_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-text handler creation should not resolve generation-config snapshots."""

    server_module = _load_server_module(monkeypatch)
    model_config = config_module.ModelEntryConfig(
        model_path="mlx-community/whisper-large-v3",
        model_type="whisper",
        model_id="whisper-large-v3",
    )

    def _unexpected_resolver(_model_path: str) -> Path:
        msg = "non-text handler creation should not resolve generation config"
        raise AssertionError(msg)

    monkeypatch.setattr(
        server_module,
        "_resolve_generation_config_model_dir",
        _unexpected_resolver,
        raising=False,
    )

    handler = server_module.create_handler_from_config(model_config)

    assert handler._uses_model_sampling_defaults is True
    assert handler.init_kwargs["model_path"] == "mlx-community/whisper-large-v3"


def test_handler_process_proxy_skips_repo_resolution_for_non_text_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-text proxy construction should not resolve generation-config snapshots."""

    model_cfg_dict = {
        "model_path": "mlx-community/whisper-large-v3",
        "model_type": "whisper",
        "model_id": "whisper-large-v3",
    }

    def _unexpected_resolver(_model_path: str) -> Path:
        msg = "non-text proxy construction should not resolve generation config"
        raise AssertionError(msg)

    monkeypatch.setattr(
        config_module,
        "_resolve_generation_config_model_dir",
        _unexpected_resolver,
        raising=False,
    )

    proxy = HandlerProcessProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="whisper",
        model_path=model_cfg_dict["model_path"],
        model_id=model_cfg_dict["model_id"],
    )

    assert proxy._uses_model_sampling_defaults is True
    assert proxy.model_path == "mlx-community/whisper-large-v3"


def test_create_handler_from_config_skips_repo_resolution_when_defaults_already_seeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child handler creation should not resolve repo snapshots once defaults exist."""

    model_config = config_module.ModelEntryConfig(
        model_path="mlx-community/model-skip-4bit",
        model_type="lm",
        model_id="model-skip",
        default_temperature=GENERATION_CONFIG_DEFAULTS["temperature"],
        default_top_p=GENERATION_CONFIG_DEFAULTS["top_p"],
        default_top_k=GENERATION_CONFIG_DEFAULTS["top_k"],
        default_min_p=GENERATION_CONFIG_DEFAULTS["min_p"],
        default_repetition_penalty=GENERATION_CONFIG_DEFAULTS["repetition_penalty"],
        default_max_tokens=GENERATION_CONFIG_DEFAULTS["max_new_tokens"],
    )

    server_module = _load_server_module(monkeypatch)

    def _unexpected_resolver(_model_path: str) -> Path:
        msg = "repo resolution should be skipped once defaults are already seeded"
        raise AssertionError(msg)

    monkeypatch.setattr(
        server_module,
        "_resolve_generation_config_model_dir",
        _unexpected_resolver,
        raising=False,
    )

    handler = server_module.create_handler_from_config(model_config)

    assert handler._uses_model_sampling_defaults is True
    assert handler.default_temperature == pytest.approx(GENERATION_CONFIG_DEFAULTS["temperature"])
    assert handler.default_top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert handler.default_top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert handler.default_min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert handler.default_repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert handler.default_max_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]


@pytest.mark.asyncio
async def test_handler_process_proxy_spawn_failure_rolls_back_proxy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawn failures should rollback running state and retire the reader thread."""

    model_cfg = config_module.ModelEntryConfig(
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

    started_threads: list[_LeakyReaderThread] = []

    class _LeakyReaderThread:
        """Thread stub that only exits when ``join()`` is called."""

        def __init__(
            self,
            *,
            target: Any,
            daemon: bool,
            name: str,
        ) -> None:
            del target, daemon, name
            self.started = False
            self.join_calls: list[float | None] = []
            started_threads.append(self)

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)
            self.started = False

    class _FailingProcess:
        """Process stub that fails immediately on spawn."""

        def __init__(
            self,
            *,
            target: Any,
            args: tuple[Any, ...],
            name: str,
        ) -> None:
            del target, args, name
            self.pid: int | None = None

        def start(self) -> None:
            msg = "boom-start"
            raise RuntimeError(msg)

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(handler_process_module.threading, "Thread", _LeakyReaderThread)
    proxy._ctx = types.SimpleNamespace(Process=_FailingProcess)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="boom-start"):
        await proxy.start({"queue_size": 5, "timeout": 19})

    assert proxy._running is False
    assert "__ready__" not in proxy._pending
    assert proxy._reader_thread is None
    assert len(started_threads) == 1
    assert started_threads[0].join_calls
    assert started_threads[0].started is False


def test_single_model_cli_defaults_beat_generation_config_on_handler_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CLI defaults should survive single-model generation-config seeding."""

    model_dir = tmp_path / "model-cli"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    cli_config = config_module.MLXServerConfig(
        model_path=str(model_dir),
        model_type="lm",
        default_temperature=0.9,
        default_top_p=0.95,
        default_top_k=31,
        default_min_p=0.03,
        default_repetition_penalty=1.11,
        default_max_tokens=2048,
    )

    model_config = cli_config.to_model_entry_config()

    assert model_config.default_temperature == pytest.approx(0.9)
    assert model_config.default_top_p == pytest.approx(0.95)
    assert model_config.default_top_k == 31
    assert model_config.default_min_p == pytest.approx(0.03)
    assert model_config.default_repetition_penalty == pytest.approx(1.11)
    assert model_config.default_max_tokens == 2048
    assert config_module.should_attempt_generation_config_seeding(model_config) is False

    server_module = _load_server_module(monkeypatch)
    handler = server_module.create_handler_from_config(model_config)

    assert handler.default_temperature == pytest.approx(0.9)
    assert handler.default_top_p == pytest.approx(0.95)
    assert handler.default_top_k == 31
    assert handler.default_min_p == pytest.approx(0.03)
    assert handler.default_repetition_penalty == pytest.approx(1.11)
    assert handler.default_max_tokens == 2048


@pytest.mark.asyncio
async def test_generation_config_seeded_defaults_beat_env_for_omitted_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded defaults should beat env fallback for omitted chat and Responses fields."""

    model_dir = tmp_path / "model-a"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-a\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    endpoints_module = _load_endpoints_module()
    captured_chat_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_chat_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = types.SimpleNamespace(
        handler_type="lm",
        _uses_model_sampling_defaults=True,
        default_temperature=model_config.default_temperature,
        default_top_p=model_config.default_top_p,
        default_top_k=model_config.default_top_k,
        default_min_p=model_config.default_min_p,
        default_repetition_penalty=model_config.default_repetition_penalty,
        default_max_tokens=model_config.default_max_tokens,
    )
    registry = _FakeRegistry({"model-a": handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    chat_request = ChatCompletionRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
    )
    responses_request = ResponsesRequest(
        model="model-a",
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_output_tokens=None,
    )

    await endpoints_module.chat_completions(chat_request, _make_raw_request(registry))
    refined_responses_request = endpoints_module.refine_responses_request(
        responses_request,
        handler,
    )

    assert captured_chat_requests[0].temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert captured_chat_requests[0].top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert captured_chat_requests[0].top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert captured_chat_requests[0].min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert captured_chat_requests[0].repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert (
        captured_chat_requests[0].max_completion_tokens
        == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]
    )

    assert refined_responses_request.temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert refined_responses_request.top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert refined_responses_request.top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert refined_responses_request.min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert refined_responses_request.repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert (
        refined_responses_request.max_output_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]
    )


@pytest.mark.asyncio
async def test_multimodal_generation_config_seeded_defaults_beat_env_for_omitted_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multimodal chat requests should also use generation-config-seeded defaults."""

    model_dir = tmp_path / "model-mm"
    _write_model_dir(model_dir, GENERATION_CONFIG_DEFAULTS)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        f"  - model_path: {model_dir}\n"
        "    model_type: multimodal\n"
        "    model_id: model-mm\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ChatCompletionRequest] = []

    async def _fake_process_multimodal_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = types.SimpleNamespace(
        handler_type="multimodal",
        _uses_model_sampling_defaults=True,
        default_temperature=model_config.default_temperature,
        default_top_p=model_config.default_top_p,
        default_top_k=model_config.default_top_k,
        default_min_p=model_config.default_min_p,
        default_repetition_penalty=model_config.default_repetition_penalty,
        default_max_tokens=model_config.default_max_tokens,
    )
    registry = _FakeRegistry({"model-mm": handler})

    monkeypatch.setattr(
        endpoints_module, "process_multimodal_request", _fake_process_multimodal_request
    )
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    chat_request = ChatCompletionRequest(
        model="model-mm",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
    )

    await endpoints_module.chat_completions(chat_request, _make_raw_request(registry))

    assert captured_requests[0].temperature == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["temperature"]
    )
    assert captured_requests[0].top_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["top_p"])
    assert captured_requests[0].top_k == GENERATION_CONFIG_DEFAULTS["top_k"]
    assert captured_requests[0].min_p == pytest.approx(GENERATION_CONFIG_DEFAULTS["min_p"])
    assert captured_requests[0].repetition_penalty == pytest.approx(
        GENERATION_CONFIG_DEFAULTS["repetition_penalty"]
    )
    assert (
        captured_requests[0].max_completion_tokens == GENERATION_CONFIG_DEFAULTS["max_new_tokens"]
    )


@pytest.mark.asyncio
async def test_invalid_generation_config_keeps_env_fallback_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid generation-config files should not block load or env fallback."""

    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text("{not-json", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-a\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    assert model_config.default_temperature is None
    assert model_config.default_top_p is None
    assert model_config.default_top_k is None
    assert model_config.default_min_p is None
    assert model_config.default_repetition_penalty is None
    assert model_config.default_max_tokens is None

    endpoints_module = _load_endpoints_module()
    captured_chat_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_chat_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = types.SimpleNamespace(
        handler_type="lm",
        _uses_model_sampling_defaults=True,
        default_temperature=model_config.default_temperature,
        default_top_p=model_config.default_top_p,
        default_top_k=model_config.default_top_k,
        default_min_p=model_config.default_min_p,
        default_repetition_penalty=model_config.default_repetition_penalty,
        default_max_tokens=model_config.default_max_tokens,
    )
    registry = _FakeRegistry({"model-a": handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    chat_request = ChatCompletionRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
    )
    responses_request = ResponsesRequest(
        model="model-a",
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_output_tokens=None,
    )

    await endpoints_module.chat_completions(chat_request, _make_raw_request(registry))
    refined_responses_request = endpoints_module.refine_responses_request(
        responses_request,
        handler,
    )

    assert captured_chat_requests[0].temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert captured_chat_requests[0].top_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"])
    )
    assert captured_chat_requests[0].top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert captured_chat_requests[0].min_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"])
    )
    assert captured_chat_requests[0].repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert captured_chat_requests[0].max_completion_tokens == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"]
    )

    assert refined_responses_request.temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert refined_responses_request.top_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"])
    )
    assert refined_responses_request.top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert refined_responses_request.min_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"])
    )
    assert refined_responses_request.repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert refined_responses_request.max_output_tokens == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"]
    )


@pytest.mark.asyncio
async def test_non_utf8_generation_config_keeps_env_fallback_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-UTF-8 generation-config files should not block load or env fallback."""

    model_dir = tmp_path / "model-b"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_bytes(b"\xff\xfe\x80")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-b\n",
        encoding="utf-8",
    )
    model_config = load_config_from_yaml(str(config_path)).models[0]

    assert model_config.default_temperature is None
    assert model_config.default_top_p is None
    assert model_config.default_top_k is None
    assert model_config.default_min_p is None
    assert model_config.default_repetition_penalty is None
    assert model_config.default_max_tokens is None

    endpoints_module = _load_endpoints_module()
    captured_chat_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_chat_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = types.SimpleNamespace(
        handler_type="lm",
        _uses_model_sampling_defaults=True,
        default_temperature=model_config.default_temperature,
        default_top_p=model_config.default_top_p,
        default_top_k=model_config.default_top_k,
        default_min_p=model_config.default_min_p,
        default_repetition_penalty=model_config.default_repetition_penalty,
        default_max_tokens=model_config.default_max_tokens,
    )
    registry = _FakeRegistry({"model-b": handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    chat_request = ChatCompletionRequest(
        model="model-b",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
    )
    responses_request = ResponsesRequest(
        model="model-b",
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        max_output_tokens=None,
    )

    await endpoints_module.chat_completions(chat_request, _make_raw_request(registry))
    refined_responses_request = endpoints_module.refine_responses_request(
        responses_request,
        handler,
    )

    assert captured_chat_requests[0].temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert captured_chat_requests[0].top_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"])
    )
    assert captured_chat_requests[0].top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert captured_chat_requests[0].min_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"])
    )
    assert captured_chat_requests[0].repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert captured_chat_requests[0].max_completion_tokens == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"]
    )

    assert refined_responses_request.temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert refined_responses_request.top_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"])
    )
    assert refined_responses_request.top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert refined_responses_request.min_p == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"])
    )
    assert refined_responses_request.repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert refined_responses_request.max_output_tokens == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"]
    )


def test_generation_config_warning_logs_interpolated_details(tmp_path: Path) -> None:
    """Generation-config warnings should include concrete model and file details."""

    model_dir = tmp_path / "model-c"
    model_dir.mkdir()
    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text("{not-json", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-c\n",
        encoding="utf-8",
    )

    warning_messages: list[str] = []
    sink_id = config_module.logger.add(
        lambda message: warning_messages.append(str(message).rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        load_config_from_yaml(str(config_path))
    finally:
        config_module.logger.remove(sink_id)

    assert any(
        "Failed to read generation config for model" in message for message in warning_messages
    )
    assert any(str(model_dir) in message for message in warning_messages)
    assert any(str(generation_config_path) in message for message in warning_messages)


def test_non_object_generation_config_warning_logs_interpolated_details(
    tmp_path: Path,
) -> None:
    """Non-object generation-config warnings should include concrete model details."""

    model_dir = tmp_path / "model-d"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text("[]", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"models:\n  - model_path: {model_dir}\n    model_type: lm\n    model_id: model-d\n",
        encoding="utf-8",
    )

    warning_messages: list[str] = []
    sink_id = config_module.logger.add(
        lambda message: warning_messages.append(str(message).rstrip()),
        level="WARNING",
        format="{message}",
    )
    try:
        load_config_from_yaml(str(config_path))
    finally:
        config_module.logger.remove(sink_id)

    assert any("Ignoring generation config for model" in message for message in warning_messages)
    assert any(str(model_dir) in message for message in warning_messages)
