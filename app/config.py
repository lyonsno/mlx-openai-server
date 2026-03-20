"""Server configuration dataclass and helpers.

This module exposes ``MLXServerConfig``, a dataclass that holds all CLI
configuration values for the server. The dataclass performs minimal
normalization in ``__post_init__`` (parsing comma-separated LoRA
arguments and applying small model-type-specific defaults).

It also provides ``ModelEntryConfig`` and ``MultiModelServerConfig``
for YAML-based multi-handler configurations, along with the
``load_config_from_yaml`` helper that parses a YAML file into
these structures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
from loguru import logger

from .message_converters import resolve_message_converter_name


@dataclass
class MLXServerConfig:
    """Container for server CLI configuration values.

    The class mirrors the Click CLI options and normalizes a few fields
    during initialization (for example converting comma-separated
    strings into lists and setting sensible defaults for image model
    configurations).
    """

    model_path: str
    model_type: str = "lm"
    context_length: int | None = None
    port: int = 8000
    host: str = "0.0.0.0"
    max_concurrency: int = 1
    queue_timeout: int = 300
    queue_size: int = 100
    disable_auto_resize: bool = False
    quantize: int | None = None
    config_name: str | None = None
    lora_paths: list[str] | None = field(default=None, init=False)
    lora_scales: list[float] | None = field(default=None, init=False)
    log_file: str | None = None
    no_log_file: bool = False
    log_level: str = "INFO"
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    trust_remote_code: bool = False
    chat_template_file: str | None = None
    debug: bool = False
    prompt_cache_size: int = 10
    prompt_cache_max_bytes: int = 1 << 63
    draft_model_path: str | None = None
    num_draft_tokens: int = 2

    # Default sampling parameters (override DEFAULT_* env when set via CLI)
    default_max_tokens: int = 100000
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    default_top_k: int = 20
    default_min_p: float = 0.0
    default_repetition_penalty: float = 1.0
    default_presence_penalty: float = 0.0
    default_xtc_probability: float = 0.0
    default_xtc_threshold: float = 0.0
    default_seed: int = 0
    default_repetition_context_size: int = 20

    # Used to capture raw CLI input before processing
    lora_paths_str: str | None = None
    lora_scales_str: str | None = None

    def __post_init__(self) -> None:
        """Normalize certain CLI fields after instantiation.

        - Convert comma-separated ``lora_paths`` and ``lora_scales`` into
          lists when provided.
        - Apply small model-type-specific defaults for ``config_name``
          and emit warnings when values appear inconsistent.
        """

        # Process comma-separated LoRA paths and scales into lists (or None)
        if self.lora_paths_str:
            self.lora_paths = [p.strip() for p in self.lora_paths_str.split(",") if p.strip()]

        if self.lora_scales_str:
            try:
                self.lora_scales = [
                    float(s.strip()) for s in self.lora_scales_str.split(",") if s.strip()
                ]
            except ValueError:
                # If parsing fails, log and set to None
                logger.warning("Failed to parse lora_scales into floats; ignoring lora_scales")
                self.lora_scales = None

        # Validate that config name is only used with image-generation and
        # image-edit model types. If missing for those types, set defaults.
        if self.config_name and self.model_type not in ["image-generation", "image-edit"]:
            logger.warning(
                "Config name parameter '%s' provided but model type is '%s'. "
                "Config name is only used with image-generation "
                "and image-edit models.",
                self.config_name,
                self.model_type,
            )
        elif self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model type is 'image-generation' but no config name "
                "specified. Using default 'flux-schnell'."
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model type is 'image-edit' but no config name "
                "specified. Using default 'flux-kontext-dev'."
            )
            self.config_name = "flux-kontext-dev"

        # Speculative decoding (draft model) is only supported for lm model type
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model / num-draft-tokens are only supported for model type 'lm'. "
                "Ignoring speculative decoding options."
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2

        if self.message_converter is not None:
            self.message_converter = self.message_converter.lower()
        elif self.model_type in {"lm", "multimodal"}:
            self.message_converter = resolve_message_converter_name(
                tool_parser_name=self.tool_call_parser,
                reasoning_parser_name=self.reasoning_parser,
            )

    @property
    def model_identifier(self) -> str:
        """Get the appropriate model identifier based on model type.

        For Flux models, we always use model_path (local directory path).
        """
        return self.model_path


# ---------------------------------------------------------------------------
# Multi-model YAML configuration
# ---------------------------------------------------------------------------

VALID_MODEL_TYPES = frozenset(
    {"lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"}
)


@dataclass
class ModelEntryConfig:
    """Configuration for a single model entry in a multi-model YAML config.

    Each entry maps to exactly one handler that will be registered in
    the ``ModelRegistry``.  The ``model_id`` defaults to ``model_path``
    when not set explicitly, giving callers a short alias they can use
    in API requests.
    """

    model_path: str
    model_type: str = "lm"
    model_id: str | None = None

    # Common options
    context_length: int | None = None
    max_concurrency: int = 1
    queue_timeout: int = 300
    queue_size: int = 100

    # Image-generation / image-edit options
    quantize: int | None = None
    config_name: str | None = None

    # LoRA options
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None

    # LM / multimodal options
    disable_auto_resize: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    message_converter: str | None = None
    trust_remote_code: bool = False
    chat_template_file: str | None = None
    debug: bool = False
    prompt_cache_size: int = 10
    prompt_cache_max_bytes: int = 1 << 63
    draft_model_path: str | None = None
    num_draft_tokens: int = 2
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_repetition_penalty: float | None = None
    default_presence_penalty: float | None = None
    default_xtc_probability: float | None = None
    default_xtc_threshold: float | None = None
    default_seed: int | None = None
    default_repetition_context_size: int | None = None
    generation_config_seed_attempted: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Resolve ``model_id`` and validate ``model_type``."""
        if self.model_id is None:
            self.model_id = self.model_path

        if self.model_type not in VALID_MODEL_TYPES:
            msg = (
                f"Invalid model_type '{self.model_type}' for model '{self.model_path}'. "
                f"Must be one of {sorted(VALID_MODEL_TYPES)}."
            )
            raise ValueError(msg)

        # Apply image-generation / image-edit defaults (same as MLXServerConfig)
        if self.model_type == "image-generation" and not self.config_name:
            logger.warning(
                "Model '%s' (image-generation) has no config_name. Defaulting to 'flux-schnell'.",
                self.model_path,
            )
            self.config_name = "flux-schnell"
        elif self.model_type == "image-edit" and not self.config_name:
            logger.warning(
                "Model '%s' (image-edit) has no config_name. Defaulting to 'flux-kontext-dev'.",
                self.model_path,
            )
            self.config_name = "flux-kontext-dev"

        # Speculative decoding is LM-only
        if self.draft_model_path and self.model_type != "lm":
            logger.warning(
                "Draft model is only supported for 'lm'. Ignoring for model '%s'.",
                self.model_path,
            )
            self.draft_model_path = None
            self.num_draft_tokens = 2

        if self.message_converter is not None:
            self.message_converter = self.message_converter.lower()
        elif self.model_type in {"lm", "multimodal"}:
            self.message_converter = resolve_message_converter_name(
                tool_parser_name=self.tool_call_parser,
                reasoning_parser_name=self.reasoning_parser,
            )


@dataclass
class MultiModelServerConfig:
    """Top-level configuration for running multiple models from a YAML file.

    The ``server`` section holds host/port/logging settings, while
    ``models`` is a list of ``ModelEntryConfig`` entries – each of
    which will be loaded as a separate handler at startup.
    """

    models: list[ModelEntryConfig]
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_file: str | None = None
    no_log_file: bool = False


_GENERATION_CONFIG_TO_DEFAULT_FIELD: dict[str, str] = {
    "max_new_tokens": "default_max_tokens",
    "temperature": "default_temperature",
    "top_p": "default_top_p",
    "top_k": "default_top_k",
    "min_p": "default_min_p",
    "repetition_penalty": "default_repetition_penalty",
}


def _coerce_generation_config_float(value: object) -> float:
    """Coerce a generation-config value to a finite float.

    Bools are rejected even though Python treats them as ints.
    """

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid floats")

    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        result = float(value.strip())
    else:
        raise TypeError("unsupported float value type")

    if not math.isfinite(result):
        raise ValueError("float value must be finite")
    return result


def _coerce_generation_config_int(value: object) -> int:
    """Coerce a generation-config value to an integer without loss.

    Accepts integral ints, integral floats, and integral numeric strings.
    Rejects bools and lossy conversions such as ``1.7``.
    """

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid integers")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("float value must be integral")
        return int(value)

    if isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError("string value must represent an integer") from exc
        if not parsed.is_finite() or parsed != parsed.to_integral_value():
            raise ValueError("string value must represent an integer")
        return int(parsed)

    raise TypeError("unsupported integer value type")


_GENERATION_CONFIG_COERCERS: dict[str, Callable[[object], int | float]] = {
    "default_max_tokens": _coerce_generation_config_int,
    "default_temperature": _coerce_generation_config_float,
    "default_top_p": _coerce_generation_config_float,
    "default_top_k": _coerce_generation_config_int,
    "default_min_p": _coerce_generation_config_float,
    "default_repetition_penalty": _coerce_generation_config_float,
}


def _resolve_generation_config_model_dir(model_path: str) -> Path | None:
    """Resolve a model path to a directory that may contain generation config.

    Local filesystem directories are returned directly. Hugging Face repo IDs
    are resolved best-effort from the local Hugging Face cache only so
    generation-config seeding does not introduce network/download work on the
    main startup path. When a cached repo snapshot resolves successfully, the
    snapshot directory is returned even if it does not contain a
    ``generation_config.json`` so callers can remember that the best-effort
    lookup already happened.
    """

    local_path = Path(model_path)
    if local_path.is_dir():
        return local_path

    if snapshot_download is None:
        return None

    try:
        snapshot_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns="generation_config.json",
            local_files_only=True,
        )
    except Exception as exc:
        logger.warning(
            f"Failed to resolve generation config snapshot for model '{model_path}': {exc}"
        )
        return None

    return Path(snapshot_dir)


def resolve_generation_config_model_dir(model_path: str) -> Path | None:
    """Public wrapper for best-effort generation-config path resolution."""

    return _resolve_generation_config_model_dir(model_path)


def _seed_model_defaults_from_generation_config(
    model_cfg: ModelEntryConfig,
    model_dir: Path | None = None,
) -> None:
    """Best-effort seed missing model defaults from ``generation_config.json``.

    Explicit YAML defaults remain authoritative; this helper only fills
    fields that are still ``None`` on the ``ModelEntryConfig``.
    Missing or malformed generation-config files are tolerated so
    request-time env fallback remains available.
    """

    if model_cfg.model_type not in {"lm", "multimodal"}:
        return

    generation_config_root = model_dir or Path(model_cfg.model_path)
    if generation_config_root.exists():
        model_cfg.generation_config_seed_attempted = True

    generation_config_path = generation_config_root / "generation_config.json"
    if not generation_config_path.exists():
        return

    try:
        generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            f"Failed to read generation config for model '{model_cfg.model_path}' "
            f"from '{generation_config_path}': {exc}"
        )
        return

    if not isinstance(generation_config, dict):
        logger.warning(
            f"Ignoring generation config for model '{model_cfg.model_path}' "
            "because it is not a JSON object."
        )
        return

    for source_key, target_field in _GENERATION_CONFIG_TO_DEFAULT_FIELD.items():
        if getattr(model_cfg, target_field) is not None:
            continue
        source_value = generation_config.get(source_key)
        if source_value is not None:
            try:
                coerced_value = _GENERATION_CONFIG_COERCERS[target_field](source_value)
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    f"Ignoring generation config value for model '{model_cfg.model_path}' "
                    f"because '{source_key}={source_value!r}' is not a valid "
                    f"{target_field} value."
                )
                continue
            setattr(model_cfg, target_field, coerced_value)


def has_missing_generation_config_defaults(model_cfg: ModelEntryConfig) -> bool:
    """Return whether mapped generation-config-backed defaults are still missing."""

    if model_cfg.model_type not in {"lm", "multimodal"}:
        return False

    return any(
        getattr(model_cfg, target_field) is None
        for target_field in _GENERATION_CONFIG_TO_DEFAULT_FIELD.values()
    )


def should_attempt_generation_config_seeding(model_cfg: ModelEntryConfig) -> bool:
    """Return whether generation-config seeding should still be attempted.

    This is stricter than ``has_missing_generation_config_defaults`` because a
    partially populated generation config can legitimately leave some mapped
    defaults unset after seeding. Once a best-effort seeding attempt has
    already happened for a text-capable model, startup should not repeat local
    cache resolution for the same model.
    """

    return (
        has_missing_generation_config_defaults(model_cfg)
        and not model_cfg.generation_config_seed_attempted
    )


def seed_model_defaults_from_generation_config(
    model_cfg: ModelEntryConfig,
    model_dir: Path | None = None,
) -> None:
    """Public wrapper for best-effort generation-config default seeding."""

    _seed_model_defaults_from_generation_config(model_cfg, model_dir=model_dir)


def load_config_from_yaml(config_path: str) -> MultiModelServerConfig:
    """Parse a YAML config file into a ``MultiModelServerConfig``.

    Parameters
    ----------
    config_path : str
        Filesystem path to the YAML configuration file.

    Returns
    -------
    MultiModelServerConfig
        Parsed and validated configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the YAML is missing required keys or contains invalid values.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with path.open("r") as fh:
        raw: dict = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        msg = f"Config file must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    # ---- server section (optional, all keys have defaults) ----
    server_raw: dict = raw.get("server", {})
    if not isinstance(server_raw, dict):
        msg = "'server' section must be a mapping"
        raise ValueError(msg)

    # ---- models section (required, at least one entry) ----
    models_raw: list = raw.get("models", [])
    if not isinstance(models_raw, list) or len(models_raw) == 0:
        msg = "'models' section must be a non-empty list of model entries"
        raise ValueError(msg)

    model_entries: list[ModelEntryConfig] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(models_raw):
        if not isinstance(entry, dict):
            msg = f"Model entry at index {idx} must be a mapping"
            raise ValueError(msg)

        if "model_path" not in entry:
            msg = f"Model entry at index {idx} is missing required key 'model_path'"
            raise ValueError(msg)

        model_cfg = ModelEntryConfig(**entry)

        # Enforce unique model_id values
        if model_cfg.model_id in seen_ids:
            msg = (
                f"Duplicate model_id '{model_cfg.model_id}' in config. "
                "Each model must have a unique model_id."
            )
            raise ValueError(msg)
        if should_attempt_generation_config_seeding(model_cfg):
            _seed_model_defaults_from_generation_config(
                model_cfg,
                model_dir=_resolve_generation_config_model_dir(model_cfg.model_path),
            )
        seen_ids.add(model_cfg.model_id)
        model_entries.append(model_cfg)

    return MultiModelServerConfig(
        models=model_entries,
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8000),
        log_level=server_raw.get("log_level", "INFO"),
        log_file=server_raw.get("log_file"),
        no_log_file=server_raw.get("no_log_file", False),
    )
