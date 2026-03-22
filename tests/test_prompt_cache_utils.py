"""Utility-level regressions for prompt-cache storage and eviction behavior."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


class _FakeLayer:
    """Minimal cache layer with token-backed byte accounting."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)

    @property
    def nbytes(self) -> int:
        """Return a deterministic byte size for the cached tokens."""

        return len(self.tokens) * 20


def _install_fake_mlx_cache_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``mlx_lm.models.cache`` stub for prompt-cache tests."""

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    def can_trim_prompt_cache(cache: list[object]) -> bool:
        return bool(cache)

    def trim_prompt_cache(cache: list[object], num_tokens: int) -> int:
        for layer in cache:
            if isinstance(layer, _FakeLayer):
                layer.tokens = layer.tokens[:-num_tokens] if num_tokens else layer.tokens[:]
        return num_tokens

    fake_cache.can_trim_prompt_cache = can_trim_prompt_cache
    fake_cache.trim_prompt_cache = trim_prompt_cache
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)


def _load_prompt_cache_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import ``app.utils.prompt_cache`` with stubbed MLX cache helpers."""

    _install_fake_mlx_cache_module(monkeypatch)
    sys.modules.pop("app.utils.prompt_cache", None)
    module = importlib.import_module("app.utils.prompt_cache")
    return importlib.reload(module)


def _make_prompt_cache(tokens: list[int]) -> list[_FakeLayer]:
    """Create a deterministic fake prompt cache for a token sequence."""

    return [_FakeLayer(tokens)]


def _assert_cache_accounting_bounds(
    cache: object,
    *,
    min_entries: int,
    max_entries: int,
    min_bytes: int,
    max_bytes: int,
) -> None:
    """Assert bounded cache-accounting invariants without fixing one storage layout."""

    assert min_entries <= len(cache) <= max_entries
    assert min_bytes <= cache.nbytes <= max_bytes


def test_insert_cache_evicts_single_oversized_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single entry larger than ``max_bytes`` should not remain cached."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=0)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    assert len(cache) == 0
    assert cache.nbytes == 0
    nearest_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3, 4, 5])
    assert nearest_cache is None
    assert remaining_tokens == [1, 2, 3, 4, 5]


def test_insert_cache_preserves_exact_longer_hit_on_regenerate_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branching to a sibling should not discard the better exact longer hit."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=1000)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    branched_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3])

    assert branched_cache is not None
    assert remaining_tokens == [3]

    # Simulate generation extending the trimmed cache into a new sibling branch.
    branched_cache[0].tokens = [1, 2, 3, 6, 7]
    cache.insert_cache([1, 2, 3, 6, 7], branched_cache)

    new_sequence_cache, new_sequence_remaining = cache.fetch_nearest_cache([1, 2, 3, 6, 7])
    old_sequence_cache, old_sequence_remaining = cache.fetch_nearest_cache([1, 2, 3, 4, 5])

    assert new_sequence_cache is not None
    assert new_sequence_remaining == []
    assert old_sequence_cache is not None
    assert old_sequence_remaining == []
    _assert_cache_accounting_bounds(
        cache,
        min_entries=2,
        max_entries=3,
        min_bytes=200,
        max_bytes=260,
    )


def test_fetch_nearest_cache_returns_isolated_trimmed_longer_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trimmed longer-hit fetches should not mutate the stored exact sequence."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=1000)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    trimmed_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3])

    assert trimmed_cache is not None
    assert remaining_tokens == [3]

    trimmed_cache[0].tokens = [9, 9]

    exact_cache, exact_remaining = cache.fetch_nearest_cache([1, 2, 3, 4, 5])

    assert exact_cache is not None
    assert exact_remaining == []
    assert exact_cache[0].tokens == [1, 2, 3, 4, 5]


def test_insert_cache_reinserts_trimmed_prefix_as_exact_hit_without_losing_longer_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reinserting a trimmed prefix should preserve both prefix and longer exact hits."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=1000)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    trimmed_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3])

    assert trimmed_cache is not None
    assert remaining_tokens == [3]

    trimmed_cache[0].tokens = [1, 2, 3]
    cache.insert_cache([1, 2, 3], trimmed_cache)

    prefix_cache, prefix_remaining = cache.fetch_nearest_cache([1, 2, 3])
    longer_cache, longer_remaining = cache.fetch_nearest_cache([1, 2, 3, 4, 5])

    assert prefix_cache is not None
    assert prefix_remaining == []
    assert prefix_cache[0].tokens == [1, 2, 3]
    assert longer_cache is not None
    assert longer_remaining == []
    assert longer_cache[0].tokens == [1, 2, 3, 4, 5]
    _assert_cache_accounting_bounds(
        cache,
        min_entries=2,
        max_entries=2,
        min_bytes=160,
        max_bytes=160,
    )


def test_insert_cache_preserves_exact_longer_hit_when_reinserting_shorter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reinserting a shorter prefix should not discard the original exact longer hit."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=1000)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    shorter_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3])

    assert shorter_cache is not None
    assert remaining_tokens == [3]

    shorter_cache[0].tokens = [1, 2, 3]
    cache.insert_cache([1, 2, 3], shorter_cache)

    prefix_cache, prefix_remaining = cache.fetch_nearest_cache([1, 2, 3])
    exact_cache, exact_remaining = cache.fetch_nearest_cache([1, 2, 3, 4, 5])

    assert prefix_cache is not None
    assert prefix_remaining == []
    assert exact_cache is not None
    assert exact_remaining == []
    _assert_cache_accounting_bounds(
        cache,
        min_entries=2,
        max_entries=3,
        min_bytes=160,
        max_bytes=260,
    )


def test_insert_cache_preserves_exact_longer_hit_after_second_generation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second-generation branching off a reinserted shorter prefix should not evict the old exact longer hit."""

    prompt_cache_module = _load_prompt_cache_module(monkeypatch)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, max_bytes=1000)

    cache.insert_cache([1, 2, 3, 4, 5], _make_prompt_cache([1, 2, 3, 4, 5]))

    shorter_cache, remaining_tokens = cache.fetch_nearest_cache([1, 2, 3])
    assert shorter_cache is not None
    assert remaining_tokens == [3]

    shorter_cache[0].tokens = [1, 2, 3]
    cache.insert_cache([1, 2, 3], shorter_cache)

    second_generation_cache, second_generation_remaining = cache.fetch_nearest_cache([1, 2, 3])
    assert second_generation_cache is not None
    assert second_generation_remaining == []

    second_generation_cache[0].tokens = [1, 2, 3, 6, 7]
    cache.insert_cache([1, 2, 3, 6, 7], second_generation_cache)

    branch_cache, branch_remaining = cache.fetch_nearest_cache([1, 2, 3, 6, 7])
    exact_cache, exact_remaining = cache.fetch_nearest_cache([1, 2, 3, 4, 5])

    assert branch_cache is not None
    assert branch_remaining == []
    assert exact_cache is not None
    assert exact_remaining == []
    _assert_cache_accounting_bounds(
        cache,
        min_entries=2,
        max_entries=3,
        min_bytes=200,
        max_bytes=260,
    )
