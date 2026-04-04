# modified from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
from typing import Any

from loguru import logger
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache


@dataclass
class PromptTrieResult:
    """Result of searching the trie for a token sequence.

    Parameters
    ----------
    exact : list[int] | None
        Exact matching token sequence, if found.
    shorter : list[int] | None
        Shorter prefix match, if found.
    longer : list[int] | None
        Longer sequence containing the query as a prefix, if found.
    common_prefix : int
        Length of common prefix with matching cache entries.
    """

    exact: list[int] | None
    shorter: list[int] | None
    longer: list[int] | None
    common_prefix: int


class PromptTrie:
    """Prefix trie for storing prompt caches keyed by token sequences."""

    def __init__(self) -> None:
        self._trie: dict[int, Any] = {}

    def add(self, tokens: list[int], value: Any) -> Any:
        """Insert a value and return the previous value if any."""
        current = self._trie
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, tokens: list[int]) -> Any:
        """Exact lookup by token sequence."""
        current = self._trie
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, tokens: list[int]) -> Any:
        """Remove and return the value at the given token sequence."""
        path = [self._trie]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, tokens: list[int]) -> list[tuple[int, Any]]:
        """Remove all prefix entries along the path to *tokens*."""
        values = []
        current = self._trie
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, tokens: list[int]) -> PromptTrieResult:
        """Search for exact, shorter, or longer matches."""
        if not self._trie:
            return PromptTrieResult(None, None, None, 0)

        current = self._trie

        if not tokens and "__value__" in current:
            return PromptTrieResult([], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer (DFS with pruning)
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    for tok in current:
                        stack.append((current[tok], [*extra, tok]))
            if best is not None:
                longer = tokens[:index] + best

        return PromptTrieResult(None, shorter, longer, common_prefix)


class LRUPromptCache:
    """LRU cache for MLX prompt KV caches.

    The cache stores token sequences in a trie so it can efficiently find exact
    matches, shorter prefixes, and longer cached sequences that can be trimmed
    down to a requested prefix. Entries are evicted using an LRU policy with
    optional byte-based trimming.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of cache entries to retain, by default 10.
    max_bytes : int, optional
        Maximum total bytes to retain across cached prompt caches, by default a
        practically unbounded value.
    """

    @dataclass
    class CacheEntry:
        """Stored prompt cache entry."""

        prompt_cache: list[Any]
        nbytes: int
        cache_type: str

    class CacheOrder:
        """Track cache recency with priority-based eviction."""

        def __init__(self, ordering: list[str] | None = None) -> None:
            if ordering is None:
                ordering = ["assistant", "user", "system"]
            self._ordering = ordering
            self._lrus: dict[str, deque[tuple[int, ...]]] = {k: deque() for k in ordering}

        def __len__(self) -> int:
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, tokens: tuple[int, ...], cache_type: str = "assistant") -> None:
            self._lrus[cache_type].append(tokens)

        def remove(self, tokens: tuple[int, ...]) -> None:
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove(tokens)
                    break
                except ValueError:
                    pass

        def pop(self) -> tuple[int, ...]:
            """Pop the least-recently-used entry, favouring lower-priority types."""
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            # Fall through to the last queue
            return self._lrus[self._ordering[-1]].popleft()

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63) -> None:
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._trie = PromptTrie()
        self._lru = self.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type: dict[str, int] = dict.fromkeys(self._lru._ordering, 0)

    def __len__(self) -> int:
        return len(self._lru)

    @property
    def nbytes(self) -> int:
        return self._n_bytes

    def fetch_nearest_cache(
        self,
        tokens_ids: list[int],
    ) -> tuple[list[Any] | None, list[int]]:
        """Fetch the nearest matching cache for the given token sequence.

        Returns
        -------
        tuple[list[Any] | None, list[int]]
            Tuple of (prompt_cache, remaining_tokens). If no cache found,
            returns (None, original_tokens).
        """
        result = self._trie.search(tokens_ids)
        if result.exact is not None:
            cache_entry = self._trie.get(result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens_ids) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens_ids[prefix:]

        if short_length > 0:
            cache_entry = self._trie.get(result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens_ids[short_length:]

        return None, tokens_ids

    def insert_cache(
        self,
        tokens_ids: list[int],
        prompt_cache: list[Any],
        *,
        cache_type: str = "assistant",
    ) -> None:
        """Insert or update a cache entry.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying this cache entry.
        prompt_cache : list[Any]
            The prompt cache data to store.
        cache_type : str, optional
            Priority category for eviction ordering, by default ``"assistant"``.
        """
        tokens_tuple = tuple(tokens_ids)

        # Make the cache entry
        entry = self.CacheEntry(
            prompt_cache, sum(getattr(c, "nbytes", 0) for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(tokens_ids, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(tokens_tuple)
        self._lru.push(tokens_tuple, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if can_trim_prompt_cache(prompt_cache):
            for prefix_len, removed_entry in self._trie.pop_prefixes(tokens_ids):
                self._n_bytes -= removed_entry.nbytes
                self._n_bytes_by_type[removed_entry.cache_type] -= removed_entry.nbytes
                self._lru.remove(tuple(tokens_ids[:prefix_len]))

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            evicted = self._lru.pop()
            evicted_entry = self._trie.pop(list(evicted))
            self._n_bytes -= evicted_entry.nbytes
            self._n_bytes_by_type[evicted_entry.cache_type] -= evicted_entry.nbytes

        while self._n_bytes > self.max_bytes:
            evicted = self._lru.pop()
            evicted_entry = self._trie.pop(list(evicted))
            self._n_bytes -= evicted_entry.nbytes
            self._n_bytes_by_type[evicted_entry.cache_type] -= evicted_entry.nbytes

    def trim_to(self, *, n_sequences: int | None = None, n_bytes: int | None = None) -> None:
        """Trim the cache down to sequence and/or byte limits."""
        max_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        max_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > max_sequences:
            evicted = self._lru.pop()
            evicted_entry = self._trie.pop(list(evicted))
            self._n_bytes -= evicted_entry.nbytes
            self._n_bytes_by_type[evicted_entry.cache_type] -= evicted_entry.nbytes

        while self._n_bytes > max_bytes:
            evicted = self._lru.pop()
            evicted_entry = self._trie.pop(list(evicted))
            self._n_bytes -= evicted_entry.nbytes
            self._n_bytes_by_type[evicted_entry.cache_type] -= evicted_entry.nbytes

    def stats_by_type(self) -> dict[str, dict[str, int]]:
        """Return per-type sequence count and byte usage."""
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result

    def log_cache_stats(self) -> None:
        """Log the current cache size, bytes, and per-type stats."""
        logger.info(
            "KV Caches: {} seq, {:.2f} GB",
            len(self),
            self.nbytes / 1e9,
        )


if __name__ == "__main__":
    from app.models.mlx_lm import MLX_LM

    model_path = "mlx-community/Qwen3-Coder-Next-8bit"
    draft_model_path = "mlx-community/Qwen3-Coder-Next-4bit"
    model = MLX_LM(model_path, draft_model_path)
    prompt_cache = LRUPromptCache()

    import time

    start_time = time.time()
    first_token = True

    prompt_1 = "Hello, how are you? I'm fine, thank you."
    input_prompt = model.create_input_prompt([{"role": "user", "content": prompt_1}], {})
    input_ids = model.encode_prompt(input_prompt)

    cache, rest_input_ids = prompt_cache.fetch_nearest_cache(input_ids)
    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key = input_ids[:]

    response_1 = model(rest_input_ids, cache, stream=True)
    for chunk in response_1:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            cache_key.append(chunk.token)

    prompt_cache.insert_cache(cache_key, cache)

    start_time = time.time()
    first_token = True
    prompt_2 = "Hello, how are you? I'm fine, thank you."
    input_prompt_2 = model.create_input_prompt([{"role": "user", "content": prompt_2}], {})
    input_ids_2 = model.encode_prompt(input_prompt_2)
    cache, rest_input_ids_2 = prompt_cache.fetch_nearest_cache(input_ids_2)

    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key_2 = input_ids_2[:]

    start_time = time.time()
    response_2 = model(rest_input_ids_2, cache, stream=True)
    raw_text = ""
    for chunk in response_2:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            raw_text += chunk.text
            cache_key_2.append(chunk.token)

    print("RAW TEXT", raw_text)

    prompt_cache.insert_cache(cache_key_2, cache)
