from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from os import PathLike, path
import pickle
from functools import lru_cache
import json
from collections import defaultdict


class BaseMemoryLoader(ABC):
    """Abstract base class for memory loaders."""

    @abstractmethod
    def get(self, query: str, *args, **kwargs) -> Any:
        """Retrieve information based on the query."""
        pass


class NaiveMemoryLoader(BaseMemoryLoader):
    """A naive memory loader that loads contexts from a JSONL file based
    on context id."""

    def __init__(
        self,
        jsonl_path: PathLike[str] | str,
        cached_index_path: Optional[PathLike[str] | str] = None,
        contexts_cache_size: int = 128,
    ) -> None:
        """
        Initialize the NaiveMemoryLoader.

        Args:
            jsonl_path (PathLike[str] | str): Path to the JSONL file containing
                memory data.
            cached_index_path (PathLike[str] | str | None): Path to the cached
                index file. If None, index will be built from scratch.
            contexts_cache_size (int): Size of the cache for contexts. Default
                is 128.

        """
        # make sure jsonl_path is an absolute path
        self.jsonl_path = path.abspath(jsonl_path)
        self.index_path = cached_index_path
        self._file_ctx = open(self.jsonl_path, "r", encoding="utf-8")

        # if cached_index_path is not None:
        if self.index_path is not None and path.exists(self.index_path):
            # load the cached index
            with open(self.index_path, "wb") as f:
                self.index = pickle.load(f)  # type: ignore
        else:
            self._index = self._build_index()
            if self.index_path:
                with open(self.index_path, "wb") as f:
                    pickle.dump(self._index, f)

        self._cached_contexts = lru_cache(maxsize=contexts_cache_size)(
            self._load_contexts
        )

    def _build_index(self) -> dict[str, int]:
        """
        Build an index from the JSONL file.
        The index maps keys to their byte offsets in the file.
        Returns:
            dict[str, int]: A dictionary mapping keys to their byte offsets in
                the file.
        """
        idx = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            # Readall lines
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                key = next(iter(json.loads(line).keys()))
                idx[key] = offset
        return idx

    def _load_contexts(self, key: str) -> Any:
        """Load contexts for a given key from the JSONL file.
        Args:
            key (str): The key for which to load contexts.
        Returns:
            Any: The contexts associated with the key.
        """
        if key not in self._index:
            raise KeyError(f"Key '{key}' not found in index.")
        self._file_ctx.seek(self._index[key])
        line = self._file_ctx.readline()
        return json.loads(line)[key]

    def get(self, query: str, end_index: Optional[int] = None) -> Any:
        context = self._cached_contexts(query)
        if end_index is not None:
            if end_index > len(context):
                raise ValueError(
                    f"end_index {end_index} is greater than "
                    f"the length of context {len(context)}"
                )
            context = context[:end_index]
        return context


class PersonaMemoryLoader(BaseMemoryLoader):
    """
    A memory loader that loads contexts based on Persona.
    """

    def __init__(
        self,
        jsonl_path: PathLike[str] | str,
    ) -> None:
        self.jsonl_path = path.abspath(jsonl_path)
        self._naive_loader = NaiveMemoryLoader(
            self.jsonl_path, contexts_cache_size=128
        )

    def _group_by_persona(self) -> Tuple[dict[str, list[str]], dict[str, str]]:
        persona_to_ids = defaultdict(list)
        id_to_persona = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                context = json.loads(line)
                assert (
                    len(context) == 1
                ), "Each line should contain a single context."
                context_id, messages = next(iter(context.items()))
                persona = messages[0]["content"]
                persona_to_ids[persona].append(context_id)
                id_to_persona[context_id] = persona
        return persona_to_ids, id_to_persona


class PersonaRAGMemoryLoader(BaseMemoryLoader):
    pass


class MemGPTMemoryLoader(BaseMemoryLoader):
    pass


class GraphitiMemoryLoader(BaseMemoryLoader):
    """Graphiti Memory Loader"""

    pass


if __name__ == "__main__":
    # Example usage
    from utils import get_datasets

    # Test NaiveMemoryLoader
    dataset, shared_contexts_path = get_datasets("32k", "data/personamem")
    memory = NaiveMemoryLoader(
        shared_contexts_path, "data/personamem_index_32k.pkl"
    )
    context_id = dataset[0]["shared_context_id"]
    end_idx = dataset[0]["end_index_in_shared_context"]
    context = memory.get(context_id, end_index=end_idx)
    print(context)

    # Test PersonaMemoryLoader
