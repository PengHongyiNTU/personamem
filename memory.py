from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import random
from typing import Any, Optional, Tuple, List, Callable, Iterator
from os import PathLike, path
import pickle
from functools import lru_cache
import json
from collections import defaultdict
from loguru import logger
import chromadb
from utils import timeit, iter_batches



class ChunkType: 
    

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
            logger.info(f"Loading index from {self.index_path}")
            with open(self.index_path, "rb") as f:
                self._index = pickle.load(f)  # type: ignore
            logger.info(f"Index loaded with {len(self._index)} entries.")
        else:
            logger.info(
                f"Building index from {self.jsonl_path} as no cached index "
                "is found."
            )
            self._index = self._build_index()
            logger.info(f"Index built with {len(self._index)} entries.")
            if self.index_path:
                with open(self.index_path, "wb") as f:
                    pickle.dump(self._index, f)
                logger.info(
                    f"Index saved to {self.index_path} with "
                    f"{len(self._index)} entries."
                )

        self._cached_contexts = lru_cache(maxsize=contexts_cache_size)(
            self._load_contexts
        )

    @timeit
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

    @timeit
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


class PersonaRAGMemoryLoader(BaseMemoryLoader):
    """Persona RAG Memory Loader"""

    def __init__(
        self,
        chroma_db_path: PathLike[str] | str,
        collection_name: str,
        jsonl_path: PathLike[str] | str,
        embedding_fn: Callable,
        split_into_chunks_fn: Callable,
        is_by_persona: bool = True,
        timestamp_min_gap: int = 10 * 60,
        timestamp_max_gap: int = 3 * 24 * 60 * 60,
        num_thread_workers: int = 8,
        max_batch_size: int = 10,
    ) -> None:

        self.jsonl_path = jsonl_path
        self.embedding_fn = embedding_fn
        self.split_into_chunks_fn = split_into_chunks_fn
        self.is_by_persona = is_by_persona
        self.timestamp_min_gap = timestamp_min_gap
        self.timestamp_max_gap = timestamp_max_gap
        self.num_thread_workers = num_thread_workers
        self.max_batch_size = max_batch_size

        # load all contexts from the jsonl file
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.shared_contexts = [json.loads(line) for line in f]

        # get persona to ids mapping and id to persona mapping
        self.persona_to_ids, self.id_to_persona = self._group_by_persona()

        chroma_db_path = path.abspath(chroma_db_path)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        try:
            self.collection = self.chroma_client.get_collection(
                collection_name
            )
        except ValueError:
            logger.warning(
                f"Collection '{collection_name}' not found. "
                "Creating a new collection."
            )
            self.collection = self.chroma_client.create_collection(
                name=collection_name
            )
            self._prepare_collection()

    def _group_by_persona(self) -> Tuple[dict[str, List[str]], dict[str, str]]:
        persona_to_ids = defaultdict(list)
        id_to_persona = {}
        for context in self.shared_contexts:
            assert len(context) == 1, (
                "Expected a single context ID" "in one shared contexts."
            )
            context_id, messages = next(iter(context.items()))
            persona_information = messages[0]["content"]
            persona_to_ids[persona_information].append(context_id)
            id_to_persona[context_id] = persona_information
        return dict(persona_to_ids), id_to_persona

    def _prepare_collection(self) -> None:
        """Prepare the ChromaDB collection by adding all contexts."""
        all_chunks = []
        for context in self.shared_contexts:
            context_id, messages = next(iter(context.items()))
            chunks = self.split_into_chunks_fn(messages)
            chunks_with_metadata = self._generate_metadata(chunks, context_id)
            all_chunks.extend(chunks_with_metadata)

        # batch generate embeddings
        chunks_with_embeddings = self._batch_embeddings(all_chunks)
        # embeddings = self._batch_embeddings(contents)
        if chunks_with_embeddings:
            pass


    @timeit
    def _batch_embeddings(
        self,
        contents: ,
    ) -> List[List[float]]:
        # Batch generate embeddings for the contents.
    
        
        
        for batch in self.split_into_batches(contents, self.max_batch_size):
            with ThreadPoolExecutor(
                max_workers=self.num_thread_workers
            ) as executor:
                embeddings = list(executor.map(self.embedding_fn, batch))
        
        

 
        
        
        

    def _generate_metadata(
        self,
        chunks: List[str],
        context_id: str,
    ) -> List[dict]:
        chunks_with_metadata = []
        start_time = datetime.now() - timedelta(days=365)
        current_time = start_time
        for chunk in chunks:
            chunk = {
                "content": chunk,
                "context_id": context_id,
                "timestamp": current_time.isoformat(),
                "persona": self.id_to_persona.get(context_id, ""),
            }
            gap = random.randint(
                self.timestamp_min_gap, self.timestamp_max_gap
            )
            current_time += timedelta(seconds=gap)
            chunks_with_metadata.append(chunk)
        return chunks_with_metadata


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
