from memory import BaseMemoryLoader
from loguru import logger
from typing import List


class Evaluation:
    def __init__(
        self,
        llm: str,
        dataset: List[dict],
        memory: BaseMemoryLoader,
        is_query_by_llm: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.llm = llm
        self.dataset = dataset
        self.is_query_by_llm = is_query_by_llm
        self.memory = memory
        self.batch_size = batch_size
        logger.info("Evaluation intialized")
