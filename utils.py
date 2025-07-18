import datasets
from typing import Literal, Tuple, Iterable
import os
import shutil
from huggingface_hub import hf_hub_download
from datasets import Dataset
import tiktoken
from loguru import logger
from functools import wraps
import time
import itertools


def get_datasets(
    split: Literal["32k", "128k", "1M"] = "32k",
    save_dir: str = "data/personamem",
) -> Tuple[Dataset, str]:
    """
    Load the PersonaMem dataset from Hugging Face and save it to a specified
    directory. Also downloads the shared contexts files if they do not exist.
    Args:
        split (Literal["32k", "128k", "1M"]): The split of the dataset to load.
        save_dir (str): The directory where the dataset will be saved.
                         Can be relative or absolute path.
    Returns:
        Tuple[datasets.Dataset, str]: The loaded dataset and the
        absolute path to the shared contexts file.
    """

    # Convert save_dir to absolute path if it's relative
    save_dir = os.path.abspath(save_dir)

    # check if the save directory exists, if not create it
    if not os.path.exists(save_dir):
        logger.info(f"Creating directory: {save_dir}")
        os.makedirs(save_dir)

    # Load the PersonaMem dataset
    personaMem_32k = datasets.load_dataset(
        "bowen-upenn/PersonaMem", split=split, cache_dir=save_dir
    )
    # assert it is a Dataset object
    assert isinstance(
        personaMem_32k, Dataset
    ), "Loaded dataset is not a Dataset object"

    # load the shared context file
    file_name = f"shared_contexts_{split}.jsonl"
    # Construct proper absolute file path
    file_path = os.path.join(save_dir, file_name)

    if not os.path.exists(file_path):
        # download the shared contexts file from Hugging Face Hub
        remote_file_path = hf_hub_download(
            repo_id="HongyiPeng/PersonaMem-Shared-Contexts",
            filename=file_name,
            repo_type="dataset",
        )
        logger.info(f"Downloading shared contexts file to {file_path}")
        # move the file to the save directory
        shutil.copy2(remote_file_path, file_path)
        logger.info(f"Shared contexts file saved to {file_path}")
    else:
        logger.info(f"Shared contexts file already exists at {file_path}")
    return personaMem_32k, file_path


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        logger.info(f"Function {func.__qualname__} took {elapsed:.4f} seconds")
        return result

    return wrapper


def iter_batches(iterable: Iterable, batch_size: int) -> Iterable:
    it = iter(iterable)
    while batch := list(itertools.islice(it, batch_size)):
        yield batch


if __name__ == "__main__":
    # Example with relative path
    dataset, shared_contexts_path = get_datasets("32k", "data/personamem")
    print(dataset[0])
    import json

    with open(shared_contexts_path, "r", encoding="utf-8") as f:
        for line in f:
            shared_context = json.loads(line)
            print(shared_context)
            break
