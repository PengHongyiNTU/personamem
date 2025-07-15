import getpass
from typing import List
from openai import OpenAI
import os


def embedding_fns(
    texts: List[str],
    model: str = "text-embedding-v4",
    dimensions: int = 1024,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using a specified model.

    Args:
        texts (List[str]): List of texts to embed.
        model (str): The model to use for embedding.
            Default is "text-embedding-v4".
        dimensions (int): The number of dimensions for the embeddings.
            Default is 1024.

    Returns:
        List[List[float]]: A list of embeddings, each represented as a list of
            floats.
    """

    if model == "text-embedding-v4":
        if len(texts) > 10:
            raise ValueError(
                "The text-embedding-v4 model can only process up to 10 texts"
                "at a time."
            )

    if "dashscope" in base_url:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    elif "openai" in base_url:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("Unsupported base URL. Use 'dashscope' or 'openai'.")
    if not api_key:
        print("No API key found in environment variables.")
        api_key = getpass.getpass("Please enter your API key: ")
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(
        input=texts, model=model, dimensions=dimensions
    )
    print(response.usage)
    return [data.embedding for data in response.data]


if __name__ == "__main__":
    # Example usage
    from utils import timeit
    from dotenv import load_dotenv

    load_dotenv()

    @timeit
    def example_usage():
        texts = [
            "欲把西湖比西子",
            "淡妆浓抹总相宜",
        ]
        embeddings = embedding_fns(texts)
        return embeddings

    embeddings = example_usage()
    print(embeddings)
