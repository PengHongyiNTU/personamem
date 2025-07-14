from typing import List
from openai import OpenAI
import os 

def embedding_fns(
    texts: List[str],
    model: str = "text-embedding-v4",
    dimensions: int = 1024,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
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
            
    if "dashcope" in base_url:
        # Set the base URL for OpenAI client
        api_key = os.getenv("DASHSCOPE_API_KEY"),
    elif "openai" in base_url:
        # Set the base URL for OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("Unsupported base URL. Use 'dashscope' or 'openai'.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(
        input=texts, model=model, dimensions=dimensions
    )

    return [embedding["embedding"] for embedding in response["data"]]
