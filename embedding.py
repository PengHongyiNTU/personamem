import getpass
from typing import List
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
import os


class EmbeddingModel:
    def __init__(
        self,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_len_per_query: int = 10,
    ):
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url
        self.max_len_per_query = max_len_per_query

        if "dashscope" in base_url:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        elif "openai" in base_url:
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(
                "Unsupported base URL. Use 'dashscope' or 'openai'."
            )

        if not self.api_key:
            print("No API key found in environment variables.")
            self.api_key = getpass.getpass(
                f"Please enter your API key for {self.base_url}: "
            )

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def __call__(
        self, texts: List[str], raw_response: bool = True
    ) -> CreateEmbeddingResponse:
        if len(texts) > self.max_len_per_query:
            raise ValueError(
                f"The model {self.model} can only process up to "
                f"{self.max_len_per_query} texts at a time."
            )
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
        )
        return response


if __name__ == "__main__":
    # Example usage
    from utils import timeit
    from dotenv import load_dotenv

    load_dotenv()
    embedding_model = EmbeddingModel(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    @timeit
    def example_usage():
        texts = [
            "欲把西湖比西子",
            "淡妆浓抹总相宜",
        ]
        response = embedding_model(texts)
        embeddings = [embedding.embedding for embedding in response.data]
        token_usage = response.usage
        print(len(embeddings), "embeddings generated.")
        print(len(embeddings[0]), "dimensions per embedding.")
        print(
            f"Prompt tokens: {token_usage.prompt_tokens}, "
            f"Total tokens: {token_usage.total_tokens}"
        )

    embeddings = example_usage()
