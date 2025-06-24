from typing import Optional
from langchain.chat_models import init_chat_model
import dotenv 


def preprate_llm(
    model: str = "qwen3",
    model_provider: str = "openai",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    *args, 
    **kwargs
): 
    llm = init_chat_model()