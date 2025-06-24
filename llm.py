from typing import Optional
from openai import OpenAI 




class LLM:  


def init_llm(
    model: str = "qwen-plus",
    provider: str = "dashscope", 
    api_key: Optional[str] = None,
) 