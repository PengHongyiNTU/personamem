from typing import List
import json
from utils import count_tokens


def split_into_chunks(
    self, messages: List[dict], max_tokens_per_chunk: int = 2048
) -> List[str]:
    chunks = []
    current_chunk = []
    current_tokens_count = 0
    for msg in messages:
        msg_text = json.dumps(msg, ensure_ascii=False)
        msg_tokens = count_tokens(msg_text)
        if msg_tokens > max_tokens_per_chunk:
            continue
        if current_tokens_count + msg_tokens <= max_tokens_per_chunk:
            current_chunk.append(msg_text)
            current_tokens_count += msg_tokens
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [msg_text]
            current_tokens_count = msg_tokens
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks
