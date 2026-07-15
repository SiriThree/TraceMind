import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CHAT_MODEL = "qwen-plus"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _get_env(*names: str, default: str = "") -> str:
    return _first_non_empty(*(os.getenv(name) for name in names), default)


def create_chat_model(role_prefix: str, **kwargs: Any) -> ChatOpenAI:
    model = _get_env(f"{role_prefix}_MODEL", "CHAT_MODEL", default=DEFAULT_CHAT_MODEL)
    base_url = _get_env(
        f"{role_prefix}_BASE_URL",
        "CHAT_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "DEEPSEEK_BASE_URL",
        "OPEANAI_BASE_URL",
        default=DEFAULT_DASHSCOPE_BASE_URL,
    )
    api_key = _get_env(
        f"{role_prefix}_API_KEY",
        "CHAT_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPEANAI_API_KEY",
        default="placeholder-key",
    )
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


def create_embedding_model() -> OpenAIEmbeddings:
    model = _get_env("EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL)
    base_url = _get_env(
        "EMBEDDING_BASE_URL",
        "CHAT_BASE_URL",
        "DASHSCOPE_BASE_URL",
        default=DEFAULT_DASHSCOPE_BASE_URL,
    )
    api_key = _get_env(
        "EMBEDDING_API_KEY",
        "CHAT_API_KEY",
        "DASHSCOPE_API_KEY",
        default="placeholder-key",
    )
    return OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
