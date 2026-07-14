import logging
import os

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def del_bank_by_handbook(source: str):
    """
    根据手册名称删除向量数据库中该手册的数据，如果某个手册有问题，只需要删除该手册的数据，再对该手册重新建库即可
    """
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
    )

    DEFAULT_MILVUS_CONNECTION = {
        "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
        "port": os.getenv("MILVUS_PORT", "19530"),
        "db_name": os.getenv("MILVUS_DB_NAME", "default"),
    }
    multi_analyzer_params = {
        "analyzers": {
            "english": {"type": "english"},
            "chinese": {"type": "chinese"},
            "default": {"tokenizer": "icu"},  # Required fallback analyzer
        },
        "by_field": "language",
    }
    collection_name = get_config()["MILVUS_COLLECTION_NAME"]
    milvus = Milvus(
        embedding_function=embedding_model,
        collection_name=collection_name,
        connection_args=DEFAULT_MILVUS_CONNECTION,
        auto_id=True,
        drop_old=False,
        enable_dynamic_field=False,
        vector_field=["dense", "sparse"],
        index_params=[
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 64,
                },
            },
            {
                "index_type": "AUTOINDEX",
                "metric_type": "BM25",
                "params": {},
            },
        ],
        builtin_function=BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="sparse",
            multi_analyzer_params=multi_analyzer_params,
        ),
    )
    res = milvus.client.delete(
        collection_name=collection_name,
        filter=f"source == '{source}'",
    )
    logger.info(f"{collection_name}集合中删除了{res['delete_count']}条{source}数据")


if __name__ == "__main__":
    del_bank_by_handbook("Rotary Blade Riding Lawn Mower Operator Handbook.txt")
