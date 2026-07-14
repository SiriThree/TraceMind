import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

from config import get_config

load_dotenv()


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
        "default": {"tokenizer": "icu"},
    },
    "by_field": "language",
}


async def retriever(
    query: str,
    query_classification: dict[str, str],
    top_k: int = 10,
    use_source: bool = True,
) -> list[Document]:
    """根据query检索出相关的上下文"""
    milvus = Milvus(
        embedding_function=embedding_model,
        collection_name=get_config()["MILVUS_COLLECTION_NAME"],
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
            # analyzer_params={"type": "chinese"},
            # enable_match=True,
        ),
    )
    use_query_cls = get_config()["USE_QUERY_CLS"]
    language, llm_predict_source = (
        query_classification["language"],
        query_classification["source"],
    )
    expr = f"language == '{language}'"
    if llm_predict_source is not None and use_source and use_query_cls:
        expr = f"source == '{llm_predict_source}' AND language == '{language}'"
    # # 不使用查询分类
    # if not use_query_cls:
    #     expr = f"language == '{language}'"

    # 根据source和language进行文档的筛选，再混合检索
    results = await milvus.asimilarity_search(
        query,
        k=top_k,
        fetch_k=top_k,
        expr=expr,
        param=[
            {
                "metric_type": "COSINE",
            },
            {
                "metric_type": "BM25",
                "analyzer_name": language,
                "params": {},
            },
        ],
        ranker_type="rrf",
    )

    if llm_predict_source is not None and use_source and use_query_cls:
        # 根据index字段，对results进行排序，确保每个chunk的相对顺序和原文的顺序一致
        sorted_results = sorted(
            results,
            key=lambda x: x.metadata["index"],
        )
        return sorted_results
    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        retriever(
            "What should I pay attention to in order to ensure my safety when using this fax?",
            {"language": "english", "source": "Multi-Function Printer User Manual.txt"},
            top_k=19,
        )
    )
