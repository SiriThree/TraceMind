import logging
import os
import re
from collections import Counter

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus

from tracemind.config import get_config
from tracemind.model_factory import create_embedding_model

load_dotenv()

logger = logging.getLogger(__name__)


embedding_model = create_embedding_model()
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


def _query_terms(query: str, language: str) -> set[str]:
    lowered = query.lower()
    if language == "english":
        return set(re.findall(r"[a-z0-9]{3,}", lowered))
    terms: set[str] = set()
    for block in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        for idx in range(len(block) - 1):
            terms.add(block[idx : idx + 2])
    return terms


def summarize_retrieval(results: list[Document]) -> dict:
    source_counter = Counter(
        result.metadata.get("source")
        for result in results
        if result.metadata.get("source")
    )
    top_snippets = []
    for result in results[:3]:
        top_snippets.append(
            {
                "source": result.metadata.get("source"),
                "index": result.metadata.get("index"),
                "snippet": result.page_content.replace("\n", " ")[:180],
            }
        )
    return {
        "hits": len(results),
        "unique_sources": len(source_counter),
        "top_sources": source_counter.most_common(5),
        "top_snippets": top_snippets,
    }


def _build_expr(
    *,
    language: str,
    selected_source: str | None,
    candidate_sources: list[str],
    use_source: bool,
    use_query_cls: bool,
) -> str:
    expr = f"language == '{language}'"
    if not use_source or not use_query_cls:
        return expr

    if selected_source:
        return f"source == '{selected_source}' AND language == '{language}'"

    if candidate_sources:
        source_expr = " OR ".join([f"source == '{source}'" for source in candidate_sources])
        return f"({source_expr}) AND language == '{language}'"

    return expr


def _apply_lightweight_rerank(
    results: list[Document],
    *,
    query: str,
    language: str,
    selected_source: str | None,
    candidate_sources: list[str],
) -> list[Document]:
    terms = _query_terms(query, language)

    def score(doc: Document) -> tuple[int, int]:
        doc_source = doc.metadata.get("source")
        doc_text = doc.page_content.lower()
        overlap = sum(1 for term in terms if term in doc_text)
        source_bonus = 0
        if selected_source and doc_source == selected_source:
            source_bonus += 5
        elif doc_source in candidate_sources:
            source_bonus += 2
        return source_bonus + overlap, -int(doc.metadata.get("index", 0))

    return sorted(results, key=score, reverse=True)


def _append_retrieval_debug(
    query_classification: dict,
    *,
    stage: str,
    expr: str,
    summary: dict,
    selected_source: str | None,
    candidate_sources: list[str],
    rerank_applied: bool,
) -> None:
    debug_event = {
        "stage": stage,
        "expr": expr,
        "selected_source": selected_source,
        "candidate_sources": candidate_sources,
        "hits": summary["hits"],
        "unique_sources": summary["unique_sources"],
        "top_sources": summary["top_sources"],
        "top_snippets": summary["top_snippets"],
        "rerank_applied": rerank_applied,
        "rerank_strategy": "lightweight_rule" if rerank_applied else "none",
        "fallback_triggered": False,
    }
    query_classification.setdefault("_retrieval_debug", []).append(debug_event)


async def retriever(
    query: str,
    query_classification: dict[str, str],
    top_k: int = 10,
    use_source: bool = True,
    stage: str = "primary",
) -> list[Document]:
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
                "params": {"M": 16, "efConstruction": 64},
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

    use_query_cls = get_config()["USE_QUERY_CLS"]
    language = query_classification["language"]
    selected_source = query_classification.get("source")
    candidate_sources = list(query_classification.get("candidate_sources", []) or [])
    expr = _build_expr(
        language=language,
        selected_source=selected_source,
        candidate_sources=candidate_sources,
        use_source=use_source,
        use_query_cls=use_query_cls,
    )

    logger.info(
        "retriever:start query=%r language=%s source=%s candidates=%s use_source=%s use_query_cls=%s top_k=%s stage=%s expr=%s",
        query,
        language,
        selected_source,
        candidate_sources[:5],
        use_source,
        use_query_cls,
        top_k,
        stage,
        expr,
    )
    results = await milvus.asimilarity_search(
        query,
        k=top_k,
        fetch_k=top_k,
        expr=expr,
        param=[
            {"metric_type": "COSINE"},
            {
                "metric_type": "BM25",
                "analyzer_name": language,
                "params": {},
            },
        ],
        ranker_type="rrf",
    )

    rerank_applied = bool(get_config()["ENABLE_LIGHTWEIGHT_RERANK"])
    ranked_results = results
    if rerank_applied:
        ranked_results = _apply_lightweight_rerank(
            results,
            query=query,
            language=language,
            selected_source=selected_source if use_source else None,
            candidate_sources=candidate_sources if use_source else [],
        )

    summary = summarize_retrieval(ranked_results)
    returned_results = ranked_results
    if selected_source and use_source and use_query_cls:
        returned_results = sorted(ranked_results, key=lambda x: x.metadata.get("index", 0))

    _append_retrieval_debug(
        query_classification,
        stage=stage,
        expr=expr,
        summary=summary,
        selected_source=selected_source if use_source else None,
        candidate_sources=candidate_sources if use_source else [],
        rerank_applied=rerank_applied,
    )

    logger.info("retriever:done hits=%s unique_sources=%s", summary["hits"], summary["unique_sources"])
    for idx, item in enumerate(summary["top_snippets"], start=1):
        logger.info(
            "retriever:hit rank=%s source=%s index=%s snippet=%r",
            idx,
            item["source"],
            item["index"],
            item["snippet"],
        )
    return returned_results
