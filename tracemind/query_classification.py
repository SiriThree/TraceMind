import ast
import json
import logging
import os
from collections import Counter
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus

from tracemind.config import get_config
from tracemind.model_factory import create_chat_model, create_embedding_model
from tracemind.routing_scope import (
    assess_query_scope_signal,
    filter_candidate_sources,
    merge_candidate_sources,
    recall_catalog_candidates,
)
from tracemind.utils import language_detect

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embedding_model = create_embedding_model()
query_classification_model = create_chat_model("CLASSIFIER_LLM")

DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "db_name": os.getenv("MILVUS_DB_NAME", "default"),
}


def parse_jsonish_dict(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict output, got: {type(parsed)!r}")
        return parsed


def _classify_query_prompt(language: Literal["chinese", "english"]) -> str:
    if language == "chinese":
        return """你是客服查询分流助手。请判断用户问题更适合走哪一类链路。

任务：
1. 如果问题明显是售前售后、退款、投诉、物流、发票、订单、人工客服等通用客服问题，则归类为 general。
2. 如果问题明显依赖某个产品手册、设备说明、功能操作、故障排查、页面设置或部件说明，则归类为 product。
3. 如果问题偏模糊，但核心仍然围绕某个设备、功能、故障或使用方法，优先归类为 product。

输出 JSON：
{
  "question_type": "product" | "general",
  "question_confidence": float,
  "reason": str
}

用户问题：
{{query}}
"""
    return """You are a support routing assistant. Decide which chain should handle the user query.

Rules:
1. Use "general" for order, refund, invoice, complaint, delivery, human support, or other generic support topics.
2. Use "product" for product-manual questions, device operation, troubleshooting, feature guidance, settings, or component explanations.
3. If the query is vague but still clearly about a device, feature, failure, or usage method, prefer "product".

Output JSON:
{
  "question_type": "product" | "general",
  "question_confidence": float,
  "reason": str
}

User query:
{{query}}
"""


def _select_source_prompt(
    language: Literal["chinese", "english"],
    candidate_entries: list[dict],
) -> str:
    if language == "chinese":
        return (
            """你是产品知识路由助手。请在候选手册中选择最可能回答用户问题的手册。

要求：
1. 只能在给定候选集中选择 source。
2. 如果你认为这些候选都不可靠，可以返回 null。
3. 优先依据产品类型、功能范围、故障现象、目录主题来判断。

候选手册：
"""
            + json.dumps(candidate_entries, ensure_ascii=False, indent=2)
            + """

输出 JSON：
{
  "source": str | null,
  "source_confidence": float | null,
  "reason": str
}

用户问题：
{{query}}
"""
        )
    return (
        """You are a product knowledge router. Choose the most likely manual from the candidate set.

Rules:
1. You must choose only from the given candidates.
2. If none of them looks reliable, return null.
3. Prefer evidence from product type, feature scope, failure symptoms, and catalog topics.

Candidate manuals:
"""
        + json.dumps(candidate_entries, ensure_ascii=False, indent=2)
        + """

Output JSON:
{
  "source": str | null,
  "source_confidence": float | null,
  "reason": str
}

User query:
{{query}}
"""
    )


async def classify_query_type(query: str) -> dict:
    language = language_detect(query)
    prompt = PromptTemplate.from_template(
        _classify_query_prompt(language),
        template_format="mustache",
    )
    chain = prompt | query_classification_model | (lambda msg: parse_jsonish_dict(msg.content))
    result = await chain.with_retry(stop_after_attempt=5).ainvoke({"query": query})
    return {"language": language, **result}


async def get_source_candidates_by_dense_store(
    query: str,
    language: Literal["chinese", "english"],
    *,
    top_k: int = 12,
) -> list[dict]:
    dense_store = Milvus(
        embedding_function=embedding_model,
        collection_name=get_config()["MILVUS_COLLECTION_NAME"],
        text_field="text",
        vector_field="dense",
        auto_id=True,
        drop_old=False,
        enable_dynamic_field=True,
        connection_args=DEFAULT_MILVUS_CONNECTION,
        index_params=[{"index_type": "HNSW", "metric_type": "COSINE"}],
    )
    dense_result = await dense_store.asimilarity_search(
        query,
        k=top_k,
        fetch_k=top_k,
        expr=f"language == '{language}'",
    )
    source_counter = Counter(
        result.metadata.get("source")
        for result in dense_result
        if result.metadata.get("source")
    )
    return [
        {"source": source, "hits": hits}
        for source, hits in source_counter.most_common(5)
    ]


async def select_source_from_candidates(
    query: str,
    language: Literal["chinese", "english"],
    candidate_sources: list[dict],
) -> dict:
    candidate_entries = [
        {
            "source": item["source"],
            "catalog_preview": item.get("catalog_preview", ""),
            "matched_terms": item.get("matched_terms", []),
            "catalog_score": item.get("catalog_score", 0),
            "dense_hits": item.get("dense_hits", 0),
        }
        for item in candidate_sources
    ]
    prompt = PromptTemplate.from_template(
        _select_source_prompt(language, candidate_entries),
        template_format="mustache",
    )
    chain = prompt | query_classification_model | (lambda msg: parse_jsonish_dict(msg.content))
    result = await chain.with_retry(stop_after_attempt=5).ainvoke({"query": query})
    return result


async def ensembles_query_classification(
    query: str,
    source_hint: str | None = None,
) -> dict[str, str | bool | list | dict | None]:
    language = language_detect(query)
    if source_hint is not None:
        return {
            "source": source_hint,
            "question_type": "product",
            "language": language,
            "source_hint_applied": True,
            "candidate_sources": [source_hint],
            "route_debug": {
                "query_type_reason": "source_hint_locked",
                "catalog_candidates": [],
                "dense_candidates": [],
                "candidate_sources": [source_hint],
                "selected_source_reason": "source_hint_locked",
            },
        }

    query_type_result = await classify_query_type(query)
    question_type = query_type_result["question_type"]
    question_confidence = query_type_result.get("question_confidence")
    scope_signal = assess_query_scope_signal(query, language)

    if question_type == "general" and question_confidence is not None and question_confidence >= 0.96:
        return {
            "source": None,
            "question_type": "general",
            "language": language,
            "source_hint_applied": False,
            "candidate_sources": [],
            "route_debug": {
                "query_type_reason": query_type_result.get("reason", ""),
                "scope_signal": scope_signal,
                "catalog_candidates": [],
                "dense_candidates": [],
                "candidate_sources": [],
                "selected_source_reason": "high_confidence_general",
            },
        }

    catalog_candidates = []
    dense_candidates = []
    if scope_signal["allow_catalog_scope"]:
        catalog_candidates = recall_catalog_candidates(query, language, top_n=5)
    if scope_signal["allow_dense_scope"]:
        dense_candidates = await get_source_candidates_by_dense_store(query, language)

    merged_candidates = merge_candidate_sources(catalog_candidates, dense_candidates, top_n=5)
    merged_candidates = filter_candidate_sources(merged_candidates)
    candidate_sources = [item["source"] for item in merged_candidates]

    selected_source = None
    selected_reason = "no_candidate_scope"
    source_confidence = None

    if candidate_sources:
        if len(candidate_sources) == 1:
            selected_source = candidate_sources[0]
            source_confidence = 0.9
            selected_reason = "single_candidate_scope"
        else:
            source_choice = await select_source_from_candidates(query, language, merged_candidates)
            selected_source = source_choice.get("source")
            source_confidence = source_choice.get("source_confidence")
            selected_reason = source_choice.get("reason", "")

    product_scope_reliable = bool(selected_source) or bool(catalog_candidates)
    if not product_scope_reliable and question_type == "general":
        candidate_sources = []
        merged_candidates = []
        selected_reason = "general_without_reliable_scope_evidence"

    final_question_type = "product" if selected_source or candidate_sources else question_type

    return {
        "source": selected_source,
        "question_type": final_question_type,
        "language": language,
        "source_hint_applied": False,
        "candidate_sources": candidate_sources,
        "source_confidence": source_confidence,
        "route_debug": {
            "query_type_reason": query_type_result.get("reason", ""),
            "query_type_confidence": question_confidence,
            "scope_signal": scope_signal,
            "catalog_candidates": catalog_candidates,
            "dense_candidates": dense_candidates,
            "merged_candidates": merged_candidates,
            "candidate_sources": candidate_sources,
            "selected_source_reason": selected_reason,
        },
    }
