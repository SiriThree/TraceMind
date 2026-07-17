import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Literal

ENGLISH_STOPWORDS = {
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "there",
    "after",
    "before",
    "have",
    "has",
    "had",
    "will",
    "would",
    "should",
    "could",
    "does",
    "did",
    "doing",
    "into",
    "onto",
    "then",
    "than",
    "they",
    "them",
    "their",
    "your",
    "about",
    "power",
    "switch",
    "button",
    "start",
    "response",
    "working",
    "feature",
    "issue",
    "problem",
    "error",
    "fault",
    "cannot",
    "wont",
    "cant",
    "not",
    "and",
    "the",
}


def _catalog_path(language: Literal["chinese", "english"]) -> Path:
    if language == "chinese":
        return Path("catalog/chinese_handbook_catalog.json")
    return Path("catalog/english_handbook_catalog.json")


@lru_cache(maxsize=2)
def load_catalog_entries(language: Literal["chinese", "english"]) -> list[dict]:
    path = _catalog_path(language)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_terms(text: str, language: Literal["chinese", "english"]) -> set[str]:
    normalized = _normalize_text(text)
    if language == "english":
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", normalized)
            if token not in ENGLISH_STOPWORDS
        }

    cjk_terms: set[str] = set()
    for block in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        cjk_terms.add(block)
        if len(block) <= 4:
            cjk_terms.add(block)
        for idx in range(len(block) - 1):
            cjk_terms.add(block[idx : idx + 2])
    return cjk_terms


def _entry_text(entry: dict) -> str:
    return f"{entry.get('handbook_name', '')}\n{entry.get('catalog', '')}"


def assess_query_scope_signal(
    query: str,
    language: Literal["chinese", "english"],
) -> dict:
    normalized = _normalize_text(query)
    query_terms = _extract_terms(query, language)
    if language == "english":
        token_count = len(re.findall(r"[a-z0-9]{2,}", normalized))
        has_failure_signal = bool(
            re.search(
                r"\b(error|issue|problem|fail|failed|broken|not working|cannot|can't|won't|stuck|warning|no response|will not start|does not start)\b",
                normalized,
            )
        )
        has_product_signal = bool(
            re.search(
                r"\b(device|machine|dryer|fryer|oven|mower|camera|microwave|vacuum|printer|screen|switch|button|power|model)\b",
                normalized,
            )
        )
    else:
        token_count = len(query_terms)
        has_failure_signal = any(
            marker in query
            for marker in [
                "报错",
                "故障",
                "异常",
                "无法",
                "不能",
                "没反应",
                "失灵",
                "卡住",
                "报警",
                "失败",
            ]
        )
        has_product_signal = any(
            marker in query
            for marker in [
                "产品",
                "设备",
                "机器",
                "手册",
                "说明书",
                "型号",
                "页面",
                "功能",
                "按钮",
                "开关",
                "吹风机",
                "空气炸锅",
            ]
        )

    has_question_signal = "?" in query or "？" in query
    term_count = len(query_terms)
    analyzable = term_count >= 2 or has_failure_signal or has_product_signal
    allow_catalog_scope = analyzable
    allow_dense_scope = analyzable and (term_count >= 2 or has_failure_signal)

    if allow_dense_scope:
        reason = "query_contains_product_or_failure_signals"
    elif allow_catalog_scope:
        reason = "query_contains_partial_scope_signals"
    else:
        reason = "query_lacks_product_specific_signals"

    return {
        "term_count": term_count,
        "token_count": token_count,
        "has_question_signal": has_question_signal,
        "has_failure_signal": has_failure_signal,
        "has_product_signal": has_product_signal,
        "allow_catalog_scope": allow_catalog_scope,
        "allow_dense_scope": allow_dense_scope,
        "reason": reason,
    }


def recall_catalog_candidates(
    query: str,
    language: Literal["chinese", "english"],
    *,
    top_n: int = 5,
) -> list[dict]:
    query_terms = _extract_terms(query, language)
    entries = load_catalog_entries(language)
    scored: list[dict] = []
    for entry in entries:
        entry_text = _entry_text(entry)
        entry_terms = _extract_terms(entry_text, language)
        overlap = sorted(query_terms & entry_terms)
        score = len(overlap)
        if score <= 0:
            continue
        scored.append(
            {
                "source": entry["handbook_name"],
                "score": score,
                "matched_terms": overlap[:8],
                "catalog_preview": entry.get("catalog", "")[:220],
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_n]


def merge_candidate_sources(
    catalog_candidates: list[dict],
    dense_candidates: list[dict],
    *,
    top_n: int = 5,
) -> list[dict]:
    merged_scores: Counter[str] = Counter()
    detail_map: dict[str, dict] = {}

    for rank, item in enumerate(catalog_candidates, start=1):
        source = item["source"]
        merged_scores[source] += max(1, 8 - rank) + item.get("score", 0)
        detail_map.setdefault(source, {}).update(
            {
                "source": source,
                "catalog_score": item.get("score", 0),
                "matched_terms": item.get("matched_terms", []),
                "catalog_preview": item.get("catalog_preview", ""),
            }
        )

    for rank, item in enumerate(dense_candidates, start=1):
        source = item["source"]
        merged_scores[source] += max(1, 8 - rank) + item.get("hits", 0)
        detail_map.setdefault(source, {}).update(
            {
                "source": source,
                "dense_hits": item.get("hits", 0),
            }
        )

    merged = []
    for source, score in merged_scores.most_common(top_n):
        detail = detail_map.get(source, {"source": source})
        detail["merged_score"] = score
        merged.append(detail)
    return merged


def filter_candidate_sources(
    merged_candidates: list[dict],
    *,
    min_catalog_score: int = 1,
    min_dense_hits: int = 2,
) -> list[dict]:
    filtered: list[dict] = []
    for item in merged_candidates:
        catalog_score = int(item.get("catalog_score", 0) or 0)
        dense_hits = int(item.get("dense_hits", 0) or 0)
        if catalog_score >= min_catalog_score or dense_hits >= min_dense_hits:
            filtered.append(item)
    return filtered
