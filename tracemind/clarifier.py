import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from tracemind.model_factory import create_chat_model
from tracemind.utils import language_detect

load_dotenv()


class ClarificationResult(TypedDict):
    need_clarification: bool
    reason: str
    clarifying_question: str
    candidate_intents: list[str]


clarifier_model = create_chat_model("CLARIFIER_LLM")


CHINESE_VAGUE_PATTERNS = [
    r"这个.*(不好用|不行|有问题|报错|失败|卡住|办不了)",
    r"这个功能.*(怎么|不好用|有问题)",
    r"这个页面.*(怎么|报错|有问题|进不去)",
    r"(怎么不好用|怎么不行|怎么失败了)",
    r"(办不了|弄不了|搞不定|不会办)",
    r"帮我看看",
    r"你看下",
]

ENGLISH_VAGUE_PATTERNS = [
    r"\bthis\b.*\b(not working|broken|error|issue|failed)\b",
    r"\bthis feature\b.*\b(how|issue|not working)\b",
    r"\bthis page\b.*\b(error|issue|can't open|not working)\b",
    r"\bhelp me check\b",
    r"\bcan you check\b",
]


def _detect_language_with_fallback(query: str) -> Literal["chinese", "english"]:
    try:
        return language_detect(query)
    except Exception:
        if re.search(r"[\u4e00-\u9fff]", query):
            return "chinese"
        return "english"


def _has_vague_signal(query: str, language: Literal["chinese", "english"]) -> bool:
    patterns = (
        CHINESE_VAGUE_PATTERNS if language == "chinese" else ENGLISH_VAGUE_PATTERNS
    )
    lowered_query = query.lower()
    if any(re.search(pattern, lowered_query) for pattern in patterns):
        return True

    stripped_query = query.strip()
    if language == "chinese":
        return len(stripped_query) <= 12 and any(
            token in stripped_query
            for token in ["这个", "这个功能", "这个页面", "不好用", "不行", "报错", "有问题"]
        )
    return len(stripped_query.split()) <= 6 and any(
        token in lowered_query
        for token in ["this", "issue", "error", "not working", "problem"]
    )


def _get_clarifier_prompt(language: Literal["chinese", "english"]) -> str:
    if language == "chinese":
        return """# Role
你是产品客服 Agent 的问题澄清助手。你的任务不是直接回答问题，而是判断用户当前问题是否信息不足，是否需要先追问。

# Task
请根据用户问题，判断是否需要澄清。

如果用户已经明确给出了产品、页面、功能、目标操作、报错现象中的大部分关键信息，则不需要澄清。
如果用户的问题仍然模糊，例如只说“这个功能不好用”“这个页面报错”“这个业务办不了”“帮我看看”，则需要澄清。

如果需要澄清：
1. 给出一句自然、客服风格的追问。
2. 给出 2 到 4 个简短的候选补充方向，帮助用户快速补充信息。

# Output Format
请严格输出 JSON：
{
  "need_clarification": bool,
  "reason": str,
  "clarifying_question": str,
  "candidate_intents": list[str]
}

# User Query
{{query}}
"""
    return """# Role
You are a clarification assistant for a customer support agent. Your job is not to answer directly. Your job is to decide whether the user's current question is still too vague and needs a follow-up question first.

# Task
If the user already provided enough detail about the product, page, feature, goal, or error symptom, set need_clarification to false.
If the user is still vague, such as saying "this feature is not working", "this page has an error", or "help me check this", set need_clarification to true.

If clarification is needed:
1. Write one short natural follow-up question.
2. Provide 2 to 4 short candidate directions the user can choose from.

# Output Format
Return strict JSON:
{
  "need_clarification": bool,
  "reason": str,
  "clarifying_question": str,
  "candidate_intents": list[str]
}

# User Query
{{query}}
"""


def _fallback_result(language: Literal["chinese", "english"]) -> ClarificationResult:
    if language == "chinese":
        return {
            "need_clarification": True,
            "reason": "用户问题过于模糊，缺少产品、页面或报错现象等关键信息。",
            "clarifying_question": "可以再补充一下具体是哪个产品、页面或功能，以及你当前遇到的现象吗？",
            "candidate_intents": [
                "具体是哪个产品或型号",
                "你想完成什么操作",
                "出现了什么报错或异常现象",
            ],
        }
    return {
        "need_clarification": True,
        "reason": "The question is still too vague and lacks the product, page, or error details.",
        "clarifying_question": "Could you share which product, page, or feature you mean, and what exactly is happening?",
        "candidate_intents": [
            "Which product or model",
            "What action you want to complete",
            "What error or abnormal behavior you saw",
        ],
    }


async def clarify_query(query: str) -> ClarificationResult:
    language = _detect_language_with_fallback(query)
    if not _has_vague_signal(query, language):
        return {
            "need_clarification": False,
            "reason": "",
            "clarifying_question": "",
            "candidate_intents": [],
        }

    prompt = PromptTemplate.from_template(
        _get_clarifier_prompt(language), template_format="mustache"
    )
    chain = prompt | clarifier_model | JsonOutputParser()

    try:
        result = await chain.with_retry(stop_after_attempt=3).ainvoke({"query": query})
    except Exception:
        return _fallback_result(language)

    need_clarification = bool(result.get("need_clarification"))
    if not need_clarification:
        return {
            "need_clarification": False,
            "reason": str(result.get("reason", "")),
            "clarifying_question": "",
            "candidate_intents": [],
        }

    clarifying_question = str(result.get("clarifying_question", "")).strip()
    candidate_intents = [
        str(item).strip()
        for item in result.get("candidate_intents", [])
        if str(item).strip()
    ][:4]

    if not clarifying_question:
        return _fallback_result(language)

    return {
        "need_clarification": True,
        "reason": str(result.get("reason", "")).strip(),
        "clarifying_question": clarifying_question,
        "candidate_intents": candidate_intents,
    }


def format_clarification_message(result: ClarificationResult) -> str:
    if not result["need_clarification"]:
        return ""

    answer = result["clarifying_question"]
    if result["candidate_intents"]:
        language = language_detect(answer)
        intro = (
            "你可以补充这些信息中的任意一项："
            if language == "chinese"
            else "You can add any of the following details:"
        )
        options = "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(result["candidate_intents"], start=1)
        )
        answer += f"\n\n{intro}\n{options}"
    return answer
