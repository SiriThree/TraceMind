import re
from typing import Literal, TypedDict

from tracemind.utils import language_detect


class CandidateIntent(TypedDict):
    id: str
    label: str
    slot_key: str
    slot_value: str


class ClarificationResult(TypedDict):
    need_clarification: bool
    reason: str
    clarifying_question: str
    candidate_intents: list[CandidateIntent]
    missing_slots: list[str]
    trigger_signals: list[str]


CHINESE_VAGUE_PATTERNS = [
    r"这个.*(不好用|不行|有问题|报错|失败|卡住|办不了)",
    r"这个功能.*(怎么|不好用|有问题)",
    r"这个页面.*(怎么|报错|有问题|进不去)",
    r"(怎么不好用|怎么不行|怎么失败了)",
    r"(办不了|弄不了|搞不定|不会用)",
    r"帮我看看",
    r"你看下",
]

ENGLISH_VAGUE_PATTERNS = [
    r"\bthis\b.*\b(not working|broken|error|issue|failed|problem)\b",
    r"\bthis feature\b.*\b(how|issue|not working)\b",
    r"\bthis page\b.*\b(error|issue|can't open|not working)\b",
    r"\bhelp me check\b",
    r"\bcan you check\b",
]

GENERIC_SUPPORT_INTENTS_ZH: list[CandidateIntent] = [
    {
        "id": "task_installation",
        "label": "安装/连接",
        "slot_key": "task_type",
        "slot_value": "installation",
    },
    {
        "id": "task_feature_usage",
        "label": "功能使用",
        "slot_key": "task_type",
        "slot_value": "feature_usage",
    },
    {
        "id": "task_fault",
        "label": "故障报错",
        "slot_key": "task_type",
        "slot_value": "fault",
    },
    {
        "id": "task_after_sales",
        "label": "售后处理",
        "slot_key": "task_type",
        "slot_value": "after_sales",
    },
]

GENERIC_SUPPORT_INTENTS_EN: list[CandidateIntent] = [
    {
        "id": "task_installation",
        "label": "Setup",
        "slot_key": "task_type",
        "slot_value": "installation",
    },
    {
        "id": "task_feature_usage",
        "label": "Feature usage",
        "slot_key": "task_type",
        "slot_value": "feature_usage",
    },
    {
        "id": "task_fault",
        "label": "Fault or error",
        "slot_key": "task_type",
        "slot_value": "fault",
    },
    {
        "id": "task_after_sales",
        "label": "After-sales",
        "slot_key": "task_type",
        "slot_value": "after_sales",
    },
]

ORDER_INTENTS_ZH: list[CandidateIntent] = [
    {
        "id": "order_refund",
        "label": "退款",
        "slot_key": "task_type",
        "slot_value": "refund",
    },
    {
        "id": "order_exchange",
        "label": "换货",
        "slot_key": "task_type",
        "slot_value": "exchange",
    },
    {
        "id": "order_repair",
        "label": "维修",
        "slot_key": "task_type",
        "slot_value": "repair",
    },
    {
        "id": "order_complaint",
        "label": "投诉",
        "slot_key": "task_type",
        "slot_value": "complaint",
    },
]

ORDER_INTENTS_EN: list[CandidateIntent] = [
    {
        "id": "order_refund",
        "label": "Refund",
        "slot_key": "task_type",
        "slot_value": "refund",
    },
    {
        "id": "order_exchange",
        "label": "Exchange",
        "slot_key": "task_type",
        "slot_value": "exchange",
    },
    {
        "id": "order_repair",
        "label": "Repair",
        "slot_key": "task_type",
        "slot_value": "repair",
    },
    {
        "id": "order_complaint",
        "label": "Complaint",
        "slot_key": "task_type",
        "slot_value": "complaint",
    },
]


def detect_language_with_fallback(query: str) -> Literal["chinese", "english"]:
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
        vague_tokens = ["这个", "这个功能", "这个页面", "不好用", "不行", "报错", "有问题"]
        return len(stripped_query) <= 16 and any(
            token in stripped_query for token in vague_tokens
        )

    vague_tokens = ["this", "issue", "error", "problem", "not working"]
    return len(stripped_query.split()) <= 8 and any(
        token in lowered_query for token in vague_tokens
    )


def _infer_missing_slots(query: str, language: Literal["chinese", "english"]) -> list[str]:
    lowered_query = query.lower()
    missing_slots: list[str] = []

    if language == "chinese":
        generic_target = any(
            token in query for token in ["这个产品", "这个型号", "这个页面", "这个功能"]
        )
        has_target = any(
            token in query
            for token in [
                "产品",
                "型号",
                "页面",
                "功能",
                "吹风机",
                "空调",
                "洗碗机",
                "蒸汽清洁机",
                "空气净化器",
            ]
        ) and not generic_target
        has_symptom = any(
            token in query
            for token in [
                "报错",
                "失败",
                "没反应",
                "没有反应",
                "打不开",
                "进不去",
                "不好用",
                "不行",
                "启动不了",
                "无法启动",
                "开不了",
            ]
        )
        has_task_type = any(
            token in query
            for token in ["退款", "换货", "维修", "投诉", "设置", "安装", "连接", "启动", "故障", "使用"]
        ) or any(
            token in lowered_query for token in ["fault", "repair", "refund", "exchange"]
        )
    else:
        generic_target = any(
            token in lowered_query
            for token in ["this product", "this model", "this page", "this feature"]
        )
        has_target = any(
            token in lowered_query for token in ["product", "model", "page", "feature", "screen"]
        ) and not generic_target
        has_symptom = any(
            token in lowered_query
            for token in ["error", "failed", "no response", "not working", "can't open", "problem"]
        )
        has_task_type = any(
            token in lowered_query
            for token in ["refund", "exchange", "repair", "complaint", "setup", "install", "connect", "fault"]
        )

    if not has_target:
        missing_slots.append("product_or_target")
    if not has_task_type:
        missing_slots.append("task_type")
    if not has_symptom:
        missing_slots.append("error_symptom")

    return missing_slots


def _build_candidate_intents(
    query: str, language: Literal["chinese", "english"]
) -> list[CandidateIntent]:
    lowered_query = query.lower()

    if language == "chinese":
        if any(token in query for token in ["订单", "退款", "换货", "维修", "投诉", "售后"]):
            return ORDER_INTENTS_ZH
        if any(token in query for token in ["页面", "报错", "功能", "不好用", "不行"]):
            return GENERIC_SUPPORT_INTENTS_ZH
        return GENERIC_SUPPORT_INTENTS_ZH[:3]

    if any(
        token in lowered_query
        for token in ["order", "refund", "exchange", "repair", "complaint", "after-sales"]
    ):
        return ORDER_INTENTS_EN
    return GENERIC_SUPPORT_INTENTS_EN[:3]


def _build_clarifying_question(
    query: str,
    language: Literal["chinese", "english"],
    missing_slots: list[str],
) -> str:
    if language == "chinese":
        if "task_type" in missing_slots:
            return "你想咨询的是哪一类问题？我先帮你把方向收窄一下。"
        if "error_symptom" in missing_slots:
            return "可以补充一下具体出现了什么现象或报错吗？"
        return "可以再补充一下是哪个产品、页面或功能，以及你当前遇到的现象吗？"

    if "task_type" in missing_slots:
        return "Which kind of issue are you dealing with? I can narrow it down first."
    if "error_symptom" in missing_slots:
        return "Could you share the exact symptom or error you are seeing?"
    return "Could you tell me which product, page, or feature you mean and what exactly is happening?"


def _fallback_result(language: Literal["chinese", "english"]) -> ClarificationResult:
    candidate_intents = (
        GENERIC_SUPPORT_INTENTS_ZH[:3]
        if language == "chinese"
        else GENERIC_SUPPORT_INTENTS_EN[:3]
    )
    question = (
        "可以再补充一下是哪个产品、页面或功能，以及你当前遇到的现象吗？"
        if language == "chinese"
        else "Could you share which product, page, or feature you mean and what you are seeing?"
    )
    reason = (
        "用户问题较模糊，缺少产品对象、任务类型或具体现象。"
        if language == "chinese"
        else "The question is still too vague and lacks the target, task type, or symptom."
    )
    return {
        "need_clarification": True,
        "reason": reason,
        "clarifying_question": question,
        "candidate_intents": candidate_intents,
        "missing_slots": ["product_or_target", "task_type", "error_symptom"],
        "trigger_signals": ["fallback"],
    }


async def clarify_query(query: str) -> ClarificationResult:
    language = detect_language_with_fallback(query)
    missing_slots = _infer_missing_slots(query, language)
    if len(missing_slots) == 0 and len(query.strip()) >= 12:
        return {
            "need_clarification": False,
            "reason": "",
            "clarifying_question": "",
            "candidate_intents": [],
            "missing_slots": [],
            "trigger_signals": [],
        }

    vague_signal = _has_vague_signal(query, language)
    if not vague_signal and len(missing_slots) <= 1 and len(query.strip()) >= 12:
        return {
            "need_clarification": False,
            "reason": "",
            "clarifying_question": "",
            "candidate_intents": [],
            "missing_slots": [],
            "trigger_signals": [],
        }

    if not vague_signal:
        return {
            "need_clarification": False,
            "reason": "",
            "clarifying_question": "",
            "candidate_intents": [],
            "missing_slots": [],
            "trigger_signals": [],
        }

    candidate_intents = _build_candidate_intents(query, language)
    clarifying_question = _build_clarifying_question(query, language, missing_slots)

    if not clarifying_question or not candidate_intents:
        return _fallback_result(language)

    return {
        "need_clarification": True,
        "reason": (
            "用户问题仍然偏模糊，直接检索和回答的命中范围过大。"
            if language == "chinese"
            else "The question is still vague enough that retrieval would be too broad."
        ),
        "clarifying_question": clarifying_question,
        "candidate_intents": candidate_intents,
        "missing_slots": missing_slots,
        "trigger_signals": ["vague_pattern", "missing_slots"],
    }


def format_clarification_message(result: ClarificationResult) -> str:
    if not result["need_clarification"]:
        return ""

    answer = result["clarifying_question"]
    if result["candidate_intents"]:
        language = detect_language_with_fallback(answer)
        intro = (
            "你可以直接点击一个方向，或者补充更具体的说明："
            if language == "chinese"
            else "You can pick one direction below or add more details:"
        )
        options = "\n".join(
            f"{index}. {item['label']}"
            for index, item in enumerate(result["candidate_intents"], start=1)
        )
        answer += f"\n\n{intro}\n{options}"
    return answer
