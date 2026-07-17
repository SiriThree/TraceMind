import re
from typing import TypedDict

from tracemind.clarifier import CandidateIntent
from tracemind.query_rewriter import build_enhanced_query
from tracemind.session_store import ConversationState


class ClarificationResolution(TypedDict):
    selected_intent: CandidateIntent | None
    enhanced_query: str
    clarification_round: int


TARGET_ALIASES: dict[str, list[str]] = {
    "吹风机": ["吹风机", "blower", "leaf blower"],
    "空调": ["空调", "air conditioner"],
    "洗碗机": ["洗碗机", "dishwasher"],
    "蒸汽清洁机": ["蒸汽清洁机", "steam cleaner"],
    "空气净化器": ["空气净化器", "air purifier"],
    "健身追踪器": ["健身追踪器", "fitness tracker"],
    "健身单车": ["健身单车", "exercise bike"],
    "电钻": ["电钻", "drill"],
    "烤箱": ["烤箱", "oven"],
    "冰箱": ["冰箱", "refrigerator", "fridge"],
    "发电机": ["发电机", "generator"],
    "摩托艇": ["摩托艇", "waverunner", "jet ski"],
    "水泵": ["水泵", "water pump", "pump"],
    "蓝牙激光鼠标": ["蓝牙激光鼠标", "bluetooth laser mouse", "mouse"],
    "VR头显": ["vr头显", "vr headset", "headset"],
}

TARGET_ALIASES.update(
    {
        "Air Fryer": ["air fryer", "airfryer"],
        "Digital SLR Camera": ["digital slr camera", "slr camera", "camera"],
        "Robot Vacuum": ["robot vacuum", "robot cleaner", "vacuum robot"],
        "Washing Machine": ["washing machine", "washer"],
        "Over-the-Range Microwave": ["over-the-range microwave", "microwave"],
        "Riding Lawn Mower": ["riding lawn mower", "lawn mower"],
    }
)


def _find_candidate_intent(
    candidate_intents: list[CandidateIntent],
    selected_intent_id: str | None,
) -> CandidateIntent | None:
    if not selected_intent_id:
        return None
    for item in candidate_intents:
        if item["id"] == selected_intent_id:
            return item
    return None


def _extract_product_name(text: str) -> str | None:
    lowered = text.lower()
    for canonical_name, aliases in TARGET_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if re.search(rf"(?<!\w){re.escape(alias_lower)}(?!\w)", lowered):
                return canonical_name
    return None


def _extract_feature_name(text: str) -> str | None:
    patterns = [
        r"(?:功能|按钮|页面|模式|开关)[：:\s]*([^\n，。,.]{2,24})",
        r"(?:feature|button|page|mode|switch)[:\s]+([^\n,.]{2,24})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def apply_clarification_followup(
    state: ConversationState,
    *,
    user_followup: str,
    selected_intent_id: str | None,
) -> ClarificationResolution:
    selected_intent = _find_candidate_intent(
        state["candidate_intents"],
        selected_intent_id,
    )
    if selected_intent is not None:
        state["selected_intent_id"] = selected_intent["id"]
        state["slots"][selected_intent["slot_key"]] = selected_intent["slot_value"]

    followup = user_followup.strip()
    if followup:
        product_name = _extract_product_name(followup)
        feature_name = _extract_feature_name(followup)
        if product_name is not None:
            state["slots"]["product_name"] = product_name
        if feature_name is not None:
            state["slots"]["feature_name"] = feature_name
        if not state["slots"]["error_symptom"]:
            state["slots"]["error_symptom"] = followup
        else:
            state["slots"]["error_symptom"] = followup
        state["history"].append({"role": "user", "content": followup})

    enhanced_query = build_enhanced_query(
        original_query=state["original_query"],
        selected_intent=selected_intent,
        user_followup=user_followup,
        slots=state["slots"],
    )
    state["current_query"] = enhanced_query
    state["clarification_round"] += 1

    return {
        "selected_intent": selected_intent,
        "enhanced_query": enhanced_query,
        "clarification_round": state["clarification_round"],
    }
