from tracemind.clarifier import CandidateIntent, detect_language_with_fallback
from tracemind.session_store import ConversationSlots


def build_enhanced_query(
    *,
    original_query: str,
    selected_intent: CandidateIntent | None,
    user_followup: str,
    slots: ConversationSlots,
) -> str:
    language = detect_language_with_fallback(f"{original_query}\n{user_followup}".strip())
    selected_label = selected_intent["label"] if selected_intent else ""

    if language == "chinese":
        parts = [f"原始问题：{original_query.strip()}"]
        if slots.get("product_name"):
            parts.append(f"产品对象：{slots['product_name']}")
        if slots.get("feature_name"):
            parts.append(f"功能或页面：{slots['feature_name']}")
        if selected_label:
            parts.append(f"澄清方向：{selected_label}")
        if slots.get("task_type"):
            parts.append(f"任务类型：{slots['task_type']}")
        if user_followup.strip():
            parts.append(f"补充说明：{user_followup.strip()}")
        return "\n".join(parts)

    parts = [f"Original question: {original_query.strip()}"]
    if slots.get("product_name"):
        parts.append(f"Product target: {slots['product_name']}")
    if slots.get("feature_name"):
        parts.append(f"Feature or page: {slots['feature_name']}")
    if selected_label:
        parts.append(f"Clarification direction: {selected_label}")
    if slots.get("task_type"):
        parts.append(f"Task type: {slots['task_type']}")
    if user_followup.strip():
        parts.append(f"Additional details: {user_followup.strip()}")
    return "\n".join(parts)
