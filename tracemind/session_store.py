import time
from copy import deepcopy
from typing import Literal, TypedDict

from tracemind.clarifier import CandidateIntent

SESSION_TTL_SECONDS = 30 * 60
_STORE: dict[str, tuple[float, "ConversationState"]] = {}


class ConversationSlots(TypedDict):
    product_name: str | None
    product_model: str | None
    task_type: str | None
    page_name: str | None
    feature_name: str | None
    error_symptom: str | None
    error_message: str | None


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ConversationState(TypedDict):
    session_id: str
    status: Literal["clarification_pending", "clarification_resolved", "answered"]
    original_query: str
    current_query: str
    clarification_round: int
    missing_slots: list[str]
    candidate_intents: list[CandidateIntent]
    selected_intent_id: str | None
    clarification_reason: str | None
    trigger_signals: list[str]
    slots: ConversationSlots
    history: list[ConversationTurn]
    last_updated_at: float


def copy_state_for_debug(state: ConversationState) -> ConversationState:
    return deepcopy(state)


def _cleanup() -> None:
    now = time.time()
    expired_keys = [
        session_id
        for session_id, (expires_at, _state) in _STORE.items()
        if expires_at <= now
    ]
    for session_id in expired_keys:
        _STORE.pop(session_id, None)


def create_initial_state(
    session_id: str,
    query: str,
    candidate_intents: list[CandidateIntent],
    missing_slots: list[str],
    assistant_message: str,
    clarification_reason: str | None = None,
    trigger_signals: list[str] | None = None,
) -> ConversationState:
    now = time.time()
    return {
        "session_id": session_id,
        "status": "clarification_pending",
        "original_query": query,
        "current_query": query,
        "clarification_round": 1,
        "missing_slots": list(missing_slots),
        "candidate_intents": deepcopy(candidate_intents),
        "selected_intent_id": None,
        "clarification_reason": clarification_reason,
        "trigger_signals": list(trigger_signals or []),
        "slots": {
            "product_name": None,
            "product_model": None,
            "task_type": None,
            "page_name": None,
            "feature_name": None,
            "error_symptom": None,
            "error_message": None,
        },
        "history": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_message},
        ],
        "last_updated_at": now,
    }


def get_session(session_id: str) -> ConversationState | None:
    _cleanup()
    if session_id not in _STORE:
        return None
    _expires_at, state = _STORE[session_id]
    return deepcopy(state)


def save_session(state: ConversationState) -> None:
    state["last_updated_at"] = time.time()
    _STORE[state["session_id"]] = (
        time.time() + SESSION_TTL_SECONDS,
        deepcopy(state),
    )


def clear_session(session_id: str) -> None:
    _STORE.pop(session_id, None)
