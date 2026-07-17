import json
import logging
from collections.abc import AsyncGenerator
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from tracemind.answer_general_query import answer_general_query
from tracemind.answer_product_query import (
    ProductClarificationNeeded,
    answer_product_query,
)
from tracemind.clarification_resolver import apply_clarification_followup
from tracemind.clarifier import CandidateIntent, clarify_query, format_clarification_message
from tracemind.query_classification import ensembles_query_classification
from tracemind.source_hints import resolve_source_hint
from tracemind.session_store import (
    clear_session,
    copy_state_for_debug,
    create_initial_state,
    get_session,
    save_session,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClarificationFollowupPayload(TypedDict, total=False):
    is_followup: bool
    selected_intent_id: str | None


class PipelineResult(TypedDict):
    answer: str
    response_type: Literal["answer", "clarification"]
    clarification_state: Literal["none", "pending", "resolved"]
    candidate_intents: list[CandidateIntent]
    missing_slots: list[str]
    effective_query: str
    session_debug: dict


def _is_generic_request_for_more_details(answer: str) -> bool:
    lowered = answer.lower()
    actionable_markers = [
        "1.",
        "2.",
        "step 1",
        "step 2",
        "<pic",
        "检查",
        "步骤",
        "确认",
        "ensure",
        "check whether",
    ]
    if sum(marker in lowered or marker in answer for marker in actionable_markers) >= 2:
        return False

    patterns = [
        "please provide",
        "could you provide",
        "please share",
        "exact error message",
        "specific error message",
        "more details",
        "more information",
        "请提供",
        "请补充",
        "更多信息",
        "更多细节",
        "具体错误",
        "报错信息",
    ]
    return any(pattern in answer or pattern in lowered for pattern in patterns)


def _build_pipeline_reclarification(state: dict | None, final_query: str) -> dict:
    is_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in final_query)
    candidate_intents = [
        {
            "id": "clarify_product_name",
            "label": "补充产品或型号" if is_chinese else "Product or model",
            "slot_key": "product_name",
            "slot_value": "",
        },
        {
            "id": "clarify_page_or_feature",
            "label": "补充功能或页面" if is_chinese else "Feature or page",
            "slot_key": "feature_name",
            "slot_value": "",
        },
        {
            "id": "clarify_error_symptom",
            "label": "补充报错或现象" if is_chinese else "Exact symptom",
            "slot_key": "error_symptom",
            "slot_value": "",
        },
    ]
    missing_slots: list[str] = []
    slots = state["slots"] if state else {}
    if not slots.get("product_name"):
        missing_slots.append("product_or_target")
    if not slots.get("feature_name"):
        missing_slots.append("feature_name")
    if not slots.get("error_symptom") and not slots.get("error_message"):
        missing_slots.append("error_symptom")
    if not missing_slots:
        missing_slots = ["product_or_target", "feature_name", "error_symptom"]

    return {
        "need_clarification": True,
        "reason": (
            "当前回复仍然是在向用户索要关键信息，说明链路还没有完成有效定位。"
            if is_chinese
            else "The reply is still asking the user for key details, so the pipeline has not yet truly resolved the issue."
        ),
        "clarifying_question": (
            "为了直接定位方案，请补充产品名称或型号、具体功能页面，以及报错提示或现象。"
            if is_chinese
            else "To move from a generic follow-up to a concrete fix, please share the product or model, the feature or page involved, and the exact symptom or error."
        ),
        "candidate_intents": candidate_intents,
        "missing_slots": missing_slots,
        "trigger_signals": ["pipeline_generic_answer_gate"],
    }


async def wrap_ensembles_query_classification(x: dict):
    return await ensembles_query_classification(
        x["query"],
        source_hint=x.get("source_hint"),
    )


async def router_by_query_cls(x: dict) -> str:
    query_cls = x["query_cls"]
    query = x["query"]
    question_type = query_cls["question_type"]
    top_k = x["top_k"]
    use_source = x["use_source"]
    thread_id = x["thread_id"]

    if query_cls["source"] is None and question_type == "general":
        return answer_general_query(query, thread_id)

    return await answer_product_query(
        query, "1", query_cls, top_k=top_k, use_source=use_source
    )


def _clarification_result_payload(result: dict) -> PipelineResult:
    return {
        "answer": format_clarification_message(result),
        "response_type": "clarification",
        "clarification_state": "pending",
        "candidate_intents": result["candidate_intents"],
        "missing_slots": result["missing_slots"],
        "effective_query": "",
        "session_debug": {},
    }


def _build_session_debug(
    *,
    state: dict | None,
    query_cls: dict | None = None,
) -> dict:
    debug = copy_state_for_debug(state) if state else {}
    if query_cls:
        debug["route_debug"] = query_cls.get("route_debug", {})
        debug["retrieval_debug"] = query_cls.get("_retrieval_debug", [])
        debug["route_source"] = query_cls.get("source")
        debug["route_question_type"] = query_cls.get("question_type")
        debug["route_candidate_sources"] = query_cls.get("candidate_sources", [])
    return debug


async def pipeline(query: str, thread_id: str | None = None, top_k: int = 19) -> str:
    result = await pipeline_result(query, thread_id=thread_id, top_k=top_k)
    return result["answer"]


async def pipeline_result(
    query: str,
    thread_id: str | None = None,
    top_k: int = 19,
    clarification: ClarificationFollowupPayload | None = None,
) -> PipelineResult:
    cleaned_query = query.strip().strip('"')
    is_followup = bool(clarification and clarification.get("is_followup"))
    final_query = cleaned_query
    state = None
    source_hint = None

    if is_followup and thread_id:
        state = get_session(thread_id)
        if state and state["status"] == "clarification_pending":
            logger.info(
                "clarification:followup_received session_id=%s selected_intent_id=%s text=%r",
                thread_id,
                clarification.get("selected_intent_id"),
                cleaned_query,
            )
            resolution = apply_clarification_followup(
                state,
                user_followup=cleaned_query,
                selected_intent_id=clarification.get("selected_intent_id"),
            )
            logger.info(
                "clarification:enhanced_query session_id=%s query=%r",
                thread_id,
                resolution["enhanced_query"],
            )
            final_query = resolution["enhanced_query"]
            source_hint = resolve_source_hint(
                state["slots"].get("product_name"),
                "chinese" if any("\u4e00" <= ch <= "\u9fff" for ch in final_query) else "english",
            )
            second_pass = await clarify_query(final_query)
            if (
                second_pass["need_clarification"]
                and state["clarification_round"] < 2
            ):
                state["candidate_intents"] = second_pass["candidate_intents"]
                state["missing_slots"] = second_pass["missing_slots"]
                state["status"] = "clarification_pending"
                state["clarification_reason"] = second_pass.get("reason")
                state["trigger_signals"] = second_pass.get("trigger_signals", [])
                state["history"].append(
                    {
                        "role": "assistant",
                        "content": format_clarification_message(second_pass),
                    }
                )
                save_session(state)
                logger.info(
                    "clarification:need_more_info session_id=%s round=%s",
                    thread_id,
                    state["clarification_round"],
                )
                payload = _clarification_result_payload(second_pass)
                payload["effective_query"] = final_query
                payload["session_debug"] = copy_state_for_debug(state)
                return payload

            state["status"] = "clarification_resolved"
            state["clarification_reason"] = "resolved_with_followup"
            state["trigger_signals"] = []
            save_session(state)
            logger.info("clarification:resolved session_id=%s", thread_id)
        else:
            logger.info(
                "clarification:missing_session session_id=%s, fallback to normal query",
                thread_id,
            )
    else:
        clarification_result = await clarify_query(final_query)
        if clarification_result["need_clarification"]:
            if thread_id:
                state = create_initial_state(
                    session_id=thread_id,
                    query=final_query,
                    candidate_intents=clarification_result["candidate_intents"],
                    missing_slots=clarification_result["missing_slots"],
                    assistant_message=format_clarification_message(clarification_result),
                    clarification_reason=clarification_result.get("reason"),
                    trigger_signals=clarification_result.get("trigger_signals", []),
                )
                save_session(state)
                logger.info(
                    "clarification:decision session_id=%s missing_slots=%s",
                    thread_id,
                    clarification_result["missing_slots"],
                )
            payload = _clarification_result_payload(clarification_result)
            if thread_id:
                payload["session_debug"] = copy_state_for_debug(state)
            return payload

    query_cls = await wrap_ensembles_query_classification(
        {
            "query": final_query,
            "source_hint": source_hint,
        }
    )
    try:
        answer = await router_by_query_cls(
            {
                "query": final_query,
                "top_k": top_k,
                "use_source": True,
                "thread_id": thread_id,
                "query_cls": query_cls,
            }
        )
    except ProductClarificationNeeded as exc:
        payload = _clarification_result_payload(exc.payload)
        payload["effective_query"] = final_query
        if thread_id:
            if state is None:
                state = create_initial_state(
                    session_id=thread_id,
                    query=final_query,
                    candidate_intents=exc.payload["candidate_intents"],
                    missing_slots=exc.payload["missing_slots"],
                    assistant_message=format_clarification_message(exc.payload),
                    clarification_reason=exc.payload.get("reason"),
                    trigger_signals=exc.payload.get("trigger_signals", []),
                )
            else:
                state["status"] = "clarification_pending"
                state["candidate_intents"] = exc.payload["candidate_intents"]
                state["missing_slots"] = exc.payload["missing_slots"]
                state["clarification_reason"] = exc.payload.get("reason")
                state["trigger_signals"] = exc.payload.get("trigger_signals", [])
                state["history"].append(
                    {
                        "role": "assistant",
                        "content": format_clarification_message(exc.payload),
                    }
                )
            save_session(state)
            payload["session_debug"] = _build_session_debug(state=state, query_cls=query_cls)
        return payload
    if is_followup and _is_generic_request_for_more_details(answer):
        payload_dict = _build_pipeline_reclarification(state, final_query)
        payload = _clarification_result_payload(payload_dict)
        payload["effective_query"] = final_query
        if thread_id and state is not None:
            state["status"] = "clarification_pending"
            state["candidate_intents"] = payload_dict["candidate_intents"]
            state["missing_slots"] = payload_dict["missing_slots"]
            state["clarification_reason"] = payload_dict["reason"]
            state["trigger_signals"] = payload_dict["trigger_signals"]
            state["history"].append(
                {
                    "role": "assistant",
                    "content": format_clarification_message(payload_dict),
                }
            )
            save_session(state)
            payload["session_debug"] = _build_session_debug(state=state, query_cls=query_cls)
        return payload
    if thread_id:
        clear_session(thread_id)
    return {
        "answer": answer,
        "response_type": "answer",
        "clarification_state": "resolved" if is_followup else "none",
        "candidate_intents": [],
        "missing_slots": [],
        "effective_query": final_query,
        "session_debug": _build_session_debug(
            state=state if is_followup and thread_id and state else None,
            query_cls=query_cls,
        ),
    }


async def pipeline_stream(
    query: str,
    thread_id: str | None = None,
    top_k: int = 19,
    clarification: ClarificationFollowupPayload | None = None,
) -> AsyncGenerator[str, None]:
    result = await pipeline_result(
        query,
        thread_id=thread_id,
        top_k=top_k,
        clarification=clarification,
    )
    if result["response_type"] == "clarification":
        data = {
            "delta": result["answer"],
            "response_type": result["response_type"],
            "clarification_state": result["clarification_state"],
            "candidate_intents": result["candidate_intents"],
            "missing_slots": result["missing_slots"],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        return

    if clarification and clarification.get("is_followup"):
        data = {
            "delta": result["answer"],
            "response_type": "answer",
            "clarification_state": result["clarification_state"],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        return

    pipeline_chain = RunnablePassthrough.assign(
        query_cls=RunnableLambda(wrap_ensembles_query_classification)
    ) | RunnableLambda(router_by_query_cls)

    async for event in pipeline_chain.with_retry().astream_events(
        {
            "query": query.strip().strip('"'),
            "top_k": top_k,
            "use_source": True,
            "thread_id": thread_id,
        },
        include_tags=["final_answer_model"],
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                data = {
                    "delta": chunk.content,
                    "response_type": "answer",
                    "clarification_state": result["clarification_state"],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
