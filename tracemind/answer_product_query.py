import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from tracemind.config import get_config
from tracemind.model_factory import create_chat_model
from tracemind.retriever import retriever, summarize_retrieval
from tracemind.utils import (
    convert_answer_to_ret,
    get_image_name,
    language_detect,
    parse_answer,
)

load_dotenv()

IMAGE_ROOT_DIR = get_config()["IMAGE_ROOT_DIR"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


refine_answer_llm = create_chat_model(
    "REFINE_LLM",
    tags=["final_answer_model"],
)


class ProductClarificationNeeded(Exception):
    def __init__(
        self,
        *,
        reason: str,
        clarifying_question: str,
        candidate_intents: list[dict],
        missing_slots: list[str],
        trigger_signals: list[str] | None = None,
    ) -> None:
        super().__init__(reason)
        self.payload = {
            "need_clarification": True,
            "reason": reason,
            "clarifying_question": clarifying_question,
            "candidate_intents": candidate_intents,
            "missing_slots": missing_slots,
            "trigger_signals": trigger_signals or ["low_retrieval_confidence"],
        }


def ensure_answer_language(
    answer: str,
    image_names: list[str],
    query_language: Literal["chinese", "english"],
) -> bool:
    del image_names
    answer_language = language_detect(answer)
    if answer_language != query_language:
        raise Exception(f"model output language is not {query_language}")
    return True


async def _rewrite_answer_language(
    *,
    answer: str,
    query_language: Literal["chinese", "english"],
) -> tuple[str, list[str]]:
    llm = create_chat_model("REFINE_LLM")
    target_language = "Chinese" if query_language == "chinese" else "English"
    prompt_template = """Rewrite the support answer into {target_language}. Keep the same meaning, step order, and any <pic image_name="..."></pic> tags. Do not add new facts.

Output format:
<answer>
</answer>

Original answer:
{answer}
"""
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser() | parse_answer
    rewritten_answer, image_names = await chain.ainvoke(
        {"answer": answer, "target_language": target_language}
    )
    ensure_answer_language(rewritten_answer, image_names, query_language)
    return rewritten_answer, image_names


def llm_can_answer_the_question(answer: str) -> bool:
    lowered = answer.lower()
    negative_patterns = [
        "我不能回答这个问题",
        "我无法回答这个问题",
        "还不能准确回答",
        "需要更多信息",
        "i cannot answer the question",
        "i cannot answer this question",
        "cannot answer the question",
        "cannot answer this question",
        "cannot answer further",
        "need more information",
        "unable to determine",
    ]
    return not any(pattern in lowered or pattern in answer for pattern in negative_patterns)


def _results_to_context(results: list) -> str:
    return "\n\n".join([result.page_content for result in results])


def _has_product_target_marker(query: str) -> bool:
    lowered = query.lower()
    return (
        "product target:" in lowered
        or "产品对象：" in query
        or "产品对象:" in query
    )


def _should_clarify_before_answer(
    query: str,
    query_cls: dict,
    retrieval_summary: dict,
) -> bool:
    if query_cls.get("source") is not None:
        return False
    if _has_product_target_marker(query):
        return False

    top_source_hits = (
        retrieval_summary["top_sources"][0][1]
        if retrieval_summary["top_sources"]
        else 0
    )
    source_concentration = top_source_hits / max(retrieval_summary["hits"], 1)
    return retrieval_summary["unique_sources"] >= 3 and source_concentration < 0.35


def _low_confidence_clarification_payload(
    language: Literal["chinese", "english"],
) -> ProductClarificationNeeded:
    if language == "chinese":
        return ProductClarificationNeeded(
            reason="当前检索结果来源比较分散，系统还无法稳定定位到同一个产品或场景。",
            clarifying_question="为了更准确定位答案，可以再补充一下是哪个产品或型号，以及你现在遇到的具体现象吗？",
            candidate_intents=[
                {
                    "id": "clarify_product_name",
                    "label": "补充产品或型号",
                    "slot_key": "product_name",
                    "slot_value": "",
                },
                {
                    "id": "clarify_page_or_feature",
                    "label": "补充页面或功能",
                    "slot_key": "feature_name",
                    "slot_value": "",
                },
                {
                    "id": "clarify_error_symptom",
                    "label": "补充报错或现象",
                    "slot_key": "error_symptom",
                    "slot_value": "",
                },
            ],
            missing_slots=["product_or_target", "error_symptom"],
        )
    return ProductClarificationNeeded(
        reason="The retrieval results are still too scattered to reliably lock onto one product or scenario.",
        clarifying_question="To narrow this down, could you share the product or model and the exact symptom you are seeing?",
        candidate_intents=[
            {
                "id": "clarify_product_name",
                "label": "Product or model",
                "slot_key": "product_name",
                "slot_value": "",
            },
            {
                "id": "clarify_page_or_feature",
                "label": "Page or feature",
                "slot_key": "feature_name",
                "slot_value": "",
            },
            {
                "id": "clarify_error_symptom",
                "label": "Exact symptom",
                "slot_key": "error_symptom",
                "slot_value": "",
            },
        ],
        missing_slots=["product_or_target", "error_symptom"],
    )


async def refine_answer_direct(query: str, origin_ret: str, context: str) -> str:
    refine_answer_prompt_template = """You are polishing a support reply using the user's question and the retrieved context.

Task:
1. Keep the answer faithful to the existing answer and context.
2. Preserve <pic image_name="..."></pic> tags.
3. Keep the response concise, readable, and professional.
4. Do not add unsupported facts.

Output JSON:
{
  "refined_answer": str
}

User question:
{{query}}

Current answer:
{{answer}}

Context:
{{context}}
"""
    data = origin_ret.split(",[")
    origin_answer = data[0]
    if len(data) > 1:
        origin_image_list = eval(f"[{data[1]}")
    else:
        origin_image_list = []

    image_list = [
        get_image_name(os.path.join(IMAGE_ROOT_DIR, image))
        for image in origin_image_list
    ]
    messages = [{"role": "human", "content": [{"type": "text", "text": refine_answer_prompt_template}]}]
    prompt = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    chain = prompt | refine_answer_llm | JsonOutputParser()
    result = await chain.with_retry().ainvoke(
        {
            "query": query,
            "answer": origin_answer,
            "image_list": origin_image_list,
            "context": context,
        }
    )

    try:
        ret = convert_answer_to_ret(result["refined_answer"])
    except Exception:
        logger.exception(
            "refine_answer_direct: failed to parse refined answer, fallback to original answer"
        )
        return origin_ret

    new_data = ret.split(",[")
    if len(new_data) > 1:
        new_image_list = eval(f"[{new_data[1]}")
    else:
        new_image_list = []

    new_image_list = [
        get_image_name(os.path.join(IMAGE_ROOT_DIR, image))
        for image in new_image_list
    ]

    if len(image_list) != len(new_image_list):
        logger.info(
            "refine_answer_direct:image_count_changed before=%s after=%s query=%r",
            len(image_list),
            len(new_image_list),
            query,
        )
    return ret


async def _generate_answer(
    *,
    query: str,
    context: str,
    query_language: Literal["chinese", "english"],
) -> tuple[str, list[str]]:
    llm = create_chat_model("PRODUCT_LLM")
    generate_answer_prompt_template = """You are a support agent answering from retrieved manual context.

Rules:
1. Use context that directly supports the answer.
2. If the context is not enough to answer, reply exactly:
   - Chinese: 我不能回答这个问题
   - English: I cannot answer the question
3. Answer in the same language as the user.
4. When useful, you may include <pic image_name="xxx"></pic>.

Output format:
<answer>
</answer>

User question:
{{query}}

Related context:
<context>
{{context}}
</context>
"""
    generate_answer_prompt = PromptTemplate.from_template(
        generate_answer_prompt_template,
        template_format="mustache",
    )
    chain = generate_answer_prompt | llm | StrOutputParser() | parse_answer
    answer, image_names = await chain.ainvoke({"query": query, "context": context})
    try:
        ensure_answer_language(answer, image_names, query_language)
    except Exception:
        answer, image_names = await _rewrite_answer_language(
            answer=answer,
            query_language=query_language,
        )
    return answer, image_names


async def _generate_troubleshooting_answer(
    *,
    query: str,
    context: str,
    query_language: Literal["chinese", "english"],
) -> tuple[str, list[str]]:
    llm = create_chat_model("PRODUCT_LLM")
    troubleshooting_prompt_template = """You are a product troubleshooting support agent. Use the provided manual context to give the user the most actionable troubleshooting help you can.

Goals:
1. If the user has already identified the product or model, do not fall back to "I cannot answer the question" unless the context is truly unrelated.
2. Prefer concrete troubleshooting guidance drawn from the manual: startup checks, installation checks, safety interlocks, operating steps, reset actions, and when to contact support.
3. If the context is only partially relevant, still give the best manual-grounded checks instead of asking the user to restate the same problem.
4. The reply must stay in the user's language.
5. If helpful images are mentioned in context, you may use <pic image_name="xxx"></pic>.

Required answer structure when you can help:
- One short opening sentence that acknowledges the issue.
- One short "Quick check" line or an equivalent diagnosis direction.
- 2 to 5 numbered troubleshooting steps.
- One final line saying when to stop and contact after-sales or support if the issue still remains.

Constraints:
1. Do not invent facts outside the provided context.
2. Do not ask the user to provide more details unless the context is truly insufficient to give even basic checks.
3. If the context is truly unrelated, answer exactly:
   - Chinese: 我不能回答这个问题
   - English: I cannot answer the question

Output format:
<answer>
</answer>

User question:
{{query}}

Related context:
<context>
{{context}}
</context>
"""
    prompt = PromptTemplate.from_template(
        troubleshooting_prompt_template,
        template_format="mustache",
    )
    chain = prompt | llm | StrOutputParser() | parse_answer
    answer, image_names = await chain.ainvoke({"query": query, "context": context})
    try:
        ensure_answer_language(answer, image_names, query_language)
    except Exception:
        answer, image_names = await _rewrite_answer_language(
            answer=answer,
            query_language=query_language,
        )
    return answer, image_names


async def answer_product_query(
    query: str,
    thread_id: str,
    query_cls: dict,
    top_k: int,
    use_source: bool,
):
    logger.info(
        "product:start query=%r source=%s language=%s top_k=%s use_source=%s",
        query,
        query_cls.get("source"),
        query_cls.get("language"),
        top_k,
        use_source,
    )

    del thread_id

    first_results = await retriever(
        query.strip('"'),
        query_cls,
        top_k,
        use_source,
        stage="primary",
    )
    first_summary = summarize_retrieval(first_results)
    logger.info("product:retrieval_summary_first %s", first_summary)
    if _should_clarify_before_answer(query, query_cls, first_summary):
        raise _low_confidence_clarification_payload(query_cls["language"])

    context = _results_to_context(first_results)
    logger.info(
        "product:get_context query=%r use_source=%s chunks=%s",
        query,
        use_source,
        len(first_results),
    )
    answer, image_names = await _generate_answer(
        query=query.strip('"'),
        context=context,
        query_language=query_cls["language"],
    )
    logger.info(
        "product:first_pass answer_preview=%r images=%s context_chars=%s",
        answer[:160],
        len(image_names),
        len(context),
    )

    if not llm_can_answer_the_question(answer):
        logger.info("first answer: %s -> %s", query, answer)
        logger.info("product:fallback retry_without_source=True")
        if query_cls.get("_retrieval_debug"):
            query_cls["_retrieval_debug"][-1]["fallback_triggered"] = True
        if query_cls.get("source") is not None:
            try:
                answer, image_names = await _generate_troubleshooting_answer(
                    query=query.strip('"'),
                    context=context,
                    query_language=query_cls["language"],
                )
                logger.info(
                    "product:troubleshooting_pass answer_preview=%r images=%s context_chars=%s",
                    answer[:160],
                    len(image_names),
                    len(context),
                )
                if llm_can_answer_the_question(answer):
                    origin_ret = answer
                    if image_names:
                        origin_ret += "," + str(image_names)
                    new_ret = await refine_answer_direct(query, origin_ret, context)
                    logger.info("product:final answer_preview=%r", new_ret[:200])
                    return new_ret
            except Exception:
                logger.exception("product:troubleshooting_pass_failed")

        second_results = await retriever(
            query.strip('"'),
            query_cls,
            top_k,
            False,
            stage="fallback_no_source",
        )
        context = _results_to_context(second_results)
        logger.info(
            "product:get_context query=%r use_source=%s chunks=%s",
            query,
            False,
            len(second_results),
        )
        answer, image_names = await _generate_answer(
            query=query.strip('"'),
            context=context,
            query_language=query_cls["language"],
        )
        logger.info(
            "product:second_pass answer_preview=%r images=%s context_chars=%s",
            answer[:160],
            len(image_names),
            len(context),
        )
        retrieval_summary = summarize_retrieval(second_results)
        logger.info("product:retrieval_summary %s", retrieval_summary)
        top_source_hits = (
            retrieval_summary["top_sources"][0][1]
            if retrieval_summary["top_sources"]
            else 0
        )
        source_concentration = top_source_hits / max(retrieval_summary["hits"], 1)
        if not llm_can_answer_the_question(answer) and (
            retrieval_summary["unique_sources"] >= 3 or source_concentration < 0.75
        ):
            raise _low_confidence_clarification_payload(query_cls["language"])

    origin_ret = answer
    if image_names:
        origin_ret += "," + str(image_names)
    new_ret = await refine_answer_direct(query, origin_ret, context)
    logger.info("product:final answer_preview=%r", new_ret[:200])
    return new_ret
