import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from tracemind.answer_general_query import answer_general_query
from tracemind.answer_product_query import answer_product_query
from tracemind.clarifier import clarify_query, format_clarification_message
from tracemind.config import get_config
from tracemind.query_classification import ensembles_query_classification
from tracemind.utils import language_detect

IMAGE_ROOT_DIR = get_config()["IMAGE_ROOT_DIR"]
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineResult(TypedDict):
    answer: str
    response_type: Literal["answer", "clarification"]
    candidate_intents: list[str]


def ensure_answer_language(
    answer: str, image_names: list[str], query_language: Literal["chinese", "english"]
) -> tuple[str, list[str]]:
    """
    确保answer的语言和query_language的语言一致
    """
    answer_language = language_detect(answer)
    if answer_language != query_language:
        raise Exception(f"模型回答的的语言不是{query_language}")

    return answer, image_names


def llm_can_answer_the_question(answer: str) -> bool:
    """
    检查LLM是否能够回答用户的问题
    """
    if "我不能回答这个问题" in answer or "I cannot answer the question" in answer:
        return False
    return True


async def wrap_ensembles_query_classification(x: dict):
    return await ensembles_query_classification(x["query"])


async def router_by_query_cls(x: dict) -> str:
    """
    根据问题分类的结果进行路由, 同时问题通过answer_general_query, 产品问题则使用answer_product_query
    """
    query_cls = x["query_cls"]
    query = x["query"]
    question_type = query_cls["question_type"]
    top_k = x["top_k"]
    use_source = x["use_source"]
    thread_id = x["thread_id"]

    if query_cls["source"] is None and question_type == "general":
        return answer_general_query(query, thread_id)
    else:
        ret = await answer_product_query(
            query, "1", query_cls, top_k=top_k, use_source=use_source
        )
        return ret


async def pipeline(query: str, thread_id: str | None = None, top_k: int = 19) -> str:
    result = await pipeline_result(query, thread_id=thread_id, top_k=top_k)
    return result["answer"]


async def pipeline_result(
    query: str, thread_id: str | None = None, top_k: int = 19
) -> PipelineResult:
    cleaned_query = query.strip('"')
    clarification = await clarify_query(cleaned_query)
    if clarification["need_clarification"]:
        return {
            "answer": format_clarification_message(clarification),
            "response_type": "clarification",
            "candidate_intents": clarification["candidate_intents"],
        }

    pipeline_chain = RunnablePassthrough.assign(
        query_cls=RunnableLambda(wrap_ensembles_query_classification)
    ) | RunnableLambda(router_by_query_cls)
    answer = await pipeline_chain.with_retry().ainvoke(
        {
            "query": cleaned_query,
            "top_k": top_k,
            "use_source": True,
            "thread_id": thread_id,
        }
    )
    return {
        "answer": answer,
        "response_type": "answer",
        "candidate_intents": [],
    }


async def pipeline_stream(
    query: str,
    thread_id: str | None = None,
    top_k: int = 19,
) -> AsyncGenerator[str, None]:
    result = await pipeline_result(query, thread_id=thread_id, top_k=top_k)
    if result["response_type"] == "clarification":
        data = {
            "delta": result["answer"],
            "response_type": result["response_type"],
            "candidate_intents": result["candidate_intents"],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        return

    pipeline_chain = RunnablePassthrough.assign(
        query_cls=RunnableLambda(wrap_ensembles_query_classification)
    ) | RunnableLambda(router_by_query_cls)

    async for event in pipeline_chain.with_retry().astream_events(
        {
            "query": query.strip('"'),
            "top_k": top_k,
            "use_source": True,
            "thread_id": thread_id,
        },
        include_tags=["final_answer_model"],
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                data = {"delta": chunk.content, "response_type": "answer"}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                # yield  chunk.content


if __name__ == "__main__":
    import asyncio

    async def main():

        query = "操作吹风机时，人员需要注意哪些安全要点？"
        query = "空气净化器通常有哪些模式？如何设置？这些模式有什么特点？"
        query = "使用吹风机时，如何调节化油器？"
        query = "吹风机冷机时，该如何启动？"
        query = "吹风机热机时，该如何启动？"

        # query = "我收到的商品和图片不一样，颜色偏差很大，我要投诉！"
        top_k = 19
        queries = [
            # "该如何关闭吹风机？",
            # "空调的重要组成部件有哪些？",
            # "如何找到空调遥控器的按键？",
            # "如何给空调遥控器安装电池？",
            # "如何安装空调遥控器支架？",
            # "如何用空调快速调节室内温度？",
            # "如何使用空调的自清洁运行功能？",
            # "如何使用空调的等离子净化功能？",
            # "单冷型空调如何开启自动运行模式？",
            # "如何使用空调的自动转换运行功能？",
            # "如何开启空调的节能制冷模式？",
            # "无遥控器时如何操作空调？",
            # "Have you ever wondered how to remove the camera shutter button? Understanding this process can enhance your photography experience and allow for quick repairs!",
            "我收到的商品和图片不一样，颜色偏差很大，我要投诉！",
            "使用吹风机时，人员需要佩戴哪些防护装备？",
            "如何清洁洗碗机的进水管滤网？",
            "空气净化器需要长期存放时该怎么做？",
            "健身追踪器是如何测量我的心率的？",
            "如何在Windows系统中为蓝牙激光鼠标设置快速配对？",
            "如何使用烤箱的滑动搁架？",
            "How can you install the handset of a landline?如何让空调实现自动重启？",
            "不同型号空调的清洁频率是多少？",
            "如何清洁空调的空气滤网？",
            "如何清洁空调的3M多重防护滤网？",
            "如何清洁空调的等离子滤网？",
            "如何快速组装蒸汽清洁机？",
            "蒸汽清洁机有哪些实用的产品功能？如何快速上手使用？",
            "如何使用蒸汽清洁机清洁硬质地面？",
            "组装人体工学椅涉及哪些部件？",
            "椅子的扶手使用一段时间后为什么会松动？",
            "这款椅子有哪些功能？",
            "洗碗机的部件有哪些？",
            "首次使用时，如何将洗碗机连接到排水口？",
            "使用前如何为洗碗机添加专用盐？",
            "如何为洗碗机添加洗涤剂？",
            "如何为洗碗机添加洗涤块？",
        ]

        for query in queries:
            thread_id = "1"
            start_time = time.time()
            answer = await pipeline(query, thread_id, top_k)
            # print(
            #     f"query: {query}\nanswer: {answer}\ncost time: {time.time() - start_time}s\n\n"
            # )
            print(f"cost time: {time.time() - start_time}s\n\n")
            pass
            # print()
            # print(answer)

    asyncio.run(main())
