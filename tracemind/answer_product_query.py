import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# from llm_judge_result import judge_result_by_llm, refine_answer
from tracemind.config import get_config
from tracemind.retriever import retriever
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


refine_answer_llm = ChatOpenAI(
    model="gpt-5.5",
    base_url=os.getenv("OPEANAI_BASE_URL"),
    api_key=os.getenv("OPEANAI_API_KEY"),
    tags=["final_answer_model"],
)


def ensure_answer_language(
    answer: str, image_names: list[str], query_language: Literal["chinese", "english"]
) -> bool:
    """
    确保answer的语言和query_language的语言一致
    """
    answer_language = language_detect(answer)
    if answer_language != query_language:
        raise Exception(f"模型回答的的语言不是{query_language}")
    return True


def llm_can_answer_the_question(answer: str) -> bool:
    """
    检查LLM是否能够回答用户的问题,如果不能回答，则可能是source的预测有问题，之后重新不使用source进行检索生成
    """
    if "我不能回答这个问题" in answer or "I cannot answer the question" in answer:
        return False
    return True


async def get_context(x: dict) -> str:
    results = await retriever(
        x["query"],
        x["query_cls"],
        x["top_k"],
        x["use_source"],
    )
    return "\n\n".join([result.page_content for result in results])


async def refine_answer_direct(query: str, origin_ret: str, context: str) -> str:
    """
    使用LLM对上一步生成的答案进行优化
    """
    refine_answer_prompt_template = """你是一个客服专家，你需要根据用户的问题和所给上下文对这段客服回答进行优化。
# 任务
我会给你用户的问题，客服对该问题的回答（包含图片）、上下文信息，你需要根据原来的回答、回答中的图片以及上下文信息优化该回答，你需要在保证原来回答的基本内容不变的前提下，对回答进行优化，使得优化后的回答结构严谨连贯，图片与文本完美互补，你优化之后的回答是最终回答用户的答案。

# 你优化的标准
你需要保证回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# TIPS
图中的<PIC>表示这个位置该放置图片，图片放置的顺序和输入的图片顺序一致

# 约束
你的回答中不要提及说明书等词语，你的回答需要符合正常客服的回答习惯等。

# 输出格式
你需要以json的格式输出优化后的答案，格式需和input_answer的结构一致，使用<answer>将答案内容包裹起来，需要插入图片的地方你需要使用PIC标签来表示图片，image_name的属性是图片名。例如：<pic image_name="Manual16_01"></pic>
{   
    "refined_answer": str,  #优化后的答案
}
# 用户问题
{{query}}
# 客服回答
{{answer}}
# 上下文信息
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
    messages = [
        {
            "role": "human",
            "content": [
                {
                    "type": "text",
                    "text": refine_answer_prompt_template,
                },
            ],
        }
    ]
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

    ret = convert_answer_to_ret(result["refined_answer"])

    new_data = ret.split(",[")
    if len(new_data) > 1:
        new_image_list = eval(f"[{new_data[1]}")
    else:
        new_image_list = []

    new_image_list = [
        get_image_name(os.path.join(IMAGE_ROOT_DIR, image)) for image in new_image_list
    ]

    # assert len(image_list) <= len(new_image_list), (
    #     "优化后的图片数量要大于等于优化前的图片数量"
    # )
    if len(image_list) != len(new_image_list):
        print(
            f"问题{query}存在图片数量不一致的情况，优化前的长度:{len(image_list)}，优化后的长度:{len(new_image_list)}"
        )
    # 大于的优化前的图片数量就直接返回，不对比
    if len(image_list) == len(new_image_list):
        for origin_image, new_image in zip(image_list, new_image_list):
            if origin_image != new_image:
                print(
                    f"问题{query}存在图片不一致的情况，优化前的图片:{image_list}，优化后的图片:{new_image_list}"
                )
                # raise Exception(
                #     f"优化前后的图片不一致，优化前的图片为{image_list}，优化后的图片为{new_image_list}"
                # )

    return ret


async def answer_product_query(
    query: str,
    thread_id: str,
    query_cls: dict,
    top_k: int,
    use_source: bool,
):
    """处理产品类的问题，得到最终的答案"""
    llm = ChatOpenAI(
        model="gpt-5.5",
        base_url=os.getenv("OPEANAI_BASE_URL"),
        api_key=os.getenv("OPEANAI_API_KEY"),
    )
    generate_answer_prompt_template = """你是一个智能客服，你需要根据用户的问题，以及知识库中检索出的相关上下文来回答用户的问题，上下文内容中<picture-description></picture-description>中的内容是插图的内容，属性image_name是对应的图片名称。
# 任务
你需要以图文互补的形式来回答用户的问题，需要插入图片的地方你需要使用PIC标签来表示图片，image_name的属性是图片名。例如：<pic image_name="Manual16_01"></pic>

# 要求
1. 上下文可能会存在冗余，你的回答必须要简洁，针对用户的问题来进行回答，**不需要回答与用户问题无关的内容，也不需要对用户提问**
2. 你的回答不必过于冗余，针对用户问题回答即可，结构严谨连贯，图片与文本完美互补，帮助用户更好地理解答案
3. 图片必须和答案相关，能够解决用户的问题，并且有助于用户更好地理解答案
4. 用户的问题是什么语言，你的答案也必须是什么语言
5. 如果所给的上下文不能回答用户的问题，你需要回答“我不能回答这个问题”,如果用户的语言是英文，则回答“I cannot answer the question”
6. 如果所给的上下文中有能够直接回答用户问题的内容，**你需要优先采用上下文中的原本的内容来回答，不要使用markdown的加粗的语法,不需要加粗，文字、图片顺序、图片名称需要和原文保持一致**
7. 如果用户的问题中有明显的语法错误、逻辑错误的内容，你在回答的时候应该先指出来，再回答问题

# 输出格式
你必须使用answer标签来包裹你的答案，如下:
<answer>
</answer>
# 用户的问题为
{{query}}

# 相关的上下文为
<context>
{{context}}
</context>
"""
    generate_answer_prompt = PromptTemplate.from_template(
        generate_answer_prompt_template, template_format="mustache"
    )

    product_answer_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(get_context))
        | RunnablePassthrough.assign(
            parsed_answer=(
                generate_answer_prompt
                | llm
                | StrOutputParser()
                | RunnableLambda(parse_answer)
            )
        )
        | RunnablePassthrough.assign(
            is_correct_language=lambda x: ensure_answer_language(
                x["parsed_answer"][0],
                x["parsed_answer"][1],
                x["query_cls"]["language"],
            )
        )
    )
    # 第一次回答
    res = await product_answer_chain.ainvoke(
        {
            "query": query.strip('"'),
            "query_cls": query_cls,
            "top_k": top_k,
            "use_source": use_source,
        }
    )
    answer = res["parsed_answer"][0]
    image_names = res["parsed_answer"][1]
    context = res["context"]
    # 如果查询分类错误可能会导致检索的上下文不相关，导致模型回答不了
    if not llm_can_answer_the_question(answer):
        logger.info(f"first answer: {query} -> {answer}")
        # 不使用查询分类预测的 source 重新检索，并基于新上下文重新生成答案
        res = await product_answer_chain.ainvoke(
            {
                "query": query.strip('"'),
                "query_cls": query_cls,
                "top_k": top_k,
                "use_source": False,
            }
        )
        answer = res["parsed_answer"][0]
        image_names = res["parsed_answer"][1]
        context = res["context"]

    origin_ret = answer
    if image_names and len(image_names) > 0:
        origin_ret += "," + str(image_names)
    # 优化生成的答案
    new_ret = await refine_answer_direct(query, origin_ret, context)
    return new_ret
