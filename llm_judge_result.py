"""使用LLM对生成的答案进行打分"""

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwQ
from tqdm import tqdm

from utils import convert_answer_to_ret, encode_image, get_image_name

load_dotenv()

IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", "data/KownledgeBase/手册/插图")

judge_result_reanswer_prompt_template = """你是一个客服专家，你需要根据用户的问题对这段客服回答进行打分并进行优化。
# 打分规则
1 分，质量差：回答未回应问题，结构混乱或缺失，图片无关或无帮助。
2 分，质量一般：回答部分回应问题，但不完整；结构较弱，图文结合较差或仅部分有帮助。
3 分，质量中等：回答回应了问题，但缺乏深度；结构清晰但可优化；图片有一定帮助，但未充分提升理解。
4 分，质量良好：回答清晰、较为全面；结构逻辑清晰、组织合理，图片有助于理解文本。
5 分，质量优秀：回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# TIPS
图中的<PIC>表示这个位置该放置图片，图片放置的顺序和输入的图片顺序一致

# 输出格式
你需要以json的格式输出大纲，json的结构如下：
{   
    "reason": str,  # 打分原因
    "score": int  # 对于这个回答的质量打分,取值范围为0-5的整数
    "new_answer":   # 你根据原来的回答以及上下文的内容优化该回答，你不能添加和减少原来答案中的PIC的数量和顺序，你的答案中，图片也是必须和原来答案中的图片顺序一致并且使用'<PIC>'作为占位符，你的回答内容不能随便编造，必须基于原来的回答以及上下文的内容,你不需要反问用户
}

# 用户问题
{{query}}
# 客服回答
{{answer}}
# 上下文
{{context}}
"""
judge_result_not_reanswer_prompt_template = """你是一个客服专家，你需要根据用户的问题对这段客服回答进行打分。
# 打分规则
1 分，质量差：回答未回应问题，结构混乱或缺失，图片无关或无帮助。
2 分，质量一般：回答部分回应问题，但不完整；结构较弱，图文结合较差或仅部分有帮助。
3 分，质量中等：回答回应了问题，但缺乏深度；结构清晰但可优化；图片有一定帮助，但未充分提升理解。
4 分，质量良好：回答清晰、较为全面；结构逻辑清晰、组织合理，图片有助于理解文本。
5 分，质量优秀：回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# TIPS
图中的<PIC>表示这个位置该放置图片，图片放置的顺序和输入的图片顺序一致

# 输出格式
你需要以json的格式输出大纲，json的结构如下：
{   
    "reason": str,  # 打分原因
    "score": int  # 对于这个回答的质量打分,取值范围为0-5的整数
}

# 用户问题
{{query}}
# 客服回答
{{answer}}
"""

judge_general_answer_prompt_template = """你是一个客服专家，你需要根据用户的问题对这段客服回答进行打分。
# 打分规则
1 分，质量差：回答未回应问题，结构混乱或缺失，图片无关或无帮助。
2 分，质量一般：回答部分回应问题，但不完整；结构较弱，图文结合较差或仅部分有帮助。
3 分，质量中等：回答回应了问题，但缺乏深度；结构清晰但可优化；图片有一定帮助，但未充分提升理解。
4 分，质量良好：回答清晰、较为全面；结构逻辑清晰、组织合理，图片有助于理解文本。
5 分，质量优秀：回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# 输出格式
你需要以json的格式输出大纲，json的结构如下：
{   
    "reason": str,  # 打分原因
    "score": float  # 对于这个回答的质量打分,取值范围为0-5的浮点数
}

# 用户问题
{{query}}
# 客服回答
{{answer}}
"""

# refine_answer_prompt_template = """你是一个客服专家，你需要根据用户的问题和所给上下文对这段客服回答进行优化。
# # 任务
# 我会给你用户的问题，客服对该问题的回答（包含图片）、上下文信息、以及别的客服专家对这个回答的评价，你需要根据原来的回答、回答中的图片，别的客服专家对这个回答的评价以及上下文信息优化该回答，你优化之后的回答是最终回答用户的答案。

# # 你优化的标准
# 你需要保证回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# # TIPS
# 图中的<PIC>表示这个位置该放置图片，图片放置的顺序和输入的图片顺序一致

# # 约束
# 你的回答中不要提及说明书等词语，你的回答需要符合正常客服的回答习惯等。

# # 输出格式
# 你需要以json的格式输出优化后的答案，格式需和input_answer的结构一致，使用<answer>将答案内容包裹起来，需要插入图片的地方你需要使用PIC标签来表示图片，image_name的属性是图片名。例如：<pic image_name="Manual16_01"></pic>
# {
#     "refined_answer": str,  #优化后的答案
# }
# # 用户问题
# {{query}}
# # 客服回答
# {{answer}}
# # 别的客服专家对这个回答的评价
# {{judge}}
# # 上下文信息
# {{context}}
# """


# async def refine_answer(query: str, origin_ret: str, reason: str, context: str) -> str:
#     data = origin_ret.split(",[")
#     origin_answer = data[0]
#     if len(data) > 1:
#         image_list = eval(f"[{data[1]}")
#     else:
#         image_list = []

#     image_list = [
#         get_image_name(os.path.join(IMAGE_ROOT_DIR, image)) for image in image_list
#     ]
#     llm = ChatOpenAI(
#         model="gpt-5.5",
#         base_url=os.getenv("OPEANAI_BASE_URL"),
#         api_key=os.getenv("OPEANAI_API_KEY"),
#     )
#     messages = [
#         {
#             "role": "human",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": refine_answer_prompt_template,
#                 },
#             ],
#         }
#     ]
#     prompt = ChatPromptTemplate.from_messages(messages, template_format="mustache")
#     chain = prompt | llm | JsonOutputParser()
#     result = await chain.with_retry().ainvoke(
#         {
#             "query": query,
#             "judge": reason,
#             "answer": origin_answer,
#             "context": context,
#         }
#     )

#     ret = convert_answer_to_ret(result["refined_answer"])
#     return ret


async def judge_general_answer_by_llm(query: str, answer: str):
    llm = ChatOpenAI(
        model="gpt-5.4",
        base_url=os.getenv("OPEANAI_BASE_URL"),
        api_key=os.getenv("OPEANAI_API_KEY"),
    )
    prompt = PromptTemplate.from_template(
        judge_general_answer_prompt_template, template_format="mustache"
    )
    chain = prompt | llm | JsonOutputParser()
    result = await chain.with_retry().ainvoke(
        {
            "query": query,
            "answer": answer,
        }
    )

    return result["reason"], result["score"]


async def judge_result_by_llm(
    query: str, ret: str, context: str | None = None, need_reanswer: bool = False
) -> tuple[str, float, str | None]:
    """使用LLM对生成的答案进行打分，用来找出哪些答案分比较低,有问题,再找出原因进行优化"""
    data = ret.split(",[")
    description = data[0]
    if len(data) > 1:
        image_list = eval(f"[{data[1]}")
    else:
        image_list = []

    image_list = [
        get_image_name(os.path.join(IMAGE_ROOT_DIR, image)) for image in image_list
    ]
    image_messages = []
    # llm = ChatOpenAI(
    #     # model="gemini-3.1-pro-preview",
    #     model="gemini-3-flash-preview",
    #     base_url=os.getenv("GEMINI_BASE_URL"),
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )

    # llm = ChatOpenAI(
    #     model="gpt-5.5",
    #     base_url=os.getenv("OPEANAI_BASE_URL"),
    #     api_key=os.getenv("OPEANAI_API_KEY"),
    # )
    llm = ChatOpenAI(
        model="gpt-5.4",
        base_url=os.getenv("OPEANAI_BASE_URL"),
        api_key=os.getenv("OPEANAI_API_KEY"),
    )
    # llm = ChatOpenAI(
    #     model="qwen3.5-27b",
    #     base_url=os.getenv("DASHSCOPE_BASE_URL"),
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     extra_body={"enable_thinking": True},
    # )
    # llm = ChatQwQ(
    #     model="qwen3.5-27b",
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )

    for image in image_list:
        if not Path(image).exists():
            raise FileNotFoundError(f"图片路径不存在：{image}")
        image_base64, mime_type = encode_image(image)
        image_messages.append(
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
            },
        )

    messages = [
        {
            "role": "human",
            "content": [
                *image_messages,
                {
                    "type": "text",
                    "text": judge_result_reanswer_prompt_template
                    if need_reanswer
                    else judge_result_not_reanswer_prompt_template,
                },
            ],
        }
    ]
    prompt = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    chain = prompt | llm | JsonOutputParser()
    inputs = {
        "query": query,
        "answer": description,
    }
    if need_reanswer:
        if context is None:
            raise ValueError("context is None")
        else:
            inputs["context"] = context
    try:
        res = await chain.with_retry().ainvoke(inputs)
    except Exception as e:
        return "", -1, None
    reason, score = res["reason"], res["score"]
    return reason, score, res.get("new_answer", None)


async def judge_all_result(
    submit_csv_path: str, question_csv_path: str = "data/question_public.csv"
):
    """对所有生成的答案进行打分，用来找出哪些答案分比较低,有问题,再找出原因进行优化"""
    answer_df = pd.read_csv(submit_csv_path, index_col="id")
    question_df = pd.read_csv(question_csv_path, index_col="id")
    save_dir = "eval"
    judge_save_file = Path(
        save_dir,
        Path(Path(submit_csv_path).stem + "_judge_result_gemini_flash.json").name,
    )
    results = []
    exist_last_id = 58
    placeholder_answer = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"
    max_concurrency = 8
    tasks = []
    batch_ids = []
    queries = []
    end_id = 436

    if judge_save_file.exists():
        results = json.load(open(judge_save_file, "r"))
        exist_last_id = max([result["id"] for result in results])

    for answer_row, question_row in tqdm(
        zip(answer_df.iterrows(), question_df.iterrows()), total=len(answer_df)
    ):
        assert answer_row[0] == answer_row[0]
        id = answer_row[0]
        ret = answer_row[1]["ret"]
        query = question_row[1]["question"]
        if id <= exist_last_id:
            continue

        if ret == placeholder_answer:
            continue

        tasks.append(asyncio.create_task(judge_result_by_llm(query, ret)))
        batch_ids.append(id)
        queries.append(query)

        max_concurrency -= 1

        if max_concurrency == 0 or id == end_id:
            batch_results = await asyncio.gather(*tasks)
            for id, query, (reason, score, _) in zip(batch_ids, queries, batch_results):
                results.append(
                    {
                        "id": id,
                        "query": query,
                        "reason": reason,
                        "score": score,
                    }
                )
            tasks = []
            batch_ids = []
            queries = []
            max_concurrency = 8

            with open(judge_save_file, "w") as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # 修改文件名
    asyncio.run(
        judge_all_result(
            "submission/submit_all_top_k=19_use_query_cls=True_handbook_knowledge_bank_test_0.83875.csv",
            "data/question_public.csv",
        )
    )
    # res = asyncio.run(
    #     judge_general_answer_by_llm(
    #         "请问你们家的商品支持7天无理由退换货吗？",
    #         "您好，本店所有商品均支持7天无理由退换货服务。只要您收到的商品保持原样，在不影响二次销售的情况下，即商品未拆封使用、吊牌齐全且外包装完好，您可以在签收之日起的7天内随时申请退换货。您只需在订单后台提交申请，我们会第一时间为您审核处理，确保您的购物过程无后顾之忧。",
    #     )
    # )
    # print(res)
