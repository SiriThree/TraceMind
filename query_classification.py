import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm

from config import get_config
from prompts import get_all_source
from utils import language_detect

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)
DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "db_name": os.getenv("MILVUS_DB_NAME", "default"),
}


query_classification_model = ChatOpenAI(
    model="gpt-5.4",
    base_url=os.getenv("OPEANAI_BASE_URL"),
    api_key=os.getenv("OPEANAI_API_KEY"),
)


def get_query_classification_prompt(language: Literal["chinese", "english"]) -> str:
    """
    提示词
    让LLM根据提供的手册名称，以及问题，判断该问题的类型，如果为product类型，则还需要判断该问题可以在哪个文档中找到答案
    """
    #     return (
    #         f"""
    # # Role
    # 你是一个专业的查询分类专家，你的任务是根据用户的问题，按照我的要求对用户的问题进行分类。

    # # Task
    # 我会提供一段用户的问题，你需要根据用户的问题，对用户的问题进行分类。
    # 1. 判断用户的问题是否和某个产品相关,是否可以在以下的某个手册中找到。
    #   目前的手册有: {get_all_source(language)}"""
    #         + """
    # # 输出的格式
    # 请严格按照以下 JSON 结构输出（注意转义文本中的特殊字符以保证 JSON 的合法性）：
    # {"""
    #         + f"""source": Literal{[*get_all_source(language)]} | None # 判断用户的问题可能能在哪个手册中找到答案，如果用户的问题是通用性的问题，和某个产品无关，不能在所给的手册中找到相关的答案，则为None
    #         """
    #         + "question_type: Literal[product, general] # 如果source 不为None,则question_type=product,如果source为None,则你需要判断用户的问题是否是通识性问题（例如物流快递、投诉、发票、退货退款、维修、售后等问题），还是某个产品相关的问题，如果是通识性问题，则question_type=general,如果是产品相关问题，则question_type=product"
    #         + """
    # }
    # # 用户的问题:
    # <user-query>
    # {{query}}
    # </user-query>
    # """
    #     )
    return (
        f"""# Role
你是一个专业的查询分类专家，你的任务是根据用户的问题，按照我的要求对用户的问题进行分类。


# Task
我会提供一段用户的问题，你需要根据用户的问题，对用户的问题进行分类。
1. 判断用户的问题是否和某个产品相关,是否可以在以下的某个手册中找到。
目前的手册有: {get_all_source(language)}"""
        + """
# 输出的格式
请严格按照以下 JSON 结构输出（注意转义文本中的特殊字符以保证 JSON 的合法性）：
{\n\t"""
        + f"""'source': Literal{[*get_all_source(language)]} | None # 判断用户的问题可能能在哪个手册中找到答案，如果用户的问题是通用性的问题，和某个产品无关，不能在所给的手册中找到相关的答案，则为None。\n\t"""
        + "'question_type': Literal[product, general] # 如果source 不为None,则question_type=product,如果source为None,则你需要判断用户的问题是否是通识性问题（例如物流快递、投诉、发票、退货退款、维修、售后等问题），还是某个产品相关的问题，如果是通识性问题，则question_type=general,如果是产品相关问题，则question_type=product。\n\t"
        + "'source_confidence': float|None #source的置信度，如果source不为None，则你需要判断这个问题是与选择的source相关的置信度，范围在[0, 1]之间,如果source为None,则source_confidence为None。\n\t"
        + "'question_confidence': float|None # 置信度，如果source为None并且question_type=general,则你需要判断这个问题属于通识性问题的置信度，范围在[0, 1]之间,其他情况则置信度为None。"
        + """
}
# 用户的问题:
<user-query>
{{query}}
</user-query>
"""
    )


async def query_classification_via_handbook_name(query: str) -> dict[str, str]:
    """
    查询分类
    让LLM根据提供的手册名称，以及问题，判断该问题的类型，如果为product类型，则还需要判断该问题可以在哪个文档中找到答案，以及需要输出对应的置信度
    output schema:
    {
        source: str| None
        question_type: Literal["product", "general"]
        source_confidence: float | None
        question_confidence: float | None
    }
    """
    language = language_detect(query)
    query_classification_prompt = PromptTemplate.from_template(
        get_query_classification_prompt(language), template_format="mustache"
    )
    query_classification_chain = (
        query_classification_prompt | query_classification_model
    ) | JsonOutputParser()
    query_classification_result = await query_classification_chain.with_retry(
        stop_after_attempt=5
    ).ainvoke({"query": query})
    return {"language": language, **query_classification_result}


def get_query_classification_toc_prompt(language: Literal["chinese", "english"]) -> str:
    """
    提示词
    让LLM根据提供的手册名称和目录内容，判断该问题可以在哪个手册中找到答案,
    """
    handbook_catalog_path = (
        Path("catalog/chinese_handbook_catalog.json")
        if language == "chinese"
        else Path("catalog/english_handbook_catalog.json")
    )
    handbook_catalog = json.load(open(handbook_catalog_path, "r", encoding="utf-8"))
    return (
        """# Role
你是一个专业的查询分类专家，你的任务是根据用户的问题,判断该问题可以在哪个手册中找到答案

# Task
我会提供用户的问题以及所有的手册名称和其目录内容，你需要根据我提供的内容，判断用户的问题可以在那个手册中找到答案


# 所有的手册名称以及对应的大纲内容
"""
        + f"\n{handbook_catalog}"
        + """
# TIPS
`handbook_name`字段为手册名称，`catalog`为手册的目录内容

# 输出格式
请严格按照以下 JSON 结构输出（注意转义文本中的特殊字符以保证 JSON 的合法性）：
{
    "source": str | None, # 手册名称，如果你觉得用户的问题可以在某个手册中找到答案，则为该手册名称，否则为None,
    "source_confidence": float | None, # source的置信度，如果source不为None，则你需要判断这个问题是与选择的source相关的置信度，范围在[0, 1]之间,如果source为None,则source_confidence为None
}

# 用户的问题:
<user-query>
{{query}}
</user-query>
"""
    )


async def query_classification_via_toc(query: str) -> dict[str, str]:
    """
    查询分类
    让LLM根据提供的手册名称和目录内容，判断该问题可以在哪个手册中找到答案,
    output schema:
    {
        source: str| None
        source_confidence: float | None
    }
    """
    language = language_detect(query)
    query_classification_toc_prompt = PromptTemplate.from_template(
        get_query_classification_toc_prompt(language), template_format="mustache"
    )

    query_classification_chain = (
        query_classification_toc_prompt | query_classification_model
    ) | JsonOutputParser()
    query_classification_result = await query_classification_chain.with_retry(
        stop_after_attempt=5
    ).ainvoke({"query": query})
    return {"language": language, **query_classification_result}


async def get_source_by_dense_store(query: str) -> dict[str, str]:
    """
    使用语义相似度搜索出top10chunk，根据chunk对应的文档的数量来预测该问题可以在哪个手册中找到答案
    """
    dense_store = Milvus(
        embedding_function=embedding_model,
        collection_name=get_config()["MILVUS_COLLECTION_NAME"],
        text_field="text",
        vector_field="dense",
        auto_id=True,
        drop_old=False,
        enable_dynamic_field=True,
        connection_args=DEFAULT_MILVUS_CONNECTION,
        index_params=[
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
            },
        ],
    )

    top_k = 10
    language = language_detect(query)
    dense_result = await dense_store.asimilarity_search(
        query,
        k=top_k,
        fetch_k=top_k,
        expr=f"language == '{language}'",
    )
    source_counter = Counter([result.metadata["source"] for result in dense_result])
    dense_predict_source = source_counter.most_common(1)[0][0]
    return {"dense_predict_source": dense_predict_source}


async def ensembles_query_classification(query: str) -> dict[str, str | bool]:
    """
    集成了query_classification_via_handbook_name、query_classification_via_toc、get_source_by_dense_store来对query进行一个分类
    具体的流程可看assest/查询分类流程.png
    """
    language = language_detect(query)
    query_classification_result_via_handbook_name = (
        await query_classification_via_handbook_name(query)
    )
    question_type = query_classification_result_via_handbook_name["question_type"]
    question_confidence_via_handbook_name = (
        query_classification_result_via_handbook_name["question_confidence"]
    )
    source_via_handbook_name = query_classification_result_via_handbook_name["source"]
    source_confidence_via_handbook_name = query_classification_result_via_handbook_name[
        "source_confidence"
    ]
    final_source = None
    final_question_type = None
    only_llm_predict_once = False
    if question_type == "general":
        if question_confidence_via_handbook_name < 0.98:
            query_classification_result_via_toc = await query_classification_via_toc(
                query
            )
            if query_classification_result_via_toc["source"] is not None:
                final_source = query_classification_result_via_toc["source"]
                final_question_type = "product"
            else:
                final_source = None
                final_question_type = question_type
        else:
            final_source = None
            final_question_type = question_type
    else:
        # question_type == "product"
        final_question_type = question_type
        # 如果handbook_name 没有source，则取toc方式预测出的source
        if source_via_handbook_name is None:
            query_classification_result_via_toc = await query_classification_via_toc(
                query
            )
            final_source = query_classification_result_via_toc["source"]
        else:
            # 如果LLM根据handbook_name 预测出了source，则需要判断置信度
            if source_confidence_via_handbook_name >= 0.9:
                final_source = source_via_handbook_name
                only_llm_predict_once = True
            else:
                tasks = [
                    asyncio.create_task(query_classification_via_toc(query)),
                    asyncio.create_task(get_source_by_dense_store(query)),
                ]
                (
                    query_classification_result_via_toc,
                    dense_predict_results,
                ) = await asyncio.gather(*tasks)
                dense_predict_source = dense_predict_results["dense_predict_source"]
                source_via_toc = query_classification_result_via_toc["source"]
                source_confidence_via_toc = query_classification_result_via_toc[
                    "source_confidence"
                ]
                # 如果有两个source 相等，则取这个为final_source
                if (
                    source_via_handbook_name == source_via_toc
                    or source_via_handbook_name == dense_predict_source
                    or source_via_toc == dense_predict_source
                ):
                    if source_via_handbook_name == source_via_toc:
                        final_source = source_via_handbook_name
                    elif source_via_handbook_name == dense_predict_source:
                        final_source = source_via_handbook_name
                    elif source_via_toc == dense_predict_source:
                        final_source = source_via_toc
                else:
                    # 三个答案都不同
                    final_source = (
                        source_via_toc
                        if source_confidence_via_toc
                        >= source_confidence_via_handbook_name
                        else source_via_handbook_name
                    )
    return {
        "source_via_handbook_name": source_via_handbook_name,
        "source": final_source,
        "question_type": final_question_type,
        "language": language,
        "only_llm_predict_once": only_llm_predict_once,
    }


async def test_query_classification_via_toc():
    """
    测试query_classification_via_toc的效果
    """
    results = []
    error_results = []
    exist_last_id = -1
    max_concurrency = 8
    tasks = []
    batch_ids = []
    queries = []
    end_id = 436
    question_file = Path("data/question_public.csv")
    query_classification_file = Path("test_query_classification_via_toc.json")
    if query_classification_file.exists():
        results = json.load(open(query_classification_file, "r"))["results"]
        error_results = json.load(open(query_classification_file, "r"))["error_results"]
        exist_last_id = max([result["id"] for result in results])
    question_df = pd.read_csv(question_file, index_col="id")
    for question_row in tqdm(question_df.iterrows(), total=len(question_df)):
        id = question_row[0]
        query = question_row[1]["question"].strip('"')
        if id <= exist_last_id:
            continue
        if id <= end_id:
            tasks.append(asyncio.create_task(query_classification_via_toc(query)))
            batch_ids.append(id)
            queries.append(query)
            max_concurrency -= 1

        if max_concurrency == 0 or id == end_id:
            query_classification_results = await asyncio.gather(*tasks)
            for id, query, query_classification_result in zip(
                batch_ids, queries, query_classification_results
            ):
                query_classification_result["id"] = id
                query_classification_result["query"] = query
                results.append(query_classification_result)
                with open(query_classification_file, "w") as f:
                    json.dump(
                        {"results": results, "error_results": error_results},
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
                batch_ids = []
                tasks = []
                queries = []
                max_concurrency = 8


async def test_ensembles_query_classification():
    """
    测试ensembles_query_classification的效果
    """
    results = []
    exist_last_id = -1
    max_concurrency = 8
    tasks = []
    batch_ids = []
    queries = []
    start_id = 64
    end_id = 436
    question_file = Path("data/question_public.csv")
    query_classification_file = Path(
        "experiment/查询分类/query_classification_ratio1.json"
    )
    exist_last_id = start_id - 1
    if query_classification_file.exists():
        results = json.load(open(query_classification_file, "r"))
        exist_last_id = max([result["id"] for result in results])

    question_df = pd.read_csv(question_file, index_col="id")
    for question_row in tqdm(question_df.iterrows(), total=len(question_df)):
        id = question_row[0]
        query = question_row[1]["question"].strip('"')
        if id <= exist_last_id:
            continue
        if id <= end_id:
            tasks.append(asyncio.create_task(ensembles_query_classification(query)))
            batch_ids.append(id)
            queries.append(query)
            max_concurrency -= 1

        if max_concurrency == 0 or id == end_id:
            query_classification_results = await asyncio.gather(*tasks)
            for id, query, query_classification_result in zip(
                batch_ids, queries, query_classification_results
            ):
                query_classification_result["id"] = id
                query_classification_result["query"] = query
                results.append(query_classification_result)
                with open(query_classification_file, "w") as f:
                    json.dump(
                        results,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
                batch_ids = []
                tasks = []
                queries = []
                max_concurrency = 8


if __name__ == "__main__":
    # query = "请问你们家的商品支持7天无理由退换货吗？"

    # query = "为保持遮光罩的清晰度和功能性，正确的清洁步骤是什么？"
    # query = "How to change the default setting of the energy saving mode"
    # query = "What is the trip screen shown?"
    # query = "What is the proper way to use the brake lever and brake button?"
    # query = " What are the best methods for memorizing communication channels using a manual program?"
    #
    # query = "我想了解一下你们的退款政策，退款多久能到账？"
    # query_classification_result = query_classification(query)

    # query = "设备或系统中构成处理器单元的关键组件或部件有哪些？"
    # query = (
    #     "If I want to listen to music on my phone, how do I turn on the sound system?"
    # )
    # query = "What are the steps to power the camera?"
    # query_classification_result = asyncio.run(ensembles_query_classification(query))
    asyncio.run(test_ensembles_query_classification())
