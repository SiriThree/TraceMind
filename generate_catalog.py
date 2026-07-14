import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from tqdm import tqdm

from utils import language_detect

load_dotenv()


def ensure_toc_language(
    toc_content: str, handbook_language: Literal["chinese", "english"]
) -> str:
    """
    确保生成的目录的语言和手册的语言一致
    """
    answer_language = language_detect(toc_content)
    if answer_language != handbook_language:
        raise Exception(f"模型回答的的语言不是{handbook_language}")

    return toc_content


def generate_catalog_or_summary(
    description: str, language: Literal["chinese", "english"]
):
    """
    根据手册内容生成目录或摘要,具体结果可看catalog目录
    """
    llm = ChatCohere(
        model="command-a-reasoning-08-2025",
        temperature=0.8,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"以下的文本是某个说明手册中的内容，如果手册中包含目录，你需要提取出目录，如果不包含目录，你需要生成一个该手册的总结,在提取目录或者生成摘要之前你需要说明这是关于什么产品的手册.使用{'中文' if language == 'chinese' else '英文'}回答",
            ),
            ("human", "手册内容如下:\n\n{{description}}"),
        ],
        template_format="mustache",
    )

    chain = (
        prompt
        | llm
        | RunnableLambda(lambda res: ensure_toc_language(res.text, language))
    )
    res = chain.with_retry(stop_after_attempt=10).invoke({"description": description})
    return res


if __name__ == "__main__":
    handbook_dir_path = Path("data/KownledgeBase/手册")
    english_gather_file = handbook_dir_path / "汇总英文手册.txt"
    english_results = []
    chinese_results = []
    english_exist_handbook_names = []
    chinese_exist_handbook_names = []
    english_handbook_name_file = os.getenv(
        "ENGLISH_HANDBOOK_NAME_FILE", "handbook_name_gemini.json"
    )

    english_catalog_file = "catalog/english_handbook_catalog.json"
    chinese_catalog_file = "catalog/chinese_handbook_catalog.json"

    if Path(chinese_catalog_file).exists():
        with open(chinese_catalog_file, "r", encoding="utf-8") as f:
            chinese_results = json.load(f)
            chinese_exist_handbook_names = [
                item["handbook_name"] for item in chinese_results
            ]

    if Path(english_catalog_file).exists():
        with open(english_catalog_file, "r", encoding="utf-8") as f:
            english_results = json.load(f)
            english_exist_handbook_names = [
                item["handbook_name"] for item in english_results
            ]

    with open(english_handbook_name_file, "r", encoding="utf-8") as f:
        english_handbook_names = json.load(f)

    for handbook_path in tqdm(
        handbook_dir_path.glob("*.txt"),
        total=len(list(handbook_dir_path.glob("*.txt"))),
        desc="处理手册",
    ):
        if handbook_path.name == "汇总英文手册.txt":
            with open(english_gather_file, "r", encoding="utf-8") as f:
                english_handbooks = f.readlines()
            for english_handbook_content, english_handbook_name in tqdm(
                zip(english_handbooks, english_handbook_names),
                total=len(english_handbooks),
                desc="处理英文手册",
            ):
                if english_handbook_name in english_exist_handbook_names:
                    continue
                data = eval(english_handbook_content)
                description, image_list = data[0], data[1]
                res = generate_catalog_or_summary(description, "english")
                english_results.append(
                    {"handbook_name": english_handbook_name + ".txt", "catalog": res}
                )
                with open(english_catalog_file, "w", encoding="utf-8") as f:
                    json.dump(
                        english_results,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )
        else:
            # 中文手册
            if handbook_path.stem in chinese_exist_handbook_names:
                continue
            with open(handbook_path, "r", encoding="utf-8") as f:
                chinese_handbook_content = f.read()
            data = eval(chinese_handbook_content)
            description, image_list = data[0], data[1]
            res = generate_catalog_or_summary(description, "chinese")
            chinese_results.append(
                {"handbook_name": handbook_path.name, "catalog": res}
            )

            with open(chinese_catalog_file, "w", encoding="utf-8") as f:
                json.dump(
                    chinese_results,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
