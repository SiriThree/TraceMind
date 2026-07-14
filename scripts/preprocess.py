import html
import logging
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tracemind.config import get_config
from tracemind.prompts import (
    doc_parse_xml_prompt_template,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class FormatException(Exception):
    pass


def parse_xml_format_description(content: str) -> str:
    """
    从LLM生成的XML结果中中提取预处理后的描述
    """
    match = re.search(
        r"<formatted_description>(.*?)</formatted_description>", content, re.DOTALL
    )
    if match:
        description = match.group(1).strip()
        return description
    else:
        logger.error("XML中没有找到formatted_description标签")
        raise FormatException("XML中没有找到formatted_description标签")


def ensure_format_correct(origin_description: str, formatted_description: str) -> str:
    """
    确保使用LLM预处理之后的内容格式正确
    1. 确保<PIC>的数量与原文一致
    2. 确保使用LLM预处理不会改变原本的文本内容，只修改标题的层级

    """
    formatted_description = html.unescape(formatted_description)
    # 找到<PIC>的数量
    origin_pic_count = origin_description.count("<PIC>")
    formatted_pic_count = formatted_description.count("<PIC>")
    if origin_pic_count != formatted_pic_count:
        logger.error(
            f"格式化后的文本中<PIC>的数量不正确，原文中有{origin_pic_count}个<PIC>，格式化后的文本中有{formatted_pic_count}个<PIC>"
        )
        raise FormatException(
            f"格式化后的文本中<PIC>的数量不正确，原文中有{origin_pic_count}个<PIC>，格式化后的文本中有{formatted_pic_count}个<PIC>"
        )
    test_count = 10
    while test_count > 0:
        length = random.randint(8, 20)
        start = random.randint(0, len(origin_description) - length)
        snippet = origin_description[start : start + length].strip()

        if (
            "\n" in snippet
            or "#" in snippet
            or "<" in snippet
            or ">" in snippet
            or "•" in snippet
            or " " in snippet
            or "：" in snippet
            or ":" in snippet
            or "・" in snippet
        ):
            continue
        else:
            # TODO:使用LCS的距离进行判断
            if (
                snippet not in formatted_description
                and snippet not in formatted_description.replace("\n", "")
                and snippet
                not in formatted_description.replace("\n", "").replace("#", "")
            ):
                logger.error(f"格式化后的文本中缺少原文中的片段:{snippet}")
                raise FormatException(f"格式化后的文本中缺少原文中的片段: {snippet}")
            else:
                test_count -= 1

    return formatted_description


def preprocess_handbook(content: str) -> tuple[str, str, list[str]]:
    """
    对单个的手册内容进行预处理
    """
    data = eval(content.strip())
    description = data[0]
    image_path_list = data[1]
    prompt = PromptTemplate.from_template(
        doc_parse_xml_prompt_template, template_format="mustache"
    )

    # llm = ChatOpenAI(
    #     model="gpt-5.5",
    #     base_url=os.getenv("OPEANAI_BASE_URL"),
    #     api_key=os.getenv("OPEANAI_API_KEY"),
    #     # default_headers = {"User-Agent": user_agent}
    # )
    llm = ChatOpenAI(
        model="gemini-2.5-pro",
        base_url=os.getenv("GEMINI_BASE_URL"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # llm  = ChatOpenAI(
    #     model="deepseek-v4-pro",
    #     base_url=os.getenv("DEEPSEEK_BASE_URL"),
    #     api_key=os.getenv("DEEPSEEK_API_KEY"),
    # )

    chain = (
        prompt
        | llm
        | StrOutputParser()
        | parse_xml_format_description
        | (lambda res: ensure_format_correct(description, res))
    )
    formatted_description = chain.with_retry(
        stop_after_attempt = 5,
        retry_if_exception_type=(Exception, FormatException)
    ).invoke({"description": description})
    # 处理之后的手册内容，没处理的手册内容，图片列表
    return formatted_description, description, image_path_list


def preprocess_all_handbook(
    handbook_dir: str,
    processed_dir: str,
):
    """
    对所有的手册进行预处理
    """
    handbook_dir_path = Path(handbook_dir)

    for file_path in handbook_dir_path.glob("*.txt"):
        formatted_description_txt_path = Path(
            processed_dir, file_path.stem + "_formatted.txt"
        )

        if formatted_description_txt_path.exists():
            logger.info(
                f"{formatted_description_txt_path} already exists, skip processing"
            )
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.readlines()
        logging.info(f"processing {file_path}")
        if len(contents) == 1:
            # 中文手册
            formatted_description, origin_description, image_path_list = (
                preprocess_handbook(contents[0])
            )
            formatted_description_md_path = Path(
                processed_dir, file_path.stem + "_formatted.md"
            )
            origin_description_md_path = Path(
                processed_dir, file_path.stem + "_origin.md"
            )
            with open(formatted_description_md_path, "w", encoding="utf-8") as f:
                f.write(formatted_description)

            with open(origin_description_md_path, "w", encoding="utf-8") as f:
                f.write(origin_description)
            with open(formatted_description_txt_path, "w", encoding="utf-8") as f:
                formatted_txt_data = str([formatted_description, image_path_list])
                f.write(formatted_txt_data)

        elif len(contents) > 1:
            # 英文手册
            for i, content in enumerate(contents):
                formatted_description_md_path = Path(
                    processed_dir, file_path.stem + f"_formatted_{i + 1}.md"
                )
                origin_description_md_path = Path(
                    processed_dir, file_path.stem + f"_origin_{i + 1}.md"
                )
                formatted_description_txt_path = Path(
                    processed_dir, file_path.stem + f"_formatted_{i + 1}.txt"
                )
                if formatted_description_txt_path.exists():
                    logger.info(
                        f"{formatted_description_txt_path} already exists, skip processing"
                    )
                    continue
                formatted_description, origin_description, image_path_list = (
                    preprocess_handbook(content)
                )
                with open(formatted_description_md_path, "w", encoding="utf-8") as f:
                    f.write(formatted_description)
                with open(origin_description_md_path, "w", encoding="utf-8") as f:
                    f.write(origin_description)
                with open(formatted_description_txt_path, "w", encoding="utf-8") as f:
                    formatted_txt_data = str([formatted_description, image_path_list])
                    f.write(formatted_txt_data)
                logger.info(
                    f"{file_path} is processed , current line {i + 1},total {len(contents)}"
                )
        else:
            raise Exception(f"{file_path} is empty")
        logger.info(f"{file_path} is processed")


if __name__ == "__main__":
    # origin_file_path = "data/KnowledgeBase/手册/冰箱手册.txt"
    # with open(origin_file_path, "r", encoding="utf-8") as f:
    #     content = f.read()
    # formatted_description, origin_description, image_path_list = format_data(content)
    # print(formatted_description)
    preprocess_all_handbook(
        get_config()["DATA_ROOT_DIR"], get_config()["PROCESSED_DATA_ROOT_DIR"]
    )
