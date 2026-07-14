import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tracemind.config import get_config

load_dotenv()

generate_handbook_name_prompt_template = """
# Role
你是一个专业的手册名称生成专家，你的任务是根据文本描述生成一个英文的手册名称。

# Task
我会提供一段文本描述，这个文本描述是某个产品手册，你需要根据文本的语义信息，生成一个手册名称。

# TIPS
文本描述中有图像的文件名，你可以根据图像的文件名来辅助你进行判断，同时需要注意图像名称中包含`Manual`的图片，这类图片是手动添加的，不能辅助你判断

# Constraints
手册名称中不能包含`'`,`"`等中英文引号

# 输出格式
你需要以json的格式输出手册名称，json的结构如下：
{
    "handbook_name": str, # 英文手册名称,应该为xx产品用户手册，例如"xxx_user_manual"
}
# 文本描述如下:
{{description}}
"""


def generate_english_handbook_name(description: str) -> str:
    """
    英文手册没有手册名称，该函数根据英文手册的内容生成对应的手册名称，如catalog/handbook_name_gemini.json中的内容
    """
    prompt = PromptTemplate.from_template(
        generate_handbook_name_prompt_template, template_format="mustache"
    )
    # llm = ChatOpenAI(
    #     model="gpt-5.4",
    #     base_url=os.getenv("OPEANAI_BASE_URL"),
    #     api_key=os.getenv("OPEANAI_API_KEY"),
    # )
    # llm = ChatOpenAI(
    #     model="deepseek-v4-pro",
    #     base_url=os.getenv("DEEPSEEK_BASE_URL"),
    #     api_key=os.getenv("DEEPSEEK_API_KEY"),
    # )
    # gemini的效果会好一丢丢
    llm = ChatOpenAI(
        # model="gemini-3.1-pro-preview",
        model="gemini-3-flash-preview",
        base_url=os.getenv("GEMINI_BASE_URL"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    chain = prompt | llm | JsonOutputParser()
    result = chain.with_retry().invoke({"description": description})
    return result["handbook_name"]


if __name__ == "__main__":
    english_handbook_file = str(Path(get_config()["DATA_ROOT_DIR"]) / "汇总英文手册.txt")
    with open(english_handbook_file, "r", encoding="utf-8") as f:
        contents = f.readlines()
    # 修改保存的文件名称
    handbook_name_file = "catalog/handbook_name_gemini.json"
    handbook_names = []
    for i, content in enumerate(contents):
        data = eval(content)
        description = data[0]
        image_path_list = data[1]
        handbook_name = generate_english_handbook_name(data)
        handbook_names.append(handbook_name)
    with open(handbook_name_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(handbook_names))
