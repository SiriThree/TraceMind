import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def get_all_source(language: Literal["chinese", "english"]) -> list[str]:
    """
    根据输入的语言返回所有手册的文件名
    """
    data_dir = os.getenv("DATA_ROOT_DIR", "data/KownledgeBase/手册")
    data_dir_path = Path(data_dir)
    source_list = data_dir_path.glob("*.txt")
    chinese_sources = [
        str(txt_path.name)
        for txt_path in source_list
        if str(txt_path.name) != "汇总英文手册.txt"
    ]
    with open(
        os.getenv("ENGLISH_HANDBOOK_NAME_FILE", "handbook_name.json"),
        "r",
        encoding="utf-8",
    ) as f:
        english_handbook_names = json.load(f)
        english_sources = [
            f"{english_handbook_name}.txt"
            for english_handbook_name in english_handbook_names
        ]
    if language == "chinese":
        return chinese_sources
    elif language == "english":
        return english_sources
    else:
        raise ValueError(f"Invalid language: {language}")


# 使用LLM进行文档预处理的提示词
doc_parse_xml_prompt_template = """
# Role
你是一位专业的文档结构解析专家。你的任务是深度理解文本内容的语义逻辑，修复并重构 Markdown 文本中错乱的标题层级（Heading levels）。

# Task
我会提供一段文本描述，其中包含了 Markdown 标题标记（如 #, ## 等），但它们的层级关系可能是不正确或混乱的。你需要通过分析段落之间的包含、并列或递进关系，将其恢复到正确的结构。

# Constraints (严格遵守)
1. **内容绝对锁定**：你不需要改变文本的内容，你只需要将该文本描述恢复其原本正确的标题层级，例如#表示一级标题，##表示二级标题，###表示三级标题，以此类推。
2. **标签原样保留**：文本中出现的 `<PIC>` 标签或其他占位符必须原封不动地保留在原位，将其视为普通文本处理。
3. **层级逻辑准则**：
   - 保持层级连贯，避免出现不符合逻辑的断层（例如直接从 # 跳到 ### 且中间无 ##）。
4. 你可以添加一些换行符来让格式更美观   
5. **纯净输出**：仅输出符合下方规范的 JSON 字符串。严禁在输出中包含任何多余的解释性文字，严禁使用 ```json 和 ``` 等 Markdown 代码块标记包裹。

# Output Format
你需要以xml的格式输出结果,例如
<formatted_description>修复后的完整 Markdown 文本内容</formatted_description>

# 文本描述如下:
{{description}}
"""

# 使用LLM生成图片描述的提示词
generate_image_description_prompt_template = """
## 任务
我会给你一段图片的上下文，图片的上下文是文件`{{file_name}}`中的部分片段，<PIC>占位符是对应的图片的位置,你需要根据图片的上下文信息以及图片的内容生成一段图片的描述,使其替换掉<PIC>时可以较好地融入原文，不会突兀。

## 约束
1. 你不需要生成多余的其他无用、无关的内容，只需要生成对应的图片描述即可。
2. 你的描述需要是`{{language}}`语言的。

##输出格式
你需要以xml的格式输出结果,例如
<image_description>图像的描述</image_description>

## 图片的上下文
<image-context>
{{image_context}}
</image-context>
"""

generate_image_description_prompt_template_without_context = """
## 任务
我会给你一张图片，请你生成这张图片的描述。

## 约束
1. 你不需要生成多余的其他无用、无关的内容，只需要生成对应的图片描述即可。
2. 你的描述需要是`{{language}}`语言的。

##输出格式
你需要以xml的格式输出结果,例如
<image_description>图像的描述</image_description>
"""
