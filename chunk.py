import json
import logging
import os
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
)
from tqdm import tqdm

from config import get_config
from prompts import (
    generate_image_description_prompt_template,
    generate_image_description_prompt_template_without_context,
)
from utils import encode_image, get_image_name, language_detect

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# openai的embedding模型效果更好
embedding_model = OpenAIEmbeddings(
    # model="qwen3-embedding-8b",
    model="text-embedding-3-large",
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)

DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "db_name": os.getenv("MILVUS_DB_NAME", "default"),
}
# 使用milvus的BM25的分词器，针对不同的语言使用不同的分词方式
multi_analyzer_params = {
    "analyzers": {
        "english": {"type": "english"},
        "chinese": {"type": "chinese"},
        "default": {"tokenizer": "icu"},  # Required fallback analyzer
    },
    "by_field": "language",
}
milvus = Milvus(
    embedding_function=embedding_model,
    collection_name=get_config()["MILVUS_COLLECTION_NAME"],
    connection_args=DEFAULT_MILVUS_CONNECTION,
    auto_id=True,
    drop_old=False,
    enable_dynamic_field=False,
    vector_field=["dense", "sparse"],
    index_params=[
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 16,
                "efConstruction": 64,
            },
        },
        {
            "index_type": "AUTOINDEX",
            "metric_type": "BM25",
            "params": {},
        },
    ],
    builtin_function=BM25BuiltInFunction(
        input_field_names="text",
        output_field_names="sparse",
        multi_analyzer_params=multi_analyzer_params,
    ),
)


def generate_description_context(content: str) -> str:
    """
    生成图片的描述文上下文：当前chunk下，该图片前面的所有内容+'<PIC>'+图片后面的文本内容直到遇到下一张图片
    """
    parts = content.split("<PIC>")
    if len(parts) <= 2:
        return content
    return "<PIC>".join(parts[:2]) + "".join(parts[2:])


def parse_image_description_tag(description: str) -> str:
    """
    解析LLM输出的xml结果，提取图片的描述
    """
    soup = BeautifulSoup(description, "xml")
    tag = soup.find("image_description")
    if tag:
        return tag.text
    raise Exception("图片描述标签不存在")


def generate_image_description(
    llm: ChatOpenAI,
    image_context: str,
    image_name: str | Path,
    file_name: str,
    use_contextual_augmentation: bool,
    language: Literal["chinese", "english"],
) -> str:
    """
    生成图片的描述
    :param image_context: 图片的上下文，通过generate_description_context函数得到
    :param image_name: 图片的名称，例如air_fryer_01.png
    :param file_name: 文件名,例如 冰箱手册.txt
    :param language: 需要生成中文还是英文的图片描述
    :return: 图片的描述
    """
    image_root_dir = Path(os.getenv("IMAGE_ROOT_DIR", "data/KownledgeBase/手册/插图"))
    image_path: Path = Path(image_root_dir, image_name)
    if not image_path.exists():
        raise FileNotFoundError(f"图片路径不存在：{image_path}")
    image_base64, mime_type = encode_image(image_path)
    if use_contextual_augmentation:
        messages = [
            {
                "role": "human",
                "content": [
                    {
                        "type": "image",
                        "base64": image_base64,
                        "mime_type": mime_type,
                    },
                    {
                        "type": "text",
                        "text": generate_image_description_prompt_template,
                    },
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "human",
                "content": [
                    {
                        "type": "image",
                        "base64": image_base64,
                        "mime_type": mime_type,
                    },
                    {
                        "type": "text",
                        "text": generate_image_description_prompt_template_without_context,
                    },
                ],
            }
        ]

    prompt = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    chain = prompt | llm | StrOutputParser() | parse_image_description_tag
    if use_contextual_augmentation:
        image_description = chain.with_retry(stop_after_attempt=10).invoke(
            {
                "file_name": file_name,
                "image_context": image_context,
                "language": language,
            }
        )
    else:
        image_description = chain.with_retry(stop_after_attempt=10).invoke(
            {"file_name": file_name, "language": language}
        )

    if language != language_detect(image_description):
        raise Exception("图片描述的语言与手册的语言不一致")

    return image_description


def chunk_and_add_document(
    file_path: str,
    language: Literal["chinese", "english"],
    source: str,
    use_contextual_augmentation: bool,
    is_save_to_local: bool = False,
):
    """
    对手册进行chunk化，并将其添加到milvus中，具体流程可看assest/知识库的构建.png
    Args:
        file_path (str): 手册的文件路径,例如processed_data/KownledgeBase/手册/冰箱手册_formatted.txt
        language (Literal["chinese", "english"]): 手册的语言，chinese或english
        source (str): 手册名称，例如冰箱手册.txt 或者 Espresso Machine User Manual.txt
        is_save_to_local (bool, optional): 是否将chunk的内容保存到本地，默认保存到knowledge_bank下，可以用来复现处理结果。查看哪里chunk的有问题，默认False

    Raises:
        ValueError: _description_
        ValueError: _description_
    """
    txt_file_path = Path(file_path)
    file_name = txt_file_path.stem.split("_")[0] + ".txt"

    content = txt_file_path.read_text()
    data = eval(content)
    description = data[0]
    image_list = data[1]

    pic_length = description.count("<PIC>")
    assert pic_length == len(image_list), (
        f"{file_path}中的描述的图片数量与图片列表数量不一致"
    )

    splitter = MarkdownHeaderTextSplitter(
        [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")],
        strip_headers=False,
        return_each_line=False,
    )
    doc_list = splitter.split_text(description)
    global_image_index = 0
    description_file_path = None
    if is_save_to_local:
        description_file_path = (
            Path("knowledge_bank", txt_file_path.name).with_suffix(".md")
            if use_contextual_augmentation
            else Path("knowledge_bank_without_context", txt_file_path.name).with_suffix(
                ".md"
            )
        )
    final_insert_doc_list: list[Document] = []
    final_save_content: str = ""

    for index, doc in tqdm(
        enumerate(doc_list), total=len(doc_list), desc=f"正在处理{txt_file_path.name}"
    ):
        doc.metadata["source"] = source
        doc.metadata["index"] = index
        doc.metadata["language"] = language

        # TODO: title中包含有<PIC>的，需要对这个document进行特殊处理，将<PIC>转到page_content中
        lines = doc.page_content.split("\n")
        if lines[0].startswith("#"):
            doc.page_content = "\n".join(lines[1:])

        # title 的内容添加到page-content中
        for i in reversed(range(1, 4)):
            header_name = f"Header_{i}"
            if header_name in doc.metadata:
                title_level = i
                if "<PIC>" in doc.metadata[header_name]:
                    # 需要手动处理一下，保证title中不包含<PIC>标签
                    raise ValueError(
                        f"标题{header_name}中包含有<PIC>标签,不能直接添加到page-content中"
                    )

                doc.page_content = (
                    f"{'#' * int(title_level)} {doc.metadata[header_name]}\n"
                    + doc.page_content
                )
                # 删除掉metadata中的标题字段
                doc.metadata.pop(header_name)

        # 处理<PIC>标签,将其转化为自然语言描述,将文件名、标题、图片相关的上下文都给到LLM ,生成图片的描述并使用描述替换原来的PIC标签
        cur_chunk_pic_length = doc.page_content.count("<PIC>")
        page_content = doc.page_content
        for _ in range(cur_chunk_pic_length):
            image_context = generate_description_context(page_content)
            # 带有扩展名的图像名称
            image_name = get_image_name(image_list[global_image_index])
            # llm = ChatOpenAI(
            #     model="gpt-5.5",
            #     base_url=os.getenv("OPEANAI_BASE_URL"),
            #     api_key=os.getenv("OPEANAI_API_KEY"),
            # )
            # gemini的多模态效果会更好一点
            llm = ChatOpenAI(
                # model="gemini-3.1-pro-preview",
                model="gemini-3-flash-preview",
                base_url=os.getenv("GEMINI_BASE_URL"),
                api_key=os.getenv("GEMINI_API_KEY"),
            )

            image_description = generate_image_description(
                llm,
                image_context,
                image_name,
                file_name,
                use_contextual_augmentation,
                language,
            )
            image_description_results = []
            if (
                get_config()["IMAGE_DESCRIPTION_RESULTS_JSON_FILE"] is not None
                and Path(get_config()["IMAGE_DESCRIPTION_RESULTS_JSON_FILE"]).exists()
            ):
                image_description_results = json.load(
                    open(
                        get_config()["IMAGE_DESCRIPTION_RESULTS_JSON_FILE"],
                        "r",
                        encoding="utf-8",
                    )
                )
            if get_config()["IMAGE_DESCRIPTION_RESULTS_JSON_FILE"] is not None:
                image_description_results.append(
                    {
                        "file_name": file_name,
                        "image_name": image_name,
                        "image_description": image_description,
                    }
                )
                with open(
                    get_config()["IMAGE_DESCRIPTION_RESULTS_JSON_FILE"],
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        image_description_results,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )

            if "<PIC>" in image_description:
                raise ValueError("生成的图片描述中包含有<PIC>标签")
            page_content = page_content.replace(
                "<PIC>",
                f"<picture-description image_name='{image_list[global_image_index]}'>{image_description}</picture-description>\n",
                1,
            )
            global_image_index += 1

        doc.page_content = page_content
        final_insert_doc_list.append(doc)
        final_save_content += page_content

    assert global_image_index == len(image_list), (
        f"{file_path}中的描述的图片数量与图片列表数量不一致"
    )
    if is_save_to_local and description_file_path is not None:
        with open(description_file_path, "w", encoding="utf-8") as f:
            f.write(final_save_content)
            logger.info(
                f"已将文件{file_path}经过切片之后的结果保存到{description_file_path}"
            )
    # 最后将所有chunk一起入库
    milvus.add_documents(final_insert_doc_list)


if __name__ == "__main__":
    # 配置processed_dir
    processed_dir = Path("processed_data/KownledgeBase/手册")
    
    config = get_config()
    collection_name = config["MILVUS_COLLECTION_NAME"]
    use_contextual_augmentation = config["USE_CONTEXTUAL_AUGMENTATION"]

    english_handbook_names = []
    
    language = None
    with open(
        os.getenv("ENGLISH_HANDBOOK_NAME_FILE", "handbook_names.json"),
        "r",
        encoding="utf-8",
    ) as f:
        english_handbook_names = json.load(f)

    for file_path in processed_dir.glob("*.txt"):
        if "汇总英文手册" in file_path.stem:
            i = int(file_path.stem.split("_")[2])
            handbook_name = english_handbook_names[i - 1]
            language = "english"

        else:
            handbook_name = file_path.stem.split("_")[0]
            language = "chinese"
        source = handbook_name + ".txt"
        # 有问题的三个文件：发电机手册_formatted，可编程温控器手册_formatted，洗碗机手册_formatted，PIC标签数量不对
        logger.info(f"正在处理文件{file_path.name}")
        
        has_collection = milvus.client.has_collection(collection_name)

        if has_collection:
            # 如果集合中已经存在该手册的数据，则会跳过该手册
            results = milvus.client.query(
                collection_name=collection_name,
                filter=f"source == '{source}'",
                output_fields=["source"],
                limit=100,
            )
            data_count = len(results)
            if data_count > 0:
                logger.info(f"文件{file_path.name}已存在{data_count}条数据")
                continue

        chunk_and_add_document(
            str(file_path),
            language,
            source,
            use_contextual_augmentation,
            is_save_to_local=True,
        )
        logger.info(f"文件{file_path.name}处理完成")
