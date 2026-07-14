import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def get_config() -> dict:
    MILVUS_COLLECTION_NAME_DEFAULT = "handbook_knowledge_bank_test"
    data_root_dir = os.getenv(
        "DATA_ROOT_DIR",
        _first_existing_path("data/KnowledgeBase/手册", "data/KownledgeBase/手册"),
    )
    processed_data_root_dir = os.getenv(
        "PROCESSED_DATA_ROOT_DIR",
        _first_existing_path(
            "processed_data/KnowledgeBase/手册", "processed_data/KownledgeBase/手册"
        ),
    )
    image_root_dir = os.getenv(
        "IMAGE_ROOT_DIR",
        _first_existing_path(
            f"{processed_data_root_dir}/插图",
            f"{data_root_dir}/插图",
            "processed_data/KnowledgeBase/手册/插图",
            "processed_data/KownledgeBase/手册/插图",
            "data/KnowledgeBase/手册/插图",
            "data/KownledgeBase/手册/插图",
        ),
    )
    IMAGE_DESCRIPTION_RESULTS_JSON_FILE = (
        f"experiment/上下文增强的实验/{MILVUS_COLLECTION_NAME_DEFAULT}.json"
        if os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1"
        else f"experiment/上下文增强的实验/{MILVUS_COLLECTION_NAME_DEFAULT}_without_context.json"
    )
    return {
        "USE_QUERY_CLS": os.getenv("USE_QUERY_CLS") == "1",
        "USE_CONTEXTUAL_AUGMENTATION": os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1",
        "IMAGE_DESCRIPTION_RESULTS_JSON_FILE": IMAGE_DESCRIPTION_RESULTS_JSON_FILE,
        "DATA_ROOT_DIR": data_root_dir,
        "PROCESSED_DATA_ROOT_DIR": processed_data_root_dir,
        "IMAGE_ROOT_DIR": image_root_dir,
        # milvus集合的名称
        "MILVUS_COLLECTION_NAME": MILVUS_COLLECTION_NAME_DEFAULT
        if os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1"
        else f"{MILVUS_COLLECTION_NAME_DEFAULT}_without_context",
    }
