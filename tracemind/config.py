import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def get_config() -> dict:
    collection_base = "handbook_knowledge_bank_test"

    data_root_dir = _env_or_default(
        "DATA_ROOT_DIR",
        _first_existing_path(
            "data/KnowledgeBase/手册",
            "data/KownledgeBase/手册",
            "data/KnowledgeBase/鎵嬪唽",
            "data/KownledgeBase/鎵嬪唽",
        ),
    )
    processed_data_root_dir = _env_or_default(
        "PROCESSED_DATA_ROOT_DIR",
        _first_existing_path(
            "processed_data/KnowledgeBase/手册",
            "processed_data/KownledgeBase/手册",
            "processed_data/KnowledgeBase/鎵嬪唽",
            "processed_data/KownledgeBase/鎵嬪唽",
        ),
    )
    image_root_dir = _env_or_default(
        "IMAGE_ROOT_DIR",
        _first_existing_path(
            f"{processed_data_root_dir}/插图",
            f"{data_root_dir}/插图",
            f"{processed_data_root_dir}/鎻掑浘",
            f"{data_root_dir}/鎻掑浘",
            "processed_data/KnowledgeBase/手册/插图",
            "processed_data/KownledgeBase/手册/插图",
            "data/KnowledgeBase/手册/插图",
            "data/KownledgeBase/手册/插图",
            "processed_data/KnowledgeBase/鎵嬪唽/鎻掑浘",
            "processed_data/KownledgeBase/鎵嬪唽/鎻掑浘",
            "data/KnowledgeBase/鎵嬪唽/鎻掑浘",
            "data/KownledgeBase/鎵嬪唽/鎻掑浘",
        ),
    )

    use_contextual_augmentation = os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1"
    image_description_results_json_file = (
        f"experiment/contextual_augmentation/{collection_base}.json"
        if use_contextual_augmentation
        else f"experiment/contextual_augmentation/{collection_base}_without_context.json"
    )

    return {
        "USE_QUERY_CLS": os.getenv("USE_QUERY_CLS") == "1",
        "USE_CONTEXTUAL_AUGMENTATION": use_contextual_augmentation,
        "IMAGE_DESCRIPTION_RESULTS_JSON_FILE": image_description_results_json_file,
        "DATA_ROOT_DIR": data_root_dir,
        "PROCESSED_DATA_ROOT_DIR": processed_data_root_dir,
        "IMAGE_ROOT_DIR": image_root_dir,
        "MILVUS_COLLECTION_NAME": collection_base
        if use_contextual_augmentation
        else f"{collection_base}_without_context",
    }
