import argparse
import ast
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter

from tracemind.config import get_config
from tracemind.model_factory import create_embedding_model
from tracemind.utils import get_image_name

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

embedding_model = create_embedding_model()

DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "db_name": os.getenv("MILVUS_DB_NAME", "default"),
}

multi_analyzer_params = {
    "analyzers": {
        "english": {"type": "english"},
        "chinese": {"type": "chinese"},
        "default": {"tokenizer": "icu"},
    },
    "by_field": "language",
}


@dataclass(frozen=True)
class ManualItem:
    file_path: Path
    language: Literal["chinese", "english"]
    source: str


@dataclass
class BuildStats:
    planned_manuals: int = 0
    skipped_existing_manuals: int = 0
    ingested_manuals: int = 0
    ingested_chunks: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the TraceMind handbook knowledge base into Milvus."
    )
    parser.add_argument(
        "--manual-filter",
        default=os.getenv("MANUAL_FILTER", "").strip(),
        help="Only build manuals whose source or filename contains this keyword.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Re-ingest manuals even if the same source already exists in Milvus.",
    )
    return parser.parse_args()


def create_vector_store() -> Milvus:
    return Milvus(
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
                "params": {"M": 16, "efConstruction": 64},
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


def load_english_handbook_names() -> list[str]:
    file_path = os.getenv(
        "ENGLISH_HANDBOOK_NAME_FILE",
        "catalog/handbook_name_gemini.json",
    )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def replace_pics_with_placeholders(
    description: str,
    image_list: list[str],
    source: str,
) -> str:
    replaced = description
    for raw_image_name in image_list:
        image_name = get_image_name(raw_image_name)
        placeholder = (
            f"<picture-description image_name='{image_name}'>"
            f"这是与《{source}》当前相邻段落相关的产品说明插图，请结合前后文理解。"
            f"</picture-description>"
        )
        replaced = replaced.replace("<PIC>", placeholder, 1)
    return replaced


def split_manual(
    file_path: Path,
    language: Literal["chinese", "english"],
    source: str,
) -> list[Document]:
    content = file_path.read_text(encoding="utf-8")
    description, image_list = ast.literal_eval(content)
    pic_count = description.count("<PIC>")
    if pic_count != len(image_list):
        raise ValueError(
            f"{file_path.name} PIC count mismatch: {pic_count} != {len(image_list)}"
        )

    normalized = replace_pics_with_placeholders(description, image_list, source)
    splitter = MarkdownHeaderTextSplitter(
        [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")],
        strip_headers=False,
        return_each_line=False,
    )
    docs = splitter.split_text(normalized)
    final_docs: list[Document] = []
    for index, doc in enumerate(docs):
        lines = doc.page_content.split("\n")
        if lines and lines[0].startswith("#"):
            doc.page_content = "\n".join(lines[1:])

        for level in reversed(range(1, 4)):
            header_name = f"Header_{level}"
            if header_name in doc.metadata:
                doc.page_content = (
                    f"{'#' * level} {doc.metadata[header_name]}\n{doc.page_content}"
                )
                doc.metadata.pop(header_name)

        doc.metadata["source"] = source
        doc.metadata["index"] = index
        doc.metadata["language"] = language
        final_docs.append(doc)

    return final_docs


def iter_manuals(manual_filter: str = "") -> list[ManualItem]:
    processed_dir = Path(get_config()["PROCESSED_DATA_ROOT_DIR"])
    english_names = load_english_handbook_names()
    items: list[ManualItem] = []

    for file_path in sorted(processed_dir.glob("*.txt")):
        stem = file_path.stem
        if not (
            stem.endswith("_formatted")
            or "_formatted_" in stem
        ):
            continue

        if "英文手册" in stem:
            index = int(stem.split("_")[2])
            source = f"{english_names[index - 1]}.txt"
            language: Literal["chinese", "english"] = "english"
        else:
            source = f"{stem.replace('_formatted', '')}.txt"
            language = "chinese"

        if manual_filter and manual_filter not in source and manual_filter not in file_path.name:
            continue
        items.append(ManualItem(file_path=file_path, language=language, source=source))
    return items


def collection_exists(vector_store: Milvus, collection_name: str) -> bool:
    return vector_store.client.has_collection(collection_name)


def manual_exists(vector_store: Milvus, collection_name: str, source: str) -> bool:
    if not collection_exists(vector_store, collection_name):
        return False
    results = vector_store.client.query(
        collection_name=collection_name,
        filter=f"source == '{source}'",
        output_fields=["source"],
        limit=1,
    )
    return len(results) > 0


def fetch_collection_rows(vector_store: Milvus, collection_name: str) -> list[dict]:
    if not collection_exists(vector_store, collection_name):
        return []

    total_rows = vector_store.client.get_collection_stats(collection_name)["row_count"]
    if not total_rows:
        return []

    return vector_store.client.query(
        collection_name=collection_name,
        filter='source != ""',
        output_fields=["source", "language", "index"],
        limit=int(total_rows),
    )


def summarize_collection(
    vector_store: Milvus,
    collection_name: str,
    expected_sources: list[str],
) -> dict:
    rows = fetch_collection_rows(vector_store, collection_name)
    counter = Counter(row["source"] for row in rows if row.get("source"))
    actual_sources = sorted(counter)
    expected_set = set(expected_sources)
    actual_set = set(actual_sources)
    in_scope_sources = sorted(actual_set & expected_set)
    outside_scope_sources = sorted(actual_set - expected_set)
    per_source = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    return {
        "row_count": len(rows),
        "source_count": len(actual_sources),
        "in_scope_source_count": len(in_scope_sources),
        "missing_sources": sorted(expected_set - actual_set),
        "outside_scope_source_count": len(outside_scope_sources),
        "outside_scope_sources_preview": outside_scope_sources[:10],
        "min_chunks_per_source": min(counter.values()) if counter else 0,
        "max_chunks_per_source": max(counter.values()) if counter else 0,
        "top_chunk_sources": per_source[:5],
        "bottom_chunk_sources": sorted(counter.items(), key=lambda item: (item[1], item[0]))[:5],
    }


def log_final_summary(
    *,
    collection_name: str,
    manual_filter: str,
    stats: BuildStats,
    collection_summary: dict,
) -> None:
    logger.info("=" * 72)
    logger.info("TraceMind knowledge base build summary")
    logger.info("collection_name=%s", collection_name)
    logger.info("manual_filter=%s", manual_filter or "<ALL>")
    logger.info("planned_manuals=%s", stats.planned_manuals)
    logger.info("ingested_manuals=%s", stats.ingested_manuals)
    logger.info("skipped_existing_manuals=%s", stats.skipped_existing_manuals)
    logger.info("ingested_chunks=%s", stats.ingested_chunks)
    logger.info("collection_source_count=%s", collection_summary["source_count"])
    logger.info("collection_row_count=%s", collection_summary["row_count"])
    logger.info("in_scope_source_count=%s", collection_summary["in_scope_source_count"])
    logger.info("missing_sources=%s", collection_summary["missing_sources"])
    logger.info(
        "outside_scope_source_count=%s",
        collection_summary["outside_scope_source_count"],
    )
    logger.info(
        "outside_scope_sources_preview=%s",
        collection_summary["outside_scope_sources_preview"],
    )
    logger.info(
        "chunks_per_source_range=%s..%s",
        collection_summary["min_chunks_per_source"],
        collection_summary["max_chunks_per_source"],
    )
    logger.info("top_chunk_sources=%s", collection_summary["top_chunk_sources"])
    logger.info("bottom_chunk_sources=%s", collection_summary["bottom_chunk_sources"])
    logger.info("=" * 72)


def main() -> None:
    args = parse_args()
    config = get_config()
    collection_name = config["MILVUS_COLLECTION_NAME"]
    vector_store = create_vector_store()
    manuals = iter_manuals(args.manual_filter)

    if not manuals:
        logger.warning("No manuals matched the current filter: %r", args.manual_filter)
        return

    stats = BuildStats(planned_manuals=len(manuals))
    logger.info(
        "Starting knowledge base build: collection=%s planned_manuals=%s rebuild_existing=%s",
        collection_name,
        stats.planned_manuals,
        args.rebuild_existing,
    )

    for item in manuals:
        if not args.rebuild_existing and manual_exists(
            vector_store, collection_name, item.source
        ):
            stats.skipped_existing_manuals += 1
            logger.info("Skip existing source=%s", item.source)
            continue

        docs = split_manual(item.file_path, item.language, item.source)
        vector_store.add_documents(docs)
        stats.ingested_manuals += 1
        stats.ingested_chunks += len(docs)
        logger.info(
            "Ingested source=%s language=%s chunks=%s file=%s",
            item.source,
            item.language,
            len(docs),
            item.file_path.name,
        )

    expected_sources = [item.source for item in manuals]
    collection_summary = summarize_collection(
        vector_store=vector_store,
        collection_name=collection_name,
        expected_sources=expected_sources,
    )
    log_final_summary(
        collection_name=collection_name,
        manual_filter=args.manual_filter,
        stats=stats,
        collection_summary=collection_summary,
    )


if __name__ == "__main__":
    main()
