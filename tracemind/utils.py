import base64
import mimetypes
import re
from pathlib import Path
from typing import Literal, cast

from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder

from tracemind.config import get_config


def get_image_name(image_name: str) -> str:
    """
    Return the image file name with extension if it exists under IMAGE_ROOT_DIR.
    """
    image_root_dir = Path(get_config()["IMAGE_ROOT_DIR"])
    raw_path = Path(image_name)
    if raw_path.suffix:
        direct_name = raw_path.name
        if Path(image_root_dir, direct_name).exists():
            return direct_name

    extensions = [".png", ".jpg", ".jpeg"]
    for ext in extensions:
        image_path = Path(image_root_dir, raw_path.stem + ext)
        if image_path.exists():
            return raw_path.stem + ext
    raise FileNotFoundError(f"图片不存在: {image_name}")


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    Encode an image to base64 and return its MIME type.
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"

    with open(path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return image_base64, mime_type


def language_detect(text: str) -> Literal["chinese", "english"]:
    """
    Detect whether the input text is Chinese or English.
    """
    stripped = text.strip()
    if not stripped:
        return "english"

    if re.search(r"[\u4e00-\u9fff]", stripped):
        return "chinese"

    if re.search(r"[A-Za-z]", stripped):
        return "english"

    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(stripped)

    if language is None:
        return "english"

    return cast(Literal["chinese", "english"], language.name.lower())


def sanitize_answer_text(text: str) -> str:
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    mojibake_replacements = {
        "�C": " - ",
        "��": "'",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for old, new in mojibake_replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = cleaned.replace("�", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_answer(answer: str) -> tuple[str, list[str]]:
    """
    Parse LLM output into plain content and image placeholders.
    Accepts either wrapped <answer>...</answer> or plain text.
    """
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL | re.IGNORECASE)
    content = match.group(1).strip() if match else answer.strip()

    soup = BeautifulSoup(content, "html.parser")
    pics = soup.find_all("pic")
    image_names = cast(
        list[str],
        [pic.get("image_name") for pic in pics if pic.get("image_name")],
    )

    for pic in pics:
        content = content.replace(str(pic), "<PIC>")

    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE).strip()
    content = sanitize_answer_text(content)

    pic_count = content.count("<PIC>")
    if pic_count != len(image_names):
        if pic_count > len(image_names):
            while content.count("<PIC>") > len(image_names):
                content = content.replace("<PIC>", "", 1)
        elif pic_count < len(image_names):
            image_names = image_names[:pic_count]

    return content, image_names


def convert_answer_to_ret(answer: str) -> str:
    """
    Convert the LLM answer into `answer, image_list` string format.
    """
    content, image_names = parse_answer(answer)
    ret = content
    if image_names:
        ret += "," + str(image_names)
    return ret


def convert_ret_to_answer(ret: str) -> str:
    data = ret.split(",[")
    description: str = data[0]
    if len(data) > 1:
        image_list = eval(f"[{data[1]}")
    else:
        image_list = []

    for image_name in image_list:
        llm_generate_pic = f"<pic image_name='{image_name}'></pic>"
        description = description.replace("<PIC>", llm_generate_pic, 1)

    soup = BeautifulSoup(description, "html.parser")
    pics = soup.find_all("pic")
    image_names = cast(list[str], [pic.get("image_name") for pic in pics])
    assert len(image_names) == len(image_list), (
        f"图片数量不一致: 解析为 {len(image_names)}, 输入为 {len(image_list)}"
    )
    return "<answer>\n" + description + "\n</answer>"
