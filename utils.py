# 一些工具函数
import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Literal, cast

from bs4 import BeautifulSoup
from lingua import Language, LanguageDetectorBuilder


def get_image_name(image_name: str) -> str:
    """
    根据图片路径的stem，或者图片名称，原本的数据集中的图片是没有扩展名的，该函数返回带有扩展名的图片名称
    """
    image_root_dir = Path(os.getenv("IMAGE_ROOT_DIR", "data/KownledgeBase/手册/插图"))
    extensions = [".png", ".jpg", ".jpeg"]
    for ext in extensions:
        image_path = Path(image_root_dir, image_name + ext)
        if image_path.exists():
            return image_name + ext
    raise FileNotFoundError(f"该图片不存在,图片名为{image_name}")


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    将图片编码为 base64 字符串，同时返回图片的 MIME 类型,用来生成图片描述
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
    对文本进行语言检测，返回检测到的语言（"chinese" 或 "english"）
    """
    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(text)

    if language is None:
        raise Exception("语言检测失败")

    return language.name.lower()


def parse_answer(answer: str) -> tuple[str, list[str]]:
    """
    解析LLM生成的内容,提取图片名称和回答内容
    """
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        content = match.group(1).strip()
        soup = BeautifulSoup(content, "html.parser")
        pics = soup.find_all("pic")
        image_names = cast(list[str], [pic.get("image_name") for pic in pics])

        for pic in pics:
            content = content.replace(str(pic), "<PIC>")

        pic_count = content.count("<PIC>")
        if pic_count != len(image_names):
            raise Exception("替换之后的PIC的数量和解析出来的图片数量不一致")
        return content, image_names
    else:
        raise Exception("LLM生成的格式有问题")


def convert_answer_to_ret(answer: str) -> str:
    """
    将LLM生成的内容转换为格式为answer, image_list的字符串
    """
    content, image_names = parse_answer(answer)
    ret = content
    if image_names and len(image_names) > 0:
        ret += "," + str(image_names)
    return ret


def convert_ret_to_answer(ret: str) -> str:
    data = ret.split(",[")
    description: str = data[0]
    if len(data) > 1:
        image_list = eval(f"[{data[1]}")
    else:
        image_list = []
    pass
    for image_name in image_list:
        llm_generate_pic = f"<pic image_name='{image_name}'></pic>"
        description = description.replace("<PIC>", llm_generate_pic, 1)

    soup = BeautifulSoup(description, "html.parser")
    pics = soup.find_all("pic")
    image_names = cast(list[str], [pic.get("image_name") for pic in pics])
    assert len(image_names) == len(image_list), (
        f"图片数量不一致,解析出来的图片数量为{len(image_names)},输入的图片数量为{len(image_list)}"
    )
    return "<answer>\n" + description + "\n</answer>"


if __name__ == "__main__":
    # ret = "使用吹风机的人员需佩戴以下防护装备：  \n1. 合格的听力防护装备  \n2. 合格的眼部防护装备  \n3. 在多尘环境中操作时佩戴面罩  \n4. 防滑鞋底的工作靴或工作鞋  \n5. 急救箱  \n\n<PIC>,['Manual04_3']"
    ret = """这款蒸汽清洁机的实用功能和快速上手方法如下，您可以按图一步步操作：

<PIC>

一、实用产品功能

1. 二合一设计
既可以作为立式蒸汽拖把清洁地面，也可以拆下手持蒸汽器，配合不同刷头和喷嘴清洁局部区域，一机多用。

2. 水箱与过滤设计
水箱位于手持蒸汽器顶部。可使用自来水，也可优先使用蒸馏水或去离子水。机器带有水过滤器，可帮助过滤矿物质和杂质，减少水垢对机器的影响。

3. 就绪指示灯
当水已加热并可正常出蒸汽时，就绪指示灯会亮起，方便判断是否可以开始使用。

4. 蒸汽开关
立式拖地时可使用手柄上的蒸汽开关；手持模式下可使用手持蒸汽器上的开关，按下即可出蒸汽。

5. 旋转拖头
拖头可灵活转向，便于清洁家具下方、墙边和较窄区域。

6. 可重复使用拖布
吸水纤维拖布可吸附被蒸汽软化的污垢，支持重复使用，也可机洗。

<PIC>

7. 多场景配件清洁
配件较丰富，可覆盖多种家庭清洁场景：
- 延长管：适合较高、较远位置
- 喷射喷嘴：适合边角、缝隙、顽固污垢集中喷射
- 角形喷嘴：适合窗台、马桶内侧等难接触位置
- 弧形喷嘴：适合硬质表面
- 圆刷：适合瓷砖缝、灶台、台面等
- 玻璃清洁组件/布艺清洁头：适合玻璃、窗帘、衣物、家具表面等

<PIC>

<PIC>

<PIC>

<PIC>

二、快速上手使用

1. 组装主机
先将主机身底部滑入旋转拖头颈部，对齐背部孔位后，将固定扣插回锁紧；再把手柄杆插入主机身顶部，直到听到“咔哒”声，表示安装到位。

<PIC>

<PIC>

<PIC>

2. 安装拖布
按下拖布释放扣，拉开拖头挡板后装入拖布，再扣合固定。

<PIC>

3. 加水
可用配套量杯直接向水箱加水；也可以按下水箱解锁扣，将水箱取下后在水槽处加水。

<PIC>

注意：
- 水箱无水时不要继续使用
- 不要加入除垢剂、香精、酒精、清洁剂等液体
- 为减少水垢，建议优先使用蒸馏水

4. 通电预热
将电源线完全展开后，插入 120V 接地插座。等待机器加热，指示灯亮起后即可开始使用。

5. 开始清洁
建议先将地面扫净或吸尘，再缓慢推动机器清洁，同时按下蒸汽开关出蒸汽。首次使用时，可能需要等待数秒蒸汽才会出来，属于正常现象。

6. 切换手持模式并安装配件
向上拉起透明锁扣，再打开手持蒸汽器解锁扣，将手持蒸汽器向上提起取出；之后根据清洁需求安装对应配件使用。

<PIC>

<PIC>

三、使用提醒
- 适用于瓷砖、Vinyl 地板、复合地板、大理石、石材及封边木地板
- 不建议用于未封边木地板、易受热受潮损坏的材质、皮革、丝绒、打蜡家具等
- 玻璃表面使用时请谨慎，不可用于结冰玻璃
- 安装或拆卸配件前，请务必先拔掉电源，并等待机器冷却后再操作

如果您是第一次使用，按“组装主机 → 装拖布 → 加水 → 通电预热 → 开始清洁”的顺序操作即可，很快就能上手。,['Manual05_1', 'Manual05_2', 'Manual05_12', 'Manual05_13', 'Manual05_14', 'Manual05_15', 'Manual05_3', 'Manual05_4', 'Manual05_5', 'Manual05_6', 'Manual05_7', 'Manual05_9', 'Manual05_8']
"""
    answer = convert_ret_to_answer(ret)
    print(answer)
