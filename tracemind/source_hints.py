from typing import Literal


CHINESE_PRODUCT_TO_SOURCE = {
    "吹风机": "吹风机手册.txt",
    "空调": "空调手册.txt",
    "洗碗机": "洗碗机手册.txt",
    "蒸汽清洁机": "蒸汽清洁机手册.txt",
    "空气净化器": "空气净化器手册.txt",
    "健身追踪器": "健身追踪器手册.txt",
    "健身单车": "健身单车手册.txt",
    "电钻": "电钻手册.txt",
    "烤箱": "烤箱手册.txt",
    "冰箱": "冰箱手册.txt",
    "发电机": "发电机手册.txt",
    "摩托艇": "摩托艇手册.txt",
    "水泵": "水泵手册.txt",
    "蓝牙激光鼠标": "蓝牙激光鼠标手册.txt",
    "VR头显": "VR头显手册.txt",
    "功能键盘": "功能键盘手册.txt",
    "相机": "相机手册.txt",
}


ENGLISH_PRODUCT_TO_SOURCE = {
    "Digital SLR Camera": "Digital SLR Camera User Manual.txt",
    "Air Fryer": "Air Fryer User Manual.txt",
    "Robot Vacuum": "Robot Vacuum User Manual.txt",
    "Washing Machine": "Washing Machine User Manual.txt",
    "Over-the-Range Microwave": "Over-the-Range Microwave User Manual.txt",
    "Riding Lawn Mower": "Riding Lawn Mower User Manual.txt",
}


def resolve_source_hint(
    product_name: str | None,
    language: Literal["chinese", "english"],
) -> str | None:
    if not product_name:
        return None
    if language == "chinese":
        return CHINESE_PRODUCT_TO_SOURCE.get(product_name)
    return ENGLISH_PRODUCT_TO_SOURCE.get(product_name)
