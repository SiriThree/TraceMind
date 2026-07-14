import base64
import hmac
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference

from tracemind.pipeline import pipeline, pipeline_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
SUPPORTED_MIME_TYPES = [
    "image/jpg",
    "image/jpeg",
    "image/png",
    "image/webp",
]

KEFU_API_TOKEN = "kf_test"


class ChatRequestBody(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="用户问题,长度不能小于1个字符",
        examples=[
            "使用吹风机时，人员需要佩戴哪些防护装备？",
            "请问你们家的商品支持7天无理由退换货吗？",
        ],
    )
    session_id: Optional[str] = Field(
        default=None, description="会话 ID,首次会话不需要传"
    )
    images: List[str] = Field(
        default=[],
        min_length=0,
        max_length=3,
        description="Base64 图片列表，格式为data:image/{png/jpg/jpeg/webp};base64,{编码内容}",
    )
    stream: bool = Field(
        default=False, description="是否使用流式响应,客服场景默认为False"
    )


class ChatResponseData(BaseModel):
    answer: str = Field(description="客服回复的答案")
    session_id: str = Field(description="会话 ID, 用于多轮对话时保持上下文一致")
    timestamp: str = Field(description="响应时间戳,单位为秒")


class ChatResponse(BaseModel):
    code: int = Field(description="响应状态码", examples=[0])
    message: str = Field(description="响应消息,错误或者成功的msg", examples=["success"])
    data: ChatResponseData = Field(
        description="响应数据",
        examples=[
            ChatResponseData(
                answer="客服回复的答案", session_id="12121223", timestamp="1781449033"
            )
        ],
    )


def validate_base64_image(base64_image: str) -> None:
    """
    校验并解码 Base64 图片,格式需为data:image/{png/jpg/jpeg/webp};base64,{编码内容}
    """
    if base64_image.startswith("data:"):
        try:
            header, data = base64_image.split(",", 1)
            image_mime_type = header.split(";")[0].split(":")[1]
            if ";base64" not in header:
                raise ValueError("Invalid data URL format")

            if not any(mime == image_mime_type for mime in SUPPORTED_MIME_TYPES):
                raise ValueError("Unsupported image MIME type")
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Base64 image data URL: {e}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Base64 image data, must start with data:",
        )
    image_bytes = base64.b64decode(data)

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size exceeds 5MB",
        )


def verify_bearer_token(authorization: str):
    """
    校验 Authorization: Bearer {KEFU_API_TOKEN}
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format, expected: Bearer xxx",
        )

    token = parts[1].strip()

    if not hmac.compare_digest(token, KEFU_API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


def verify_x_client_type(x_client_type: Optional[str]):
    """
    校验 X-Client-Type: app,ios,web,wx_miniprogram
    """
    if x_client_type:
        allowed_client_types = {"app", "ios", "web", "wx_miniprogram"}

        for client_type in x_client_type.split():
            if client_type not in allowed_client_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported X-Client-Type: {client_type}",
                )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="客服问答接口",
    description="提供客服问答服务，支持文本和图片输入，返回客服回复的答案,同时支持流式返回和非流式返回",
)
async def chat(
    body: ChatRequestBody,
    authorization: str = Header(
        alias="Authorization",
        description="鉴权的参数，格式为Bearer {token},测试的时候传入Bearer kf_test",
        example="Bearer kf_test",
    ),
    x_request_id: Optional[str] = Header(
        default=None,
        alias="X-Request-Id",
        description="请求的唯一标识，用来客服问题追溯",
    ),
    x_client_type: Optional[str] = Header(
        default=None,
        alias="X-Client-Type",
        description="标识调用终端，用于客服话术适配,只能为app,ios,web,wx_miniprogram中的一个",
        example="app",
    ),
) -> ChatResponse | StreamingResponse:
    verify_bearer_token(authorization)
    if x_client_type is not None:
        verify_x_client_type(x_client_type)
    question = body.question
    session_id = body.session_id
    # 图片没用，不进行处理
    images = body.images
    is_stream = body.stream

    for base64_image in images:
        validate_base64_image(base64_image)
    if session_id is None:
        session_id = f"kf_{str(uuid.uuid4())}"
    if is_stream:
        return StreamingResponse(
            pipeline_stream(question, thread_id=session_id),
            media_type="text/event-stream",
        )
    else:
        answer = await pipeline(question, thread_id=session_id)
        timestamp = str(int(time.time()))
        return ChatResponse(
            code=0,
            message="success",
            data=ChatResponseData(
                answer=answer, session_id=session_id, timestamp=timestamp
            ),
        )


@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="用户服务 API 文档",
    )


def main():
    uvicorn.run(
        "tracemind.api:app", host="0.0.0.0", port=8000, env_file=".env", reload=False
    )


if __name__ == "__main__":
    main()
