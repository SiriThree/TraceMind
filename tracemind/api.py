import base64
import hmac
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference

from tracemind.pipeline import pipeline_result, pipeline_stream

app = FastAPI()
PLAYGROUND_HTML = Path(__file__).resolve().parents[1] / "assets" / "playground.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = 5 * 1024 * 1024
SUPPORTED_MIME_TYPES = [
    "image/jpg",
    "image/jpeg",
    "image/png",
    "image/webp",
]

KEFU_API_TOKEN = "kf_test"


class ClarificationPayload(BaseModel):
    is_followup: bool = Field(default=False, description="是否为澄清补充轮")
    selected_intent_id: str | None = Field(
        default=None,
        description="用户点击的候选意图 ID",
    )


class CandidateIntentModel(BaseModel):
    id: str
    label: str
    slot_key: str
    slot_value: str


class ChatRequestBody(BaseModel):
    question: str = Field(default="", description="用户问题或澄清补充")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    images: list[str] = Field(
        default=[],
        min_length=0,
        max_length=3,
        description="Base64 图片列表",
    )
    stream: bool = Field(default=False, description="是否使用流式响应")
    clarification: ClarificationPayload | None = Field(
        default=None,
        description="澄清补充信息",
    )


class ChatResponseData(BaseModel):
    answer: str = Field(description="客服回复文本")
    session_id: str = Field(description="会话 ID")
    timestamp: str = Field(description="响应时间戳")
    response_type: Literal["answer", "clarification"] = Field(
        default="answer",
        description="answer 表示直接回答，clarification 表示需要补充信息",
    )
    clarification_state: Literal["none", "pending", "resolved"] = Field(
        default="none",
        description="当前澄清状态",
    )
    candidate_intents: list[CandidateIntentModel] = Field(
        default_factory=list,
        description="候选补充方向",
    )
    missing_slots: list[str] = Field(
        default_factory=list,
        description="当前仍缺失的信息槽位",
    )
    effective_query: str = Field(
        default="",
        description="本轮最终送入主链路的查询文本",
    )
    session_debug: dict = Field(
        default_factory=dict,
        description="当前会话状态调试信息",
    )


class ChatResponse(BaseModel):
    code: int = Field(description="响应状态码", examples=[0])
    message: str = Field(description="响应消息", examples=["success"])
    data: ChatResponseData = Field(description="响应数据")


def validate_base64_image(base64_image: str) -> None:
    if base64_image.startswith("data:"):
        try:
            header, data = base64_image.split(",", 1)
            image_mime_type = header.split(";")[0].split(":")[1]
            if ";base64" not in header:
                raise ValueError("Invalid data URL format")
            if image_mime_type not in SUPPORTED_MIME_TYPES:
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


def verify_bearer_token(authorization: str) -> None:
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
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def verify_x_client_type(x_client_type: Optional[str]) -> None:
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
    description="提供客服问答服务，支持文本和图片输入。",
)
async def chat(
    body: ChatRequestBody,
    authorization: str = Header(
        alias="Authorization",
        description="Bearer token",
        examples=["Bearer kf_test"],
    ),
    x_request_id: Optional[str] = Header(
        default=None,
        alias="X-Request-Id",
        description="请求唯一标识",
    ),
    x_client_type: Optional[str] = Header(
        default=None,
        alias="X-Client-Type",
        description="调用终端类型",
        examples=["app"],
    ),
) -> ChatResponse | StreamingResponse:
    verify_bearer_token(authorization)
    if x_client_type is not None:
        verify_x_client_type(x_client_type)
    del x_request_id

    session_id = body.session_id or f"kf_{str(uuid.uuid4())}"
    question = body.question.strip()

    if not question and not (
        body.clarification and body.clarification.is_followup and body.clarification.selected_intent_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question cannot be empty unless a clarification choice is selected",
        )

    for base64_image in body.images:
        validate_base64_image(base64_image)

    clarification_payload = (
        body.clarification.model_dump() if body.clarification is not None else None
    )

    if body.stream:
        return StreamingResponse(
            pipeline_stream(
                question,
                thread_id=session_id,
                clarification=clarification_payload,
            ),
            media_type="text/event-stream",
        )

    result = await pipeline_result(
        question,
        thread_id=session_id,
        clarification=clarification_payload,
    )
    timestamp = str(int(time.time()))
    return ChatResponse(
        code=0,
        message="success",
        data=ChatResponseData(
            answer=result["answer"],
            session_id=session_id,
            timestamp=timestamp,
            response_type=result["response_type"],
            clarification_state=result["clarification_state"],
            candidate_intents=[
                CandidateIntentModel(**item) for item in result["candidate_intents"]
            ],
            missing_slots=result["missing_slots"],
            effective_query=result["effective_query"],
            session_debug=result["session_debug"],
        ),
    )


@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="TraceMind API Docs",
    )


@app.get("/playground", include_in_schema=False)
async def playground():
    return FileResponse(PLAYGROUND_HTML)


def main():
    uvicorn.run(
        "tracemind.api:app",
        host="0.0.0.0",
        port=8000,
        env_file=".env",
        reload=False,
    )


if __name__ == "__main__":
    main()
