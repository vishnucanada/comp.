"""Pydantic request/response models for the dashboard API."""

from pydantic import BaseModel, Field


class PolicySaveRequest(BaseModel):
    name: str = Field(..., max_length=64)
    description: str = Field(..., max_length=4096)
    structured: str | None = Field(None, max_length=16384)


class TestCase(BaseModel):
    prompt: str = Field(..., max_length=2048)
    expected: str


class EnactRequest(BaseModel):
    policy_name: str = Field(..., max_length=64)
    description: str = Field(..., max_length=4096)
    test_cases: list[TestCase]
    low_privilege: int = 1
    rmax: int = 100


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., max_length=4096)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4096)
    policy_name: str | None = Field(None, max_length=64)
    history: list[ChatMessage] = []


class StreamChatRequest(BaseModel):
    message: str = Field(..., max_length=4096)
    policy_name: str | None = Field(None, max_length=64)
    history: list[ChatMessage] = []


class TrainRequest(BaseModel):
    model_id: str = "Qwen/Qwen2.5-0.5B"
    epochs: int = Field(3, ge=1, le=20)
    lr: float = Field(1e-4, gt=0)
    orth_reg: float = Field(0.0, ge=0)
