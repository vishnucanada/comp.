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
    description: str = Field("", max_length=4096)
    # `structured` is the policy DSL itself. If absent we fall back to using
    # `description` as the DSL — supports older clients that put the rules in
    # the description field.
    structured: str | None = Field(None, max_length=16384)
    test_cases: list[TestCase] = []


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., max_length=4096)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4096)
    policy_name: str | None = Field(None, max_length=64)
    history: list[ChatMessage] = []
    user_role: str | None = Field(None, max_length=64)


class StreamChatRequest(BaseModel):
    message: str = Field(..., max_length=4096)
    policy_name: str | None = Field(None, max_length=64)
    history: list[ChatMessage] = []
    user_role: str | None = Field(None, max_length=64)
