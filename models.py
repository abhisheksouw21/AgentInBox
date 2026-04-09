"""Typed models for the WhatsApp Business Triage Simulator OpenEnv.

These models are intentionally strict because the hackathon validator expects
typed Observation, Action, and Reward structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ToolName(str, Enum):
    """Supported action tools exposed by the environment."""

    QUERY_ORDER_DB = "query_order_db"
    READ_RETURN_POLICY = "read_return_policy"
    SEND_WHATSAPP_MESSAGE = "send_whatsapp_message"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class Observation(BaseModel):
    """Webhook-like observation delivered to the agent each step."""

    event_type: Literal["whatsapp.inbound.message"] = "whatsapp.inbound.message"
    sender_id: str = Field(..., description="Customer WhatsApp sender id.")
    message_body: str = Field(..., min_length=1, description="Inbound customer text.")
    timestamp: datetime = Field(..., description="ISO8601 timestamp of inbound message.")
    ticket_id: str = Field(..., description="Unique support ticket id.")
    customer_name: Optional[str] = Field(
        default=None,
        description="Customer display name if available.",
    )
    language: str = Field(default="en", description="Detected message language.")
    channel: Literal["whatsapp"] = "whatsapp"
    crm_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context returned from internal systems.",
    )
    available_tools: list[ToolName] = Field(
        default_factory=lambda: [
            ToolName.QUERY_ORDER_DB,
            ToolName.READ_RETURN_POLICY,
            ToolName.SEND_WHATSAPP_MESSAGE,
            ToolName.ESCALATE_TO_HUMAN,
        ]
    )


class Action(BaseModel):
    """Agent-selected tool invocation."""

    tool: ToolName = Field(..., description="Tool name chosen by the agent.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable arguments for the selected tool.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Optional reason for choosing this action.",
    )

    @field_validator("arguments")
    @classmethod
    def validate_args_shape(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("Action.arguments must be a dictionary.")
        return value


class Reward(BaseModel):
    """Reward model constrained to OpenEnv-compatible range [0.0, 1.0]."""

    score: float = Field(..., ge=0.0, le=1.0, description="Primary reward signal.")
    reason: str = Field(..., min_length=1, description="Human-readable reward reason.")
    partial_credit: Dict[str, float] = Field(
        default_factory=dict,
        description="Component-level reward attribution (each value in [0.0, 1.0]).",
    )
    penalties: Dict[str, float] = Field(
        default_factory=dict,
        description="Applied penalties (negative magnitudes are not allowed).",
    )

    @field_validator("partial_credit", mode="after")
    @classmethod
    def validate_partial_credit(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, component in value.items():
            if component < 0.0 or component > 1.0:
                raise ValueError(
                    f"partial_credit['{key}'] must be in [0.0, 1.0], got {component}."
                )
        return value

    @field_validator("penalties", mode="after")
    @classmethod
    def validate_penalties(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, component in value.items():
            if component < 0.0:
                raise ValueError(
                    f"penalties['{key}'] must be >= 0.0, got {component}."
                )
        return value


class StepResult(BaseModel):
    """Convenience response container for step(action)."""

    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

