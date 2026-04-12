"""FastAPI server exposing the WhatsApp Business Triage environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from env import WhatsAppBusinessTriageEnv
from models import Action, Observation, Reward


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None)
    episode_id: Optional[str] = Field(default=None)
    task_id: Optional[str] = Field(default=None)
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {},
                {"task_id": "shipping_status_easy"},
                {"seed": 42, "task_id": "valid_refund_medium"},
            ]
        }
    )


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = Field(default=None)
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action": {
                        "tool": "query_order_db",
                        "arguments": {"order_id": "ORD-1001"},
                    }
                },
                {
                    "action": {
                        "tool": "send_whatsapp_message",
                        "arguments": {"text": "Expected delivery is 2026-04-14"},
                    }
                },
            ]
        }
    )


class EnvStepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool


app = FastAPI(
    title="WhatsApp Business Triage Simulator",
    version="1.0.0",
    description=(
        "OpenEnv-compatible API for WhatsApp customer support triage. "
        "Use /reset, /step, and /state to run episodes."
    ),
    contact={"name": "Meta x Scaler Hackathon Submission"},
    openapi_tags=[
        {"name": "system", "description": "Service health and metadata endpoints."},
        {"name": "interaction", "description": "Episode interaction endpoints."},
    ],
)

_ENV = WhatsAppBusinessTriageEnv(seed=42)


@app.get("/", tags=["system"], summary="API root", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", tags=["system"], include_in_schema=False)
def favicon() -> Dict[str, str]:
    # Return a lightweight no-op response to avoid browser 404 noise.
    return {"status": "no_favicon"}


@app.get("/health", tags=["system"], summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata", tags=["system"], summary="Environment metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "WhatsAppBusinessTriageEnv",
        "description": "WhatsApp customer support triage simulation environment.",
        "version": "1.0.0",
        "author": "Meta x Scaler Hackathon Submission",
    }


@app.get("/schema", tags=["system"], summary="Action/observation/state schemas")
def schema() -> Dict[str, Any]:
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "reward": Reward.model_json_schema(),
        "state": {"type": "object"},
    }


@app.post(
    "/reset",
    tags=["interaction"],
    summary="Reset environment",
    response_model=EnvStepResponse,
)
def reset(payload: Optional[ResetRequest] = None) -> EnvStepResponse:
    req = payload or ResetRequest()
    if req.seed is not None:
        global _ENV
        _ENV = WhatsAppBusinessTriageEnv(seed=req.seed)
    obs = _ENV.reset(task_id=req.task_id)
    return EnvStepResponse(
        observation=obs.model_dump(mode="json"),
        reward=None,
        done=False,
    )


@app.post(
    "/step",
    tags=["interaction"],
    summary="Apply one tool action",
    response_model=EnvStepResponse,
)
def step(payload: Dict[str, Any]) -> EnvStepResponse:
    # Accept both action body styles (OpenEnv clients vary):
    # 1) {"tool": "...", "arguments": {...}}
    # 2) {"action": {"tool": "...", "arguments": {...}}, "timeout_s": ...}
    action_payload = payload.get("action", payload)
    result = _ENV.step(action_payload)
    # Scalar reward matches openenv.core StepResult / reference my_env HTTP shape.
    return EnvStepResponse(
        observation=result.observation.model_dump(mode="json"),
        reward=result.reward.score,
        done=result.done,
    )


@app.get("/state", tags=["interaction"], summary="Get full internal state")
def state() -> Dict[str, Any]:
    return _ENV.state()


def main() -> FastAPI:
    """Return the ASGI application for OpenEnv multi-mode validators."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

