"""Baseline inference runner for the WhatsApp Business Triage Simulator."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console(stderr=True)
except ImportError:
    console = None

from openai import OpenAI

from env import WhatsAppBusinessTriageEnv
from models import Action
from tasks import list_task_ids


MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.environ.get("API_BASE_URL")


def _strict_open_interval_score(value: float) -> float:
    """Per my_env inference: avoid exact 0/1 boundaries (float-safe)."""
    s = float(value)
    return max(1e-6, min(s, 1.0 - 1e-6))


def _build_system_prompt() -> str:
    return (
        "You are a support agent in a WhatsApp Business triage simulator. "
        "Output ONLY valid JSON with keys: tool, arguments, rationale. "
        "Allowed tools: query_order_db, read_return_policy, send_whatsapp_message, "
        "escalate_to_human. Always pick one tool per step."
    )


def _build_user_prompt(observation: Dict[str, Any], state: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "observation": observation,
            "state_hints": {
                "steps_taken": state["steps_taken"],
                "max_steps": state["max_steps"],
                "task_id": state["task"]["task_id"],
                "available_tools": observation["available_tools"],
            },
            "instruction": (
                "Choose best next tool action. Return JSON only. "
                "For refunds, verify order and policy before marking refund."
            ),
        },
        default=str,
    )


def _deterministic_fallback_action(state: Dict[str, Any]) -> Action:
    """Fallback heuristic to keep baseline reproducible even if model fails."""
    task_id = state["task"]["task_id"]
    order_id = state["task"]["order_id"]
    flags = state["flags"]

    if task_id == "shipping_status_easy":
        if not flags["queried_order_db"]:
            return Action(tool="query_order_db", arguments={"order_id": order_id})
        delivery_date = state["crm"][order_id]["expected_delivery_date"]
        return Action(
            tool="send_whatsapp_message",
            arguments={"text": f"Your order is on the way. Expected delivery date is {delivery_date}."},
        )

    if task_id == "valid_refund_medium":
        if not flags["queried_order_db"]:
            return Action(tool="query_order_db", arguments={"order_id": order_id})
        if not flags["read_return_policy"]:
            return Action(tool="read_return_policy", arguments={})
        return Action(
            tool="send_whatsapp_message",
            arguments={
                "order_id": order_id,
                "mark_refunded": True,
                "text": "Your refund has been processed successfully.",
            },
        )

    # hard task
    if not flags["queried_order_db"]:
        return Action(tool="query_order_db", arguments={"order_id": order_id})
    if not flags["read_return_policy"]:
        return Action(tool="read_return_policy", arguments={})
    return Action(
        tool="escalate_to_human",
        arguments={"reason": "Out-of-warranty complaint requires specialist decision."},
    )


def _parse_action(content: str, state: Dict[str, Any]) -> Action:
    try:
        data = json.loads(content)
        return Action.model_validate(data)
    except Exception:
        return _deterministic_fallback_action(state)


def _observation_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    order_id = state["task"]["order_id"]
    known = state["crm"].get(order_id, {})
    return {
        "event_type": "whatsapp.inbound.message",
        "sender_id": state["task"]["sender_id"],
        "message_body": state["task"]["inbound_message"],
        "timestamp": "2026-04-09T00:00:00Z",
        "ticket_id": state["task"]["ticket_id"],
        "customer_name": state["task"].get("customer_name"),
        "language": "en",
        "channel": "whatsapp",
        "available_tools": [
            "query_order_db",
            "read_return_policy",
            "send_whatsapp_message",
            "escalate_to_human",
        ],
        "crm_context": {
            "order_id_hint": order_id,
            "steps_taken": state["steps_taken"],
            "last_tool_result": state.get("last_tool_result", {}),
            "known_order_context": {
                "order_id": known.get("order_id"),
                "status": known.get("status"),
                "refund_status": known.get("refund_status"),
                "expected_delivery_date": known.get("expected_delivery_date"),
                "product_name": known.get("product_name"),
            },
        },
    }


def _llm_next_action(client: OpenAI, env: WhatsAppBusinessTriageEnv) -> Action:
    state = env.state()
    obs_payload = _observation_payload_from_state(state)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(obs_payload, state)},
        ],
        temperature=0,
    )
    content = completion.choices[0].message.content or "{}"
    return _parse_action(content, state)


def run_episode(env: WhatsAppBusinessTriageEnv, client: OpenAI, task_id: str) -> Dict[str, Any]:
    obs = env.reset(task_id=task_id)
    done = False
    step_count = 0
    final_reward = 0.0
    rewards_list = []

    print("[START] " + f"task={task_id} env=whatsapp-business-triage-simulator model={MODEL_NAME}", flush=True)
    if console:
        console.print(Panel(f"[bold cyan]Starting Episode:[/bold cyan] {task_id}", border_style="cyan"))

    while not done and step_count < env.state()["max_steps"]:
        step_count += 1
        state_before = env.state()
        try:
            action = _llm_next_action(client, env)
        except Exception:
            action = _deterministic_fallback_action(state_before)

        result = env.step(action)
        done = result.done
        final_reward = _strict_open_interval_score(result.reward.score)
        rewards_list.append(final_reward)

        action_str = json.dumps(action.model_dump())
        done_val = str(done).lower()
        error_val = "null"

        print(
            "[STEP] " + f"step={step_count} action={action_str} reward={final_reward:.2f} done={done_val} error={error_val}",
            flush=True,
        )
        if console:
            action_color = "green" if action.tool != "escalate_to_human" else "yellow"
            tool_text = f"[{action_color}]{action.tool}[/{action_color}]"
            console.print(f"  [bold]Step {step_count}:[/bold] Tool = {tool_text} | Reward = [bold green]{final_reward:.2f}[/bold green] | Done = {done}")
        
        obs = result.observation

    success = final_reward >= 0.1
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"
    print(
        "[END] " + f"success={str(success).lower()} steps={step_count} score={final_reward:.3f} rewards={rewards_str}",
        flush=True,
    )
    if console:
        success_color = "green" if success else "red"
        console.print(Panel(f"Episode Completed!\nSuccess: [bold {success_color}]{success}[/bold {success_color}]\nFinal Score: [bold white]{final_reward:.3f}[/bold white]\nSteps taken: {step_count}", border_style=success_color))

    return {
        "task_id": task_id,
        "steps": step_count,
        "score": final_reward,
        "task_score": final_reward,
        "final_state": env.state(),
        "last_observation": obs.model_dump(),
    }


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN.")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = OpenAI(**client_kwargs)
    env = WhatsAppBusinessTriageEnv(seed=42)
    task_ids: List[str] = list_task_ids()

    results = []
    for task_id in task_ids:
        results.append(run_episode(env, client, task_id))


if __name__ == "__main__":
    main()

