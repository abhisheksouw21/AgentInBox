"""Core OpenEnv environment for WhatsApp Business Triage Simulator."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from models import Action, Observation, Reward, StepResult, ToolName
from tasks import GRADERS, RETURN_POLICY_TEXT, get_task_fixture, initial_crm_state, list_task_ids


class WhatsAppBusinessTriageEnv:
    """OpenEnv-compatible environment with reset(), step(), and state()."""

    def __init__(self, seed: int = 42, max_steps_default: int = 8) -> None:
        self.seed = seed
        self.max_steps_default = max_steps_default
        self._task_cycle = list_task_ids()
        self._task_cursor = 0
        self._state: Dict[str, Any] = {}
        self.reset()

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Resets the environment with a clean CRM and returns initial observation."""
        if task_id is None:
            task_id = self._task_cycle[self._task_cursor % len(self._task_cycle)]
            self._task_cursor += 1
        task = get_task_fixture(task_id)

        self._state = {
            "seed": self.seed,
            "task": task,
            "crm": initial_crm_state(),
            "policy_text": RETURN_POLICY_TEXT,
            "steps_taken": 0,
            "max_steps": task.get("max_steps", self.max_steps_default),
            "done": False,
            "flags": {
                "queried_order_db": False,
                "read_return_policy": False,
                "escalated_to_human": False,
            },
            "last_tool_result": {},
            "outbound_messages": [],
            "action_log": [],
            "reward_log": [],
            "now_date": "2026-04-09",
        }
        return self._make_observation()

    def step(self, action: Action | Dict[str, Any]) -> StepResult:
        """Applies an action and returns observation, reward, done, and info."""
        if self._state.get("done"):
            reward = Reward(
                score=0.0,
                reason="Episode already finished. Call reset() for a new episode.",
                partial_credit={},
                penalties={"action_after_done": 0.0},
            )
            return StepResult(
                observation=self._make_observation(),
                reward=reward,
                done=True,
                info={"warning": "episode_already_done"},
            )

        if not isinstance(action, Action):
            action = Action.model_validate(action)

        self._state["steps_taken"] += 1
        tool_result = self._apply_tool(action)
        self._state["last_tool_result"] = tool_result
        self._state["action_log"].append(
            {
                "step": self._state["steps_taken"],
                "tool": action.tool.value,
                "arguments": deepcopy(action.arguments),
                "result": deepcopy(tool_result),
            }
        )

        grader = GRADERS[self._state["task"]["task_id"]]
        graded = grader(self._state)
        done = bool(graded["done"]) or self._state["steps_taken"] >= self._state["max_steps"]
        self._state["done"] = done

        reward = Reward(
            score=float(graded["score"]),
            reason=graded["reason"],
            partial_credit=graded["partial_credit"],
            penalties=graded["penalties"],
        )
        self._state["reward_log"].append(reward.model_dump())

        info = {
            "task_id": self._state["task"]["task_id"],
            "steps_taken": self._state["steps_taken"],
            "tool_result": tool_result,
        }
        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Returns the full current environment state."""
        return deepcopy(self._state)

    def _make_observation(self) -> Observation:
        task = self._state["task"]
        crm_slice = self._build_crm_context(task["order_id"])
        return Observation(
            sender_id=task["sender_id"],
            message_body=task["inbound_message"],
            timestamp=datetime.now(timezone.utc),
            ticket_id=task["ticket_id"],
            customer_name=task.get("customer_name"),
            language="en",
            crm_context={
                "order_id_hint": task["order_id"],
                "steps_taken": self._state["steps_taken"],
                "last_tool_result": deepcopy(self._state.get("last_tool_result", {})),
                "known_order_context": crm_slice,
            },
        )

    def _build_crm_context(self, order_id: str) -> Dict[str, Any]:
        order = self._state["crm"].get(order_id, {})
        public_fields = {
            "order_id": order.get("order_id"),
            "status": order.get("status"),
            "refund_status": order.get("refund_status"),
            "expected_delivery_date": order.get("expected_delivery_date"),
            "product_name": order.get("product_name"),
        }
        return public_fields

    def _apply_tool(self, action: Action) -> Dict[str, Any]:
        if action.tool == ToolName.QUERY_ORDER_DB:
            return self._tool_query_order_db(action.arguments)
        if action.tool == ToolName.READ_RETURN_POLICY:
            return self._tool_read_return_policy()
        if action.tool == ToolName.SEND_WHATSAPP_MESSAGE:
            return self._tool_send_whatsapp_message(action.arguments)
        if action.tool == ToolName.ESCALATE_TO_HUMAN:
            return self._tool_escalate_to_human(action.arguments)
        return {"ok": False, "error": f"Unsupported tool: {action.tool.value}"}

    def _tool_query_order_db(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        order_id = str(arguments.get("order_id", "")).strip()
        if not order_id:
            return {"ok": False, "error": "order_id is required"}
        order = self._state["crm"].get(order_id)
        if not order:
            return {"ok": False, "error": f"order {order_id} not found"}

        self._state["flags"]["queried_order_db"] = True
        return {
            "ok": True,
            "order": deepcopy(order),
        }

    def _tool_read_return_policy(self) -> Dict[str, Any]:
        self._state["flags"]["read_return_policy"] = True
        return {
            "ok": True,
            "policy_text": self._state["policy_text"],
        }

    def _tool_send_whatsapp_message(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "text is required"}

        record = {
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._state["outbound_messages"].append(record)

        # Optional state update tool behavior for refund task.
        order_id = str(arguments.get("order_id", self._state["task"].get("order_id", ""))).strip()
        mark_refunded = bool(arguments.get("mark_refunded", False))
        if mark_refunded and order_id in self._state["crm"]:
            self._state["crm"][order_id]["refund_status"] = "refunded"

        return {"ok": True, "sent": record}

    def _tool_escalate_to_human(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(arguments.get("reason", "Complex policy case")).strip()
        self._state["flags"]["escalated_to_human"] = True
        return {
            "ok": True,
            "escalated": True,
            "reason": reason,
            "queue": "warranty-specialist",
        }

