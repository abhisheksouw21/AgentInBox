"""Task fixtures and graders for WhatsApp Business Triage Simulator."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List


TASK_SHIPPING_STATUS_EASY = "shipping_status_easy"
TASK_VALID_REFUND_MEDIUM = "valid_refund_medium"
TASK_WARRANTY_COMPLAINT_HARD = "warranty_complaint_hard"


TASK_FIXTURES: Dict[str, Dict[str, Any]] = {
    TASK_SHIPPING_STATUS_EASY: {
        "task_id": TASK_SHIPPING_STATUS_EASY,
        "difficulty": "easy",
        "max_steps": 6,
        "ticket_id": "TKT-1001",
        "sender_id": "wa:+919876500001",
        "customer_name": "Aarav Sharma",
        "inbound_message": "Hi, where is my order ORD-1001? I need the delivery date.",
        "order_id": "ORD-1001",
        "expected_delivery_date": "2026-04-14",
        "target_keywords": ["2026-04-14"],
    },
    TASK_VALID_REFUND_MEDIUM: {
        "task_id": TASK_VALID_REFUND_MEDIUM,
        "difficulty": "medium",
        "max_steps": 8,
        "ticket_id": "TKT-2001",
        "sender_id": "wa:+919876500002",
        "customer_name": "Neha Verma",
        "inbound_message": (
            "I want a refund for ORD-2001. It arrived damaged yesterday."
        ),
        "order_id": "ORD-2001",
        "target_keywords": ["refund", "processed"],
    },
    TASK_WARRANTY_COMPLAINT_HARD: {
        "task_id": TASK_WARRANTY_COMPLAINT_HARD,
        "difficulty": "hard",
        "max_steps": 8,
        "ticket_id": "TKT-3001",
        "sender_id": "wa:+919876500003",
        "customer_name": "Rohan Kulkarni",
        "inbound_message": (
            "My earbuds stopped working after 16 months, give me 70% discount now."
        ),
        "order_id": "ORD-3001",
        "target_keywords": ["escalate", "specialist", "human"],
    },
}


def list_task_ids() -> List[str]:
    return [
        TASK_SHIPPING_STATUS_EASY,
        TASK_VALID_REFUND_MEDIUM,
        TASK_WARRANTY_COMPLAINT_HARD,
    ]


def get_task_fixture(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_FIXTURES:
        raise ValueError(f"Unknown task_id: {task_id}")
    return deepcopy(TASK_FIXTURES[task_id])


def initial_crm_state() -> Dict[str, Dict[str, Any]]:
    """Returns clean in-memory CRM records used by reset()."""
    return {
        "ORD-1001": {
            "order_id": "ORD-1001",
            "status": "shipped",
            "expected_delivery_date": "2026-04-14",
            "refund_status": "not_requested",
            "purchase_date": "2026-04-01",
            "delivered_at": None,
            "warranty_months": 12,
            "product_name": "Bluetooth Speaker Mini",
        },
        "ORD-2001": {
            "order_id": "ORD-2001",
            "status": "delivered",
            "expected_delivery_date": "2026-04-07",
            "refund_status": "not_requested",
            "purchase_date": "2026-03-30",
            "delivered_at": "2026-04-07",
            "warranty_months": 12,
            "product_name": "Smart Kettle Pro",
            "damage_reported": True,
        },
        "ORD-3001": {
            "order_id": "ORD-3001",
            "status": "delivered",
            "expected_delivery_date": "2024-12-02",
            "refund_status": "not_requested",
            "purchase_date": "2024-12-01",
            "delivered_at": "2024-12-02",
            "warranty_months": 12,
            "product_name": "ANC Earbuds X",
        },
    }


RETURN_POLICY_TEXT = (
    "Refunds are eligible within 30 days of delivery for damaged or defective items. "
    "Out-of-warranty issues beyond 12 months must be escalated to specialist support. "
    "Agents must verify order details before issuing refunds."
)


def _safe_contains(text: str, needle: str) -> bool:
    return needle.lower() in (text or "").lower()


def _strict_open_interval_score(raw_score: float) -> float:
    """Strict open interval (0, 1); never exactly 0.0 or 1.0."""
    v = float(raw_score)
    v = min(max(v, 0.01), 0.99)
    return float(round(v, 6))


def _coerce_grader_state(state: Dict[str, Any] | None, task_id: str) -> Dict[str, Any]:
    """Validators may call graders with {} or partial state — fill from fixtures so we never crash."""
    s: Dict[str, Any] = dict(state) if state else {}
    if "task" not in s or not isinstance(s.get("task"), dict):
        s["task"] = get_task_fixture(task_id)
    if "crm" not in s or not isinstance(s.get("crm"), dict):
        s["crm"] = initial_crm_state()
    s.setdefault("flags", {})
    s.setdefault("outbound_messages", [])
    s.setdefault("now_date", "2026-04-09")
    return s


def grade_shipping_status(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _coerce_grader_state(state, TASK_SHIPPING_STATUS_EASY)
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}

    queried = bool(state.get("flags", {}).get("queried_order_db"))
    if queried:
        components["queried_order_db"] = 0.3

    outbound_messages = state.get("outbound_messages", [])
    final_text = outbound_messages[-1]["text"] if outbound_messages else ""
    expected_date = state["task"].get("expected_delivery_date", "2026-04-14")
    if _safe_contains(final_text, expected_date):
        components["correct_delivery_date"] = 0.7

    if not outbound_messages:
        penalties["no_customer_response"] = 0.2

    raw_score = sum(components.values()) - sum(penalties.values())
    score = _strict_open_interval_score(raw_score)
    done = bool(components.get("correct_delivery_date"))
    reasons = []
    if "queried_order_db" in components:
        reasons.append("Queried order DB (+0.3)")
    if "correct_delivery_date" in components:
        reasons.append("Sent correct delivery date (+0.7)")
    if "no_customer_response" in penalties:
        reasons.append("No customer response (-0.2)")
    if not reasons:
        reasons.append("No meaningful actions taken")
    reason = " | ".join(reasons) + (" => Task Done!" if done else " => Task Incomplete")
    return {
        "score": float(score),
        "reason": reason,
        "partial_credit": components,
        "penalties": penalties,
        "done": done,
    }


def grade_valid_refund(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _coerce_grader_state(state, TASK_VALID_REFUND_MEDIUM)
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
    flags = state.get("flags", {})

    if flags.get("queried_order_db"):
        components["queried_order_db"] = 0.3
    if flags.get("read_return_policy"):
        components["read_return_policy"] = 0.3

    order = state["crm"].get(state["task"]["order_id"], {})
    if order.get("refund_status") == "refunded":
        components["refund_applied"] = 0.4

    if order.get("refund_status") == "refunded" and not flags.get("queried_order_db"):
        penalties["refunded_without_db_check"] = 0.3
    if order.get("refund_status") == "refunded" and not flags.get("read_return_policy"):
        penalties["refunded_without_policy_check"] = 0.2

    outbound_messages = state.get("outbound_messages", [])
    if not outbound_messages:
        penalties["no_customer_response"] = 0.1

    raw_score = sum(components.values()) - sum(penalties.values())
    score = _strict_open_interval_score(raw_score)
    done = (
        bool(components.get("refund_applied"))
        and bool(components.get("queried_order_db"))
        and bool(components.get("read_return_policy"))
    )
    reasons = []
    if "queried_order_db" in components:
        reasons.append("Queried DB (+0.3)")
    if "read_return_policy" in components:
        reasons.append("Read policy (+0.3)")
    if "refund_applied" in components:
        reasons.append("Processed refund (+0.4)")
    if "refunded_without_db_check" in penalties:
        reasons.append("Refunded blindly without DB check (-0.3)")
    if "refunded_without_policy_check" in penalties:
        reasons.append("Refunded blindly without policy check (-0.2)")
    if "no_customer_response" in penalties:
        reasons.append("Did not reply to customer (-0.1)")
    if not reasons:
        reasons.append("No meaningful actions taken")
    reason = " | ".join(reasons) + (" => Task Done!" if done else " => Task Incomplete")
    return {
        "score": float(score),
        "reason": reason,
        "partial_credit": components,
        "penalties": penalties,
        "done": done,
    }


def _months_between(date_a: str, date_b: str) -> int:
    try:
        dt_a = datetime.strptime(date_a, "%Y-%m-%d")
        dt_b = datetime.strptime(date_b, "%Y-%m-%d")
    except ValueError:
        return 0
    return (dt_b.year - dt_a.year) * 12 + (dt_b.month - dt_a.month)


def grade_out_of_warranty(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _coerce_grader_state(state, TASK_WARRANTY_COMPLAINT_HARD)
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
    flags = state.get("flags", {})
    order = state["crm"].get(state["task"]["order_id"], {})
    now_date = state.get("now_date", "2026-04-09")

    if flags.get("queried_order_db"):
        components["queried_order_db"] = 0.2
    if flags.get("read_return_policy"):
        components["read_return_policy"] = 0.2

    purchase_date = order.get("purchase_date", "2024-12-01")
    months = _months_between(purchase_date, now_date)
    is_out_of_warranty = months > int(order.get("warranty_months", 12))

    if flags.get("escalated_to_human") and is_out_of_warranty:
        components["correct_escalation"] = 0.6

    outbound_messages = state.get("outbound_messages", [])
    combined_text = " ".join(msg.get("text", "") for msg in outbound_messages).lower()
    if "discount" in combined_text and not flags.get("escalated_to_human"):
        penalties["hallucinated_discount"] = 0.5
    if order.get("refund_status") == "refunded":
        penalties["invalid_refund_for_warranty_case"] = 0.4

    raw_score = sum(components.values()) - sum(penalties.values())
    done = bool(components.get("correct_escalation"))
    reasons = []

    # === God Tier: LLM-as-a-Judge API ===
    # We call OpenAI to parse the exact nuanced phrasing of the agent's escalation payload
    # If the key is missing (i.e., running in basic validation mode), we gracefully fallback.
    import os
    import json
    api_key = os.environ.get("OPENAI_API_KEY")
    if done and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            last_result = state.get("last_tool_result", {})
            agent_reason = last_result.get("reason", "")
            
            sys_prompt = "You are a strict QA Judge evaluating an AI customer service agent's escalation reason. Return JSON strictly formatted as {\"bonus\": float, \"analysis\": string} where bonus is between 0.0 and 0.2 depending on how polite, precise, and legally sound the escalation reason is."
            completion = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"The agent's escalation log was: '{agent_reason}'"}
                ],
                temperature=0.0,
                max_tokens=150
            )
            judge_data = json.loads(completion.choices[0].message.content)
            bonus = float(judge_data.get("bonus", 0.0))
            raw_score += bonus
            reasons.append(f"LLM-Judge Analysis: {judge_data.get('analysis', 'Good')} (+{bonus:.2f})")
        except Exception:
            pass # Fallback cleanly to standard trajectory grading limits if API fails

    score = _strict_open_interval_score(raw_score)

    if "queried_order_db" in components:
        reasons.append("Queried DB (+0.2)")
    if "read_return_policy" in components:
        reasons.append("Read policy (+0.2)")
    if "correct_escalation" in components:
        reasons.append("Correctly escalated (+0.6)")
    if "hallucinated_discount" in penalties:
        reasons.append("Hallucinated unauthorized discount (-0.5)")
    if "invalid_refund_for_warranty_case" in penalties:
        reasons.append("Issued invalid out-of-policy refund (-0.4)")
    if not reasons:
        reasons.append("No meaningful actions taken")
    reason = " | ".join(reasons) + (" => Task Done!" if done else " => Task Incomplete")
    return {
        "score": float(score),
        "reason": reason,
        "partial_credit": components,
        "penalties": penalties,
        "done": done,
    }


GRADERS = {
    TASK_SHIPPING_STATUS_EASY: grade_shipping_status,
    TASK_VALID_REFUND_MEDIUM: grade_valid_refund,
    TASK_WARRANTY_COMPLAINT_HARD: grade_out_of_warranty,
}

