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
    """Clamp score to strict open interval (0, 1) for validator compliance."""
    eps = 0.01
    return max(eps, min(1.0 - eps, raw_score))


def grade_shipping_status(state: Dict[str, Any]) -> Dict[str, Any]:
    components: Dict[str, float] = {}
    penalties: Dict[str, float] = {}

    queried = bool(state.get("flags", {}).get("queried_order_db"))
    if queried:
        components["queried_order_db"] = 0.3

    outbound_messages = state.get("outbound_messages", [])
    final_text = outbound_messages[-1]["text"] if outbound_messages else ""
    expected_date = state["task"]["expected_delivery_date"]
    if _safe_contains(final_text, expected_date):
        components["correct_delivery_date"] = 0.7

    if not outbound_messages:
        penalties["no_customer_response"] = 0.2

    raw_score = sum(components.values()) - sum(penalties.values())
    score = _strict_open_interval_score(raw_score)
    done = bool(components.get("correct_delivery_date"))
    reason = (
        "Shared the correct delivery date with customer."
        if done
        else "Need to query order and send exact delivery date."
    )
    return {
        "score": score,
        "reason": reason,
        "partial_credit": components,
        "penalties": penalties,
        "done": done,
    }


def grade_valid_refund(state: Dict[str, Any]) -> Dict[str, Any]:
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
    reason = (
        "Refund processed with proper checks."
        if done
        else "Refund flow incomplete or missing required checks."
    )
    return {
        "score": score,
        "reason": reason,
        "partial_credit": components,
        "penalties": penalties,
        "done": done,
    }


def _months_between(date_a: str, date_b: str) -> int:
    dt_a = datetime.strptime(date_a, "%Y-%m-%d")
    dt_b = datetime.strptime(date_b, "%Y-%m-%d")
    return (dt_b.year - dt_a.year) * 12 + (dt_b.month - dt_a.month)


def grade_out_of_warranty(state: Dict[str, Any]) -> Dict[str, Any]:
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
    score = _strict_open_interval_score(raw_score)
    done = bool(components.get("correct_escalation"))
    reason = (
        "Correctly escalated out-of-warranty complaint."
        if done
        else "Must escalate this complaint to human specialist."
    )
    return {
        "score": score,
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

