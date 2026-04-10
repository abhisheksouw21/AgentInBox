"""Pre-submission validator for hackathon/OpenEnv checks.

Run:
    python prevalidate.py
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi.testclient import TestClient

from env import WhatsAppBusinessTriageEnv
from models import Action
from server.app import app
from tasks import GRADERS, list_task_ids, grade_out_of_warranty, grade_shipping_status, grade_valid_refund


ROOT = Path(__file__).resolve().parent


def _check_openenv_manifest() -> tuple[bool, str]:
    path = ROOT / "openenv.yaml"
    if not path.exists():
        return False, "openenv.yaml missing"
    data = yaml.safe_load(path.read_text())
    required = ["spec_version", "name", "runtime", "app", "port"]
    missing = [key for key in required if key not in data]
    if missing:
        return False, f"openenv.yaml missing keys: {missing}"
    if data.get("runtime") != "fastapi":
        return False, "openenv.yaml runtime must be fastapi"
    return True, "openenv.yaml looks valid"


def _check_server_endpoints() -> tuple[bool, str]:
    client = TestClient(app)
    checks: List[tuple[str, int]] = [
        ("/health", 200),
        ("/metadata", 200),
        ("/schema", 200),
        ("/state", 200),
    ]
    for route, expected in checks:
        r = client.get(route)
        if r.status_code != expected:
            return False, f"{route} expected {expected}, got {r.status_code}"

    r_reset = client.post("/reset", json={})
    if r_reset.status_code != 200:
        return False, f"/reset failed: {r_reset.status_code}"
    if "observation" not in r_reset.json():
        return False, "/reset missing observation payload"

    r_step = client.post(
        "/step",
        json={"action": {"tool": "query_order_db", "arguments": {"order_id": "ORD-1001"}}},
    )
    if r_step.status_code != 200:
        return False, f"/step failed: {r_step.status_code}"
    if "reward" not in r_step.json():
        return False, "/step missing reward payload"
    return True, "API endpoints healthy"


def _check_reward_range() -> tuple[bool, str]:
    env = WhatsAppBusinessTriageEnv(seed=42)
    for task_id in list_task_ids():
        env.reset(task_id=task_id)
        actions = [
            Action(tool="query_order_db", arguments={"order_id": env.state()["task"]["order_id"]}),
            Action(tool="read_return_policy", arguments={}),
            Action(
                tool="send_whatsapp_message",
                arguments={"text": "Acknowledged. Working on your request."},
            ),
        ]
        for action in actions:
            result = env.step(action)
            score = result.reward.score
            if score < 0.0 or score > 1.0:
                return False, f"reward out of range for {task_id}: {score}"
            if result.done:
                break
    return True, "reward range is within [0.0, 1.0]"


def _check_graders_exist() -> tuple[bool, str]:
    task_ids = list_task_ids()
    for task_id in task_ids:
        if task_id not in GRADERS:
            return False, f"missing grader for {task_id}"
    return True, "all tasks have graders"


def _strict_open_score_ok(score: float) -> bool:
    """Match platform rule: strictly between 0 and 1 (exclude 0.0 and 1.0)."""
    if not isinstance(score, (int, float)):
        return False
    s = float(score)
    return 0.0 < s < 1.0


def _check_grader_imports_and_scores() -> tuple[bool, str]:
    """Same checks an external validator often runs: import grader paths from YAML and score with {}."""
    path = ROOT / "openenv.yaml"
    data = yaml.safe_load(path.read_text())
    tasks = data.get("tasks") or []
    if len(tasks) < 3:
        return False, f"openenv.yaml must list at least 3 tasks, got {len(tasks)}"

    for t in tasks:
        tid = t.get("id")
        grader_path = t.get("grader")
        if not tid or not grader_path:
            return False, f"task entry missing id or grader: {t}"
        mod_name, _, fn_name = grader_path.rpartition(".")
        if not mod_name or not fn_name:
            return False, f"invalid grader path: {grader_path}"
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
        except Exception as exc:
            return False, f"cannot import {grader_path}: {exc}"
        try:
            out = fn({})
        except Exception as exc:
            return False, f"{grader_path}({{}}) raised: {exc}"
        if not isinstance(out, dict) or "score" not in out:
            return False, f"{grader_path} must return dict with score, got {type(out)}"
        if not _strict_open_score_ok(float(out["score"])):
            return False, f"{grader_path} score out of strict (0,1): {out['score']}"

    # Direct function smoke test (same as YAML paths)
    for fn in (grade_shipping_status, grade_valid_refund, grade_out_of_warranty):
        out = fn({})
        if not _strict_open_score_ok(float(out["score"])):
            return False, f"{fn.__name__} on empty state bad score: {out!r}"
    return True, "graders importable and return strict (0,1) scores even on empty state"


def _check_inference_logging_format() -> tuple[bool, str]:
    # Keep this check lightweight and offline by scanning source format contract.
    src = (ROOT / "inference.py").read_text()
    required_tokens = ['"[START] "', '"[STEP] "', '"[END] "']
    missing = [tok for tok in required_tokens if tok not in src]
    if missing:
        return False, f"inference.py missing log tokens: {missing}"

    # Optional runtime check if API key exists.
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN"):
        proc = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout
        if "[START]" not in output or "[END]" not in output:
            return False, "inference.py runtime output missing START/END"
    return True, "inference log format contract present"


def main() -> None:
    checks = [
        ("manifest", _check_openenv_manifest),
        ("api_endpoints", _check_server_endpoints),
        ("graders", _check_graders_exist),
        ("grader_contract", _check_grader_imports_and_scores),
        ("reward_range", _check_reward_range),
        ("inference_log_format", _check_inference_logging_format),
    ]

    report: Dict[str, Any] = {"passed": True, "checks": []}
    for name, fn in checks:
        ok, message = fn()
        report["checks"].append({"name": name, "passed": ok, "message": message})
        if not ok:
            report["passed"] = False

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

