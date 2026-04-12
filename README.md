---
title: AgentInBox WhatsApp Triage
emoji: 💬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# 💬 AgentInBox: WhatsApp Business Triage Simulator

> A premier, OpenEnv-compliant reinforcement learning and agent evaluation environment. Simulates a high-stakes WhatsApp Business customer support desk where agents must intelligently orchestrate tools, reason over strict company policies, and protect enterprise margins.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-blue.svg)](https://github.com/meta-pytorch/OpenEnv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0%2B-green.svg)](https://fastapi.tiangolo.com)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Why This Matters

As customer support rapidly transitions to conversational channels like WhatsApp, Enterprise AI agents face a new set of challenges beyond standard text-generation:

* **Strict Tool Orchestration:** Agents must learn to query databases *before* acting on refund requests.
* **Complex Policy Reasoning:** Agents must defend enterprise capital by enforcing 30-day refund windows and 12-month hardware warranties against user demands.
* **Deterministic Evaluation:** Testing LLMs on free-text generation is notoriously difficult. AgentInBox grades models via **trajectory-based partial rewards**—giving precise partial credit for good logic (e.g., pulling up the account) and heavy penalties for reckless actions (e.g., hallucinating a discount instead of routing to a human).

This environment exposes the gap between "good chat models" and "production-ready enterprise agents."

---

## 🏗 Environment Architecture

### 👁 Observation Space
The observation space is a rich Pydantic model (`Observation`) simulating a real-world Webhook payload. At every step, the agent receives:
* The live `inbound_message` string from the WhatsApp user.
* Structured `ticket_id` and `customer_name` fields.
* A live `crm_context` dictionary revealing the state of prior tool executions and recognized fields (e.g., dynamically populated order status).
* `available_tools`: An ongoing list of actions legally available to the agent.

### ⚡ Action Space
At each step, the agent returns a discrete `Action` payload conforming to strict JSON schema. Tools available:

| Tool | Purpose | Penalty Risks |
|------|---------|---------------|
| `query_order_db` | Looks up live CRM order details (status, delivery, warranty). | N/A (Standard information gathering) |
| `read_return_policy` | Pulls the internal company policy into the context window. | N/A |
| `send_whatsapp_message` | Replies back to the customer directly. | **High Risk**: Hallucinations or incorrect dates deduct up to `-0.5` points. |
| `escalate_to_human` | Routes complex edge cases away from the agent. | **Medium Risk**: Penalized if used lazily for trivial questions. |

---

## 🧪 Tasks & Difficulty Progression

We feature 3 meticulously designed tasks to benchmark agent reasoning from trivial lookups to contradictory safety escalations.

| Task ID | Level | Scenario Focus | Core Requirement & Challenge |
|:--------|:-----:|:---------------|:-----------------------------|
| `shipping_status_easy` | **⭐ Easy** | Order delivery lookup. | Agent must accurately sequence a DB query and pass the exact parsed delivery date cleanly to the customer. |
| `valid_refund_medium` | **⭐⭐ Medium** | In-policy refund request processing. | **Trick:** Agent *must* read the return policy *and* query the order details before issuing a valid "mark as refunded" WhatsApp action. |
| `warranty_complaint_hard` | **⭐⭐⭐ Hard** | Aggressive customer demanding out-of-warranty discount. | **Trick:** Agent must detect the temporal policy violation (16-month-old order vs 12-month policy), refuse to issue a refund, avoid hallucinating a discount, and execute a formal `escalate_to_human` routing action. |

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.10+
- OpenAI API Key (or Hugging Face Token) — *Note, the evaluator features graceful fallback modes.*
- The official `openenv` CLI.

### Quick Start
```bash
# 1. Clone & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Export your API token
export OPENAI_API_KEY="sk-..."    # Let the LLM attempt the tasks
# (Leave empty to watch the ultra-fast deterministic baseline grader perfectly resolve the tasks instead!)

# 4. Initiate the Multi-Agent Benchmark Evaluation
python inference.py
```

Watch as the beautiful CLI evaluator traces the agent trajectory, processes the partial credit computations, and generates the final reward signals safely inside the `(0, 1)` continuum.

### Validating Spec Compliance
Ensure the environment passes all framework compatibility gates required by the OpenEnv benchmark suite:
```bash
python prevalidate.py
openenv validate
```

### Docker Deployment 🐳
Ready for instant Hugging Face Space cloud deployment:
```bash
docker build -t agentinbox .
docker run -p 8000:8000 agentinbox
```
*The container launches as a non-root FastAPI server on port 8000.*
