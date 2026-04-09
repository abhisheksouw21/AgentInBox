---
title: AgentInBox
emoji: 🧰
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# WhatsApp Business Triage Simulator (OpenEnv)

A real-world OpenEnv environment where an AI agent acts as a WhatsApp Business customer support rep handling triage tickets with tool use and policy constraints.

## Implemented Files

- `openenv.yaml`: environment metadata and task config
- `models.py`: typed `Observation`, `Action`, `Reward`, and `StepResult` models
- `env.py`: core environment implementing `reset()`, `step()`, and `state()`
- `tasks.py`: 3 tasks and deterministic graders with partial rewards + penalties
- `inference.py`: baseline OpenAI inference script with strict structured logs
- `requirements.txt`: Python dependencies
- `Dockerfile`: containerized runtime for local + HF Space deployment

## Action Space

Supported tools in this environment:

- `query_order_db(order_id)`
- `read_return_policy()`
- `send_whatsapp_message(text, order_id?, mark_refunded?)`
- `escalate_to_human(reason?)`

## Observation Space

Observation mimics a WhatsApp webhook payload:

- `sender_id`, `message_body`, `timestamp`, `ticket_id`
- `crm_context` with order hint, current progress, and last tool result
- `available_tools` listing allowed actions

## Tasks and Graders

1. **Easy (`shipping_status_easy`)**
   - Objective: provide accurate delivery date.
   - Partial reward: DB lookup + correct final response date.

2. **Medium (`valid_refund_medium`)**
   - Objective: process valid refund safely.
   - Partial reward: DB query + policy read + refund state update.
   - Penalty: refund without required checks.

3. **Hard (`warranty_complaint_hard`)**
   - Objective: escalate out-of-warranty issue.
   - Partial reward: DB query + policy read + escalation.
   - Penalty: hallucinated discount or invalid refund.

All graders output score in `[0.0, 1.0]`.

## Local Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your_key>"
# Optional for OpenAI-compatible gateways:
# export API_BASE_URL="https://<provider>/v1"
# export HF_TOKEN="<token>"  # can be used instead of OPENAI_API_KEY
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Run API Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Server endpoints:
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`

## Pre-Submission Validation

Run one command before submitting:

```bash
python prevalidate.py
```

It checks:
- `openenv.yaml` critical keys (`spec_version`, `runtime`, `app`, `port`)
- API endpoint health (`/reset`, `/step`, `/state`, `/health`, `/metadata`, `/schema`)
- all 3 tasks have graders
- reward values remain in `[0.0, 1.0]`
- `inference.py` contains required `[START]`, `[STEP]`, `[END]` structured logs

## Required Log Format (Validator-Friendly)

`inference.py` emits strict structured logs:

- `[START] { ... }`
- `[STEP] { ... }` (for every step)
- `[END] { ... }`

## Docker

Build for Hugging Face compatible linux/amd64 (even on Apple Silicon):

```bash
docker buildx build --platform linux/amd64 -t whatsapp-triage-openenv:latest .
docker run --rm \
  -p 8000:8000 \
  whatsapp-triage-openenv:latest
```

Test containerized API:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

If you only need local native architecture testing:

```bash
docker build -t whatsapp-triage-openenv:local .
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" whatsapp-triage-openenv:local
```

## Notes

- `reset()` reinitializes CRM and all flags for a clean episode state.
- Graders are deterministic and designed for reproducible baseline scoring.
- Baseline uses model outputs when valid JSON is returned, with a deterministic fallback policy for reliability.
