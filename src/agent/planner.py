from __future__ import annotations

import json
import os
from typing import Any

import requests


def decide(observation: dict[str, Any], use_llm: bool = False) -> dict[str, str]:
    """Constrained decision; LLM sees diagnostics, never credentials or raw SCADA."""
    allowed = ["retry_older", "halt"] if observation["weather_failed"] else ["publish", "halt"]
    if not use_llm:
        return {"action": allowed[0], "reason": "Explicit recovery/quality policy", "planner": "deterministic"}
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("--llm requires OPENAI_API_KEY in the environment")
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"action": {"type": "string", "enum": allowed}, "reason": {"type": "string"}},
              "required": ["action", "reason"]}
    response = requests.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"}, timeout=(10, 60),
        json={"model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
              "messages": [{"role": "system", "content":
                "You supervise a wind-power forecasting pipeline. Select one permitted action. "
                "On unavailable weather retry an older archived release if allowed. "
                "Publish validated numerical forecasts with uncertainty flags; never modify numbers. "
                "Halt for an unrecoverable integrity failure. Briefly explain in Russian."},
                {"role": "user", "content": json.dumps(observation)}],
              "response_format": {"type": "json_schema", "json_schema": {
                  "name": "forecast_decision", "strict": True, "schema": schema}}})
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    if message.get("refusal"):
        raise RuntimeError("Planner refused the decision")
    result = json.loads(message["content"])
    if result.get("action") not in allowed or not isinstance(result.get("reason"), str):
        raise ValueError("Invalid planner action")
    return {**result, "planner": "openai"}
