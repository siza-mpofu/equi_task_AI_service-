from __future__ import annotations

from typing import Any, Dict
import json

from openai import OpenAI

client = OpenAI()


def task_simplifier_schema() -> Dict[str, Any]:
    """
    JSON Schema for Structured Outputs.
    Keep it strict so the model returns reliably parseable JSON.
    """
    return {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "simplified_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer", "minimum": 1},
                        "instruction": {"type": "string", "minLength": 1},
                    },
                    "required": ["step_number", "instruction"],
                    "additionalProperties": False,
                },
            },
            "clarification_needed": {"type": "boolean"},
            "clarification_question": {"type": "string"},
        },
        "required": [
            "task_id",
            "confidence_score",
            "simplified_steps",
            "clarification_needed",
            "clarification_question",
        ],
        "additionalProperties": False,
    }


def call_llm_structured(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Real AI call using OpenAI Responses API with Structured Outputs (JSON Schema).
    """
    schema = task_simplifier_schema()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "task_simplifier_response",
                "schema": schema,
                "strict": True,
            }
        },
        temperature=temperature,
    )

    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model returned non-JSON output: {e}") from e
