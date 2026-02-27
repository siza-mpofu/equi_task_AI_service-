from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.task_simplifier import simplify_task

app = FastAPI(title="EquiTask AI Task Simplifier", version="1.0.0")


class TaskSimplifyRequest(BaseModel):
    task_id: str
    task_text: str = Field(min_length=1)
    task_type: str = "Unknown"
    accessibility_mode: str = "Standard"
    model: Optional[str] = None  # allow override; default handled in simplify_task()


@app.post("/ai/task-simplify")
def task_simplify(req: TaskSimplifyRequest) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the server.")
    return simplify_task(req.model_dump())
