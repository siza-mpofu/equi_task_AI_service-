## EquiTask AI Task Simplifier Service (Final)

### What this service does
- Exposes: POST /ai/task-simplify
- Uses: OpenAI (real AI) with Structured Outputs (JSON Schema)
- Includes: validation, confidence thresholds, retry once, fallback logic

### Setup
1) Install dependencies:
   pip install -r requirements.txt

2) Set environment variable:
   - copy .env.example to .env
   - set OPENAI_API_KEY in your environment (do NOT commit .env)

### Run
uvicorn app:app --reload --port 8000

### Test (cURL)
curl -X POST http://localhost:8000/ai/task-simplify \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"T001",
    "task_text":"Prepare monthly stakeholder report",
    "task_type":"Reporting",
    "accessibility_mode":"Simplified"
  }'
