# check_financebench_trace.py
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

backend_path = PROJECT_ROOT / "backend"
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

from agent import chat_with_agent

question = "Has AMD's gross margin improved or declined in FY2022 compared with FY2021?"

result = chat_with_agent(
    user_text=question,
    user_id="debug_user",
    session_id="debug_financebench_trace",
)

print("Response:")
print(result.get("response") if isinstance(result, dict) else result)

print("\nRAG trace keys:")
rag_trace = result.get("rag_trace", {}) if isinstance(result, dict) else {}
print(rag_trace.keys())

print("\nSelected docs:")
print(json.dumps(rag_trace.get("selected_docs", []), ensure_ascii=False, indent=2)[:3000])

print("\nSelected pages:")
print(json.dumps(rag_trace.get("selected_pages", []), ensure_ascii=False, indent=2)[:5000])

print("\nFinal evidence pack:")
print(json.dumps(rag_trace.get("final_evidence_pack", []), ensure_ascii=False, indent=2)[:8000])