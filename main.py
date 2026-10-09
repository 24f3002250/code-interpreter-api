
import os
import io
import contextlib
import traceback
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Code Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}


def analyze_error(code: str, error_text: str) -> list[int]:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        print("AI analysis failed: AIPIPE_TOKEN is missing")
        return []

    try:
        client = OpenAI(
            api_key=token,
            base_url="https://aipipe.org/openrouter/v1",
        )

        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Find the 1-based line numbers in the submitted "
                        "Python code responsible for the traceback. "
                        'Return JSON only: {"error_lines": [2]}. '
                        'If unknown, return {"error_lines": []}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Code:\n{code}\n\nTraceback:\n{error_text}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content or '{"error_lines": []}')
        lines = data.get("error_lines", [])

        if not isinstance(lines, list):
            return []

        code_line_count = len(code.splitlines())
        return [
            line for line in lines
            if isinstance(line, int)
            and not isinstance(line, bool)
            and 1 <= line <= code_line_count
        ]

    except Exception as exc:
        print(f"AI analysis failed: {exc}")
        return []


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    output = io.StringIO()

    try:
        with contextlib.redirect_stdout(output):
            exec(compile(request.code, "<user_code>", "exec"), {})

        return {
            "error": [],
            "result": output.getvalue(),
        }

    except Exception:
        error_text = traceback.format_exc()
        error_lines = analyze_error(request.code, error_text)

        return {
            "error": error_lines,
            "result": error_text,
        }
