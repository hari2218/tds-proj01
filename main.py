from fastapi import FastAPI, HTTPException, Response
import subprocess
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
import markdown2
from bs4 import BeautifulSoup
from openai import OpenAI
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from typing import Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Don't worry about root path
# ROOT_DIR: str = app.root_path or '.'
# DATA_DIR: str = f"{ROOT_DIR}/data"
#
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# AI Proxy
AI_URL: str = "https://api.openai.com/v1"
AIPROXY_TOKEN: Optional[str] = os.environ.get("AIPROXY_TOKEN")
AI_MODEL: str = "gpt-4o-mini"

# for debugging use LLM token
if not AIPROXY_TOKEN:
    AI_URL = "https://llmfoundry.straive.com/openai/v1"
    AIPROXY_TOKEN = os.environ.get("LLM_TOKEN")

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variables is missing")


# POST `/run?task=<task description>`` Executes a plain‑English task.
# The agent should parse the instruction, execute one or more internal steps (including taking help from an LLM), and produce the final output.
# - If successful, return a HTTP 200 OK response
# - If unsuccessful because of an error in the task, return a HTTP 400 Bad Request response
# - If unsuccessful because of an error in the agent, return a HTTP 500 Internal Server Error response
# - The body may optionally contain any useful information in each of these cases
@app.post("/run")
def run_task(task: str):
    if not task:
        raise HTTPException(status_code=400, detail="Task description is required")

    try:
        tool = get_task_tool(task, task_tools)
        return execute_tool_calls(tool)

    except Exception as e:
        detail: str = e.detail if hasattr(e, "detail") else str(e)

        raise HTTPException(status_code=500, detail=detail)


def execute_tool_calls(tool: Dict[str, Any]) -> Any:
    if "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                if isinstance(function_args, dict):
                    return function_chosen(**function_args)

    raise HTTPException(status_code=400, detail="Unknown task")


def parse_function_args(function_args: Optional[Any]) -> Dict[str, Any]:
    if function_args is not None:
        if isinstance(function_args, str):
            function_args = json.loads(function_args)

        elif not isinstance(function_args, dict):
            function_args = {"args": function_args}
    else:
        function_args = {}

    return function_args


# GET `/read?path=<file path>` Returns the content of the specified file.
# This is critical for verification of the exact output.
# - If successful, return a HTTP 200 OK response with the file content as plain text
# - If the file does not exist, return a HTTP 404 Not Found response and an empty body
@app.get("/read")
def read_file(path: str) -> Response:
    if not path:
        raise HTTPException(status_code=400, detail="File path is required")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(path, "r") as f:
            content = f.read()
        return Response(content=content, media_type="text/plain")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Task implementations
task_tools = [
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a file using prettier",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path to format",
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def get_task_tool(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": task}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )

    return response.json()["choices"][0]["message"]


# Format a file using prettier
def format_file(file_path: str) -> dict:
    if not file_path:
        raise HTTPException(status_code=400, detail="File path is required")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        result = subprocess.run(
            ["prettier", "--write", file_path],
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

        if result.stderr:
            raise HTTPException(status_code=500, detail=result.stderr)

        return {"message": "File formatted", "status": "success"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_recent_logs(task):
    raise NotImplementedError


def extract_markdown_titles(task):
    raise NotImplementedError


def extract_credit_card(task):
    raise NotImplementedError


def find_similar_comments(task):
    raise NotImplementedError


# A3
def count_wednesdays(task):
    file_path = os.path.join(DATA_DIR, "dates.txt")
    output_path = os.path.join(DATA_DIR, "dates-wednesdays.txt")

    with open(file_path, "r") as f:
        dates = [
            datetime.strptime(line.strip(), "%Y-%m-%d") for line in f if line.strip()
        ]

    wednesday_count = sum(1 for d in dates if d.weekday() == 2)

    with open(output_path, "w") as f:
        f.write(str(wednesday_count))

    return {
        "message": "Wednesdays counted",
        "count": wednesday_count,
        "status": "success",
    }


# A4
def sort_contacts(task):
    file_path = os.path.join(DATA_DIR, "contacts.json")
    output_path = os.path.join(DATA_DIR, "contacts-sorted.json")

    with open(file_path, "r") as f:
        contacts = json.load(f)

    contacts.sort(key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))

    with open(output_path, "w") as f:
        json.dump(contacts, f, indent=4)

    return {"message": "Contacts sorted", "status": "success"}


# A7
def extract_email_sender(task):
    file_path = os.path.join(DATA_DIR, "email.txt")
    output_path = os.path.join(DATA_DIR, "email-sender.txt")

    with open(file_path, "r") as f:
        email_content = f.read()

    client = OpenAI(api_key="YOUR_API_KEY")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract the sender's email."},
            {"role": "user", "content": email_content},
        ],
    )

    extracted_email = response.choices[0].message["content"].strip()

    with open(output_path, "w") as f:
        f.write(extracted_email)

    return {"message": "Email extracted", "email": extracted_email, "status": "success"}


# A10
def calculate_ticket_sales(task):
    db_path = os.path.join(DATA_DIR, "ticket-sales.db")
    output_path = os.path.join(DATA_DIR, "ticket-sales-gold.txt")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0] or 0

    conn.close()

    with open(output_path, "w") as f:
        f.write(str(total_sales))

    return {
        "message": "Sales calculated",
        "total_sales": total_sales,
        "status": "success",
    }
