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
from dateutil import parser
import sys
import logging

app = FastAPI()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

ROOT_DIR: str = app.root_path
DATA_DIR: str = f"{ROOT_DIR}/data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

DEV_EMAIL: str = "hariharan.chandran@straive.com"

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
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekday",
            "description": "Count the occurrences of a specific weekday in the file `/data/dates.txt`",
            "parameters": {
                "type": "object",
                "properties": {
                    "weekday": {"type": "string", "description": "Day of the week"},
                    "source": {
                        "type": "string",
                        "description": "Path to the source file (optional)",
                        "nullable": True,
                    },
                },
                "required": ["weekday", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort an array of contacts by first or last name, in the file `/data/contacts.json` to `/data/contacts-sorted.json`",
            "parameters": {
                "type": "object",
                "properties": {
                    "order": {
                        "type": "string",
                        "description": "Sorting order, based on name",
                        "enum": ["last_name", "first_name"],
                        "default": "last_name",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the source file (optional)",
                        "nullable": True,
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the destination file (optional)",
                        "nullable": True,
                    },
                },
                "required": ["order", "source", "destination"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
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

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


# A1. Data initialization
def initialize_data():
    logging.info(f"DATA - {DATA_DIR}")
    logging.info(f"USER - {DEV_EMAIL}")

    try:
        # Ensure the 'uv' package is installed
        try:
            import uv

        except ImportError:
            logging.info("'uv' package not found. Installing...")

            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "uv"]
            )

            import uv

        # Run the data generation script
        result = subprocess.run(
            [
                "uv",
                "run",
                "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py",
                f"--root={DATA_DIR}",
                DEV_EMAIL,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logging.info("Data initialization completed successfully.")

        else:
            logging.error(
                f"Data initialization failed with return code {result.returncode}"
            )
            logging.error(f"Error output: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error: {e}")
        logging.error(f"Output: {e.output}")

    except Exception as e:
        logging.error(f"Error in initializing data: {e}")


# A2. Format a file using prettier
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

        return {"message": "File formatted", "source": file_path, "status": "success"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


# A3. Count the number of week-days in the list of dates
day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def count_weekday(weekday: str, source: Optional[str] = None) -> dict:
    weekday = normalize_weekday(weekday)
    weekday_index = day_names.index(weekday)

    file_path: str = source or os.path.join(DATA_DIR, "dates.txt")
    output_path: str = os.path.join(DATA_DIR, f"dates-{weekday}.txt".lower())

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        dates = [parser.parse(line.strip()) for line in f if line.strip()]

    day_count = sum(1 for d in dates if d.weekday() == weekday_index)

    with open(output_path, "w") as f:
        f.write(str(day_count))

    return {
        "message": f"{weekday} counted",
        "count": day_count,
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


def normalize_weekday(weekday):
    if isinstance(weekday, int):  # If input is an integer (0-6)
        return day_names[weekday % 7]

    elif isinstance(weekday, str):  # If input is a string
        weekday = weekday.strip().lower()
        days = {day.lower(): day for day in day_names}
        short_days = {day[:3].lower(): day for day in day_names}

        if weekday in days:
            return days[weekday]

        elif weekday in short_days:
            return short_days[weekday]

    raise ValueError("Invalid weekday input")


# A4. Sort the array of contacts by last name and first name
def sort_contacts(
    order: str = "last_name",
    source: Optional[str] = None,
    destination: Optional[str] = None,
) -> dict:
    file_path = source or os.path.join(DATA_DIR, "contacts.json")
    output_path = destination or os.path.join(DATA_DIR, "contacts-sorted.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        contacts = json.load(f)

    key1: str = "last_name" if order != "first_name" else "first_name"
    key2: str = "last_name" if key1 == "first_name" else "first_name"

    contacts.sort(key=lambda x: (x.get(key1, ""), x.get(key2, "")))

    with open(output_path, "w") as f:
        json.dump(contacts, f, indent=4)

    return {
        "message": "Contacts sorted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


def extract_recent_logs(task):
    raise NotImplementedError


def extract_markdown_titles(task):
    raise NotImplementedError


def extract_credit_card(task):
    raise NotImplementedError


def find_similar_comments(task):
    raise NotImplementedError


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


initialize_data()
