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
import re
import base64
from PIL import Image
from io import BytesIO
import easyocr

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
AIPROXY_TOKEN: str = os.environ.get("AIPROXY_TOKEN")
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
                    "source": {
                        "type": "string",
                        "description": "File path to format",
                    }
                },
                "required": ["source"],
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
                    "weekday": {
                        "type": "string",
                        "description": "Day of the week (in English)",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the source file",
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
            "description": "Sort an array of contacts by first or last name, in the file `/data/contacts.json`",
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
                        "description": "Path to the source file",
                        "nullable": True,
                    },
                },
                "required": ["order", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_logs",
            "description": "Write the first line of the **10** most recent `.log` files in `/data/logs/`, most recent first",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of records to be listed",
                    },
                    "source": {
                        "type": "string",
                        "description": "Path to the directory containing log files",
                        "nullable": True,
                    },
                },
                "required": ["count", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_markdown_titles",
            "description": "Index Markdown (.md) files in `/data/docs/` and extract their titles",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the directory containing Markdown files",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract the **sender's** email address from an email message from `/data/email.txt`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source file containing the email message",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract the 16 digit code from the image `/data/credit_card.png`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source image file containing the credit card",
                        "nullable": True,
                    }
                },
                "required": ["source"],
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


def get_chat_completions(messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": messages,
        },
    )

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def file_rename(name: str, suffix: str) -> str:
    return (re.sub(r"\.(\w+)$", "", name) + suffix).lower()


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
def format_file(source: str) -> dict:
    file_path = source or os.path.join(DATA_DIR, "format.md")

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


def count_weekday(weekday: str, source: str = None) -> dict:
    weekday = normalize_weekday(weekday)
    weekday_index = day_names.index(weekday)

    file_path: str = source or os.path.join(DATA_DIR, "dates.txt")
    output_path: str = file_rename(file_path, f"-{weekday}.txt")

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
def sort_contacts(order: str, source: str) -> dict:
    order = order or "last_name"
    file_path = source or os.path.join(DATA_DIR, "contacts.json")
    output_path = file_rename(file_path, "-sorted.json")

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


# A5. Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first
def write_recent_logs(count: int, source: str):
    file_path: str = source or os.path.join(DATA_DIR, "logs")
    file_dir_name: str = os.path.dirname(file_path)
    output_path: str = os.path.join(DATA_DIR, f"{file_dir_name}-recent.txt")

    if count < 1:
        raise HTTPException(status_code=400, detail="Invalid count")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    log_files = sorted(
        [
            os.path.join(file_path, f)
            for f in os.listdir(file_path)
            if f.endswith(".log")
        ],
        key=os.path.getmtime,
        reverse=True,
    )

    with open(output_path, "w") as out:
        for log_file in log_files[:count]:
            with open(log_file, "r") as f:
                first_line = f.readline().strip()
                out.write(f"{first_line}\n")

    return {
        "message": "Recent logs written",
        "log_dir": file_path,
        "output_file": output_path,
        "status": "success",
    }


# A6. Index for Markdown (.md) files in /data/docs/
def extract_markdown_titles(source: str):
    file_path = source or os.path.join(DATA_DIR, "docs")
    output_path = os.path.join(file_path, "index.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    index = {}
    collect_markdown_titles(file_path, index)

    with open(output_path, "w") as f:
        json.dump(index, f, indent=4)

    return {
        "message": "Markdown titles extracted",
        "file_dir": file_path,
        "index_file": output_path,
        "status": "success",
    }


def collect_markdown_titles(directory: str, index: dict):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    title = None
                    for line in f:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break

                    if title:
                        relative_path = os.path.relpath(file_path, directory)
                        relative_path = re.sub(r"[\\/]+", "/", relative_path)
                        index[relative_path] = title


# A7. Extract the sender's email address from an email message
def extract_email_sender(source: str):
    file_path = source or os.path.join(DATA_DIR, "email.txt")
    output_path = file_rename(file_path, "-sender.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "r") as f:
        email_content = f.read()

    response = get_chat_completions(
        [
            {"role": "system", "content": "Extract the sender's email."},
            {"role": "user", "content": email_content},
        ]
    )

    extracted_email = response["content"].strip()

    with open(output_path, "w") as f:
        f.write(extracted_email)

    return {
        "message": "Email extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


# A8. Extract credit card number.
def encode_image(image_path: str, format: str):
    image = Image.open(image_path)

    buffer = BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return base64_image


def extract_credit_card_number(source: str):
    file_path = source or os.path.join(DATA_DIR, "credit_card.png")
    output_path = file_rename(file_path, "-number.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    # Taking more time
    reader = easyocr.Reader(["en"])
    results = reader.readtext(file_path, detail=0)

    extracted_text = "\n".join(results)
    extracted_text = re.sub(r"[- ]+", "", extracted_text)
    matches = re.findall(
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12}|3(?:0[0-5]|[68]\d)\d{11}|(?:2131|1800|35\d{3})\d{11})\b",
        extracted_text,
    )

    extracted_number = (
        matches[0] if (matches and len(matches) > 0) else "No credit card number found"
    )

    ## hard to install pytesseract
    # image = Image.open(file_path)
    # extracted_text = pytesseract.image_to_string(image)

    ## below not working because of sensity data
    # base64_image = encode_image(file_path, "PNG")
    # image_url = f"data:image/png;base64,{base64_image}"
    #
    # response = get_chat_completions(
    #     [
    #         {
    #             "role": "system",
    #             "content": "Extract the credit card number from the image.",
    #         },
    #         {
    #             "role": "user",
    #             "content": [{"type": "image_url", "image_url": {"url": image_url}}],
    #         },
    #     ]
    # )
    #
    # extracted_number = response["content"].strip()

    with open(output_path, "w") as f:
        f.write(extracted_number)

    return {
        "message": "Credit card number extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


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
