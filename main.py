import base64
import httpx
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from io import BytesIO
from typing import Any, Dict, Optional
import easyocr
import numpy as np
from dateutil import parser
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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

# try in Windows local
# ROOT_DIR: str = app.root_path
# DATA_DIR: str = f"{ROOT_DIR}/data"
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

DATA_DIR: str = "/data"
DEV_EMAIL: str = "hariharan.chandran@straive.com"

# AI Proxy
AI_URL: str = "https://api.openai.com/v1"
AIPROXY_TOKEN: str = os.environ.get("AIPROXY_TOKEN")
AI_MODEL: str = "gpt-4o-mini"
AI_EMBEDDINGS_MODEL: str = "text-embedding-3-small"

# for debugging use LLM token
if not AIPROXY_TOKEN:
    AI_URL = "https://llmfoundry.straive.com/openai/v1"
    AIPROXY_TOKEN = os.environ.get("LLM_TOKEN")

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variables is missing")

APP_ID = "hari-tds2025-project1"


# POST `/run?task=<task description>`` Executes a plain‑English task.
# The agent should parse the instruction, execute one or more internal steps (including taking help from an LLM), and produce the final output.
# - If successful, return a HTTP 200 OK response
# - If unsuccessful because of an error in the task, return a HTTP 400 Bad Request response
# - If unsuccessful because of an error in the agent, return a HTTP 500 Internal Server Error response
# - The body may optionally contain any useful information in each of these cases
@app.post("/run")
def run_task(task: str) -> Response:
    try:
        if not task:
            raise ValueError("Task description is required")

        tool = get_task_tool(task, task_tools)
        return execute_tool_calls(tool)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def execute_tool_calls(tool: Dict[str, Any]):
    if tool and "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                if isinstance(function_args, dict):
                    return function_chosen(**function_args)

    raise NotImplementedError("Unknown task")


def parse_function_args(function_args: Optional[Any]):
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
    try:
        if not path:
            raise ValueError("File path is required")

        # dont allow path pout side DATA_DIR
        abs1 = os.path.abspath(path).lower()
        abs2 = os.path.abspath(DATA_DIR).lower()
        if not os.path.abspath(abs1).startswith(abs2):
            raise PermissionError("Acces denied")

        if not os.path.exists(path):
            raise FileNotFoundError("File not found")

        with open(path, "r") as f:
            content = f.read()

        return Response(content=content, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                        "description": "File path to format.",
                    }
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekday",
            "description": "Count the occurrences of a specific weekday in the provided file",
            "parameters": {
                "type": "object",
                "properties": {
                    "weekday": {
                        "type": "string",
                        "description": "Day of the week (in English)",
                    },
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the source file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["weekday", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort an array of contacts by first or last name",
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
                        "type": ["string", "null"],
                        "description": "Path to the source file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["order", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_logs",
            "description": "Write the first line of the **10** most recent `.log` files in the directory `/data/logs/`, most recent first",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of records to be listed",
                    },
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the directory containing log files. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["count", "source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_markdown_titles",
            "description": "Index Markdown (.md) files in the directory `/data/docs/` and extract their titles",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the directory containing Markdown files. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract the **sender's** email address in an email message from `/data/email.txt`",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the source file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_card_number",
            "description": "Extract the 16 digit code from the image",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the source image file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similar_comments",
            "description": "Find the most similar pair of comments",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the source file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ticket_sales",
            "description": "Calculate ticket sales for a specific item from a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The type of ticket item to calculate sales for",
                    },
                    "source": {
                        "type": ["string", "null"],
                        "description": "Path to the source file. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "destination": {
                        "type": ["string", "null"],
                        "description": "Path to the destination file. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["item", "source"],
                "additionalProperties": False,
            },
        },
    },
]


def get_task_tool(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": task}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def get_chat_completions(messages: list[Dict[str, Any]]):
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": messages,
        },
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def get_embeddings(text: str):
    response = httpx.post(
        f"{AI_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_EMBEDDINGS_MODEL,
            "input": text,
        },
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["data"][0]["embedding"]


def file_rename(name: str, suffix: str):
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
def format_file(source: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    result = subprocess.run(
        ["prettier", "--write", file_path],
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )

    if result.stderr:
        raise HTTPException(status_code=500, detail=result.stderr)

    return {"status": "success", "source": file_path}


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


def count_weekday(weekday: str, source: str = None, destination: str = None):
    weekday = normalize_weekday(weekday)
    weekday_index = day_names.index(weekday)

    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or file_rename(file_path, f"-{weekday}.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    with open(file_path, "r") as f:
        dates = [parser.parse(line.strip()) for line in f if line.strip()]

    day_count = sum(1 for d in dates if d.weekday() == weekday_index)

    with open(output_path, "w") as f:
        f.write(str(day_count))

    return {
        "message": f"{weekday} counted",
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
def sort_contacts(order: str, source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or file_rename(file_path, "-sorted.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    with open(file_path, "r") as f:
        contacts = json.load(f)

    key1 = "last_name" if order != "first_name" else "first_name"
    key2 = "last_name" if key1 == "first_name" else "first_name"

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
def write_recent_logs(count: int, source: str = None, destination: str = None):
    if count < 1:
        raise ValueError("Invalid count")

    if not source:
        raise ValueError("Source directory is required")

    if not os.path.isdir(source):
        raise FileNotFoundError("Source directory not found")

    output_path = destination or os.path.join(DATA_DIR, "logs-recent.txt")

    log_files = sorted(
        [os.path.join(source, f) for f in os.listdir(source) if f.endswith(".log")],
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
        "source": source,
        "destination": output_path,
        "status": "success",
    }


# A6. Index for Markdown (.md) files in /data/docs/
def extract_markdown_titles(source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or os.path.join(file_path, "index.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Directory not found")

    index = {}

    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                title = extract_title_from_markdown(file_path)
                if title:
                    relative_path = os.path.relpath(file_path, source)
                    relative_path = re.sub(r"[\\/]+", "/", relative_path)
                    index[relative_path] = title

    with open(output_path, "w") as f:
        json.dump(index, f, indent=4)

    return {
        "message": "Markdown titles extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


def extract_title_from_markdown(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# "):
                return line[2:].strip()


# A7. Extract the sender's email address from an email message
def extract_email_sender(source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or file_rename(file_path, "-sender.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

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
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_card_number(source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or file_rename(file_path, "-number.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Image file not found")

    reader = easyocr.Reader(["en"])
    results = reader.readtext(file_path, detail=0)

    extracted_text = "".join(results).replace(" ", "").replace("-", "")
    matches = re.findall(
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12}|3(?:0[0-5]|[68]\d)\d{11}|(?:2131|1800|35\d{3})\d{11})\b",
        extracted_text,
    )

    extracted_number = matches[0] if matches else "No credit card number found"

    with open(output_path, "w") as f:
        f.write(extracted_number)

    return {
        "message": "Credit card number extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


# A9. Simillar Comments
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def similar_comments(source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    file_path = source
    output_path = destination or file_rename(file_path, "-similar.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    with open(file_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines()]

    embeddings = [get_embeddings(comment) for comment in comments]

    most_similar_pair = max(
        (
            (
                comments[i],
                comments[j],
                cosine_similarity(embeddings[i], embeddings[j]),
            )
            for i in range(len(comments))
            for j in range(i + 1, len(comments))
        ),
        key=lambda x: x[2],
        default=(None, None, -1),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(most_similar_pair[:2]))

    return {
        "message": "Similar comments extracted",
        "source": file_path,
        "destination": output_path,
        "status": "success",
    }


# A10
def calculate_ticket_sales(item: str, source: str = None, destination: str = None):
    if not source:
        raise ValueError("Source file is required")

    db_path = source
    output_path = destination or file_rename(db_path, f"-{item}.txt")

    if not os.path.exists(db_path):
        raise FileNotFoundError("File not found")

    query = "SELECT SUM(units * price) FROM tickets WHERE LOWER(TRIM(type))=?"
    total_sales = 0

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (item.lower().strip(),))
        total_sales = cursor.fetchone()[0] or 0

    with open(output_path, "w") as f:
        f.write(str(total_sales))

    return {
        "message": "Ticket sales calculated",
        "source": db_path,
        "destination": output_path,
        "status": "success",
    }


# Installion of data is done through Dockerfile
# initialize_data()
