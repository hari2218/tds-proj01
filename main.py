from fastapi import FastAPI, HTTPException
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

app = FastAPI()

DATA_DIR = "/data"


@app.post("/run")
def run_task(task: str):
    try:
        if "install uv" in task:
            return install_uv(task)

        elif "format" in task:
            return format_markdown(task)

        elif "count Wednesdays" in task:
            return count_wednesdays(task)

        elif "sort contacts" in task:
            return sort_contacts(task)

        elif "first line of logs" in task:
            return extract_recent_logs(task)

        elif "extract markdown titles" in task:
            return extract_markdown_titles(task)

        elif "extract email" in task:
            return extract_email_sender(task)

        elif "extract credit card" in task:
            return extract_credit_card(task)

        elif "find similar comments" in task:
            return find_similar_comments(task)

        elif "calculate ticket sales" in task:
            return calculate_ticket_sales(task)

        else:
            return {"message": "Unknown task", "status": "error"}, 400

    except Exception as e:
        return {"message": str(e), "status": "agent error"}, 500


# A1
def install_uv(task):
    os.system(
        "uv venv venv && source venv/bin/activate && uv pip install -r requirements.txt"
    )
    subprocess.run(["python", "datagen.py", os.getenv("USER_EMAIL")])
    return {"message": "uv installed and datagen.py executed", "status": "success"}


def format_markdown(task):
    raise NotImplementedError

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


# B1 & B2
@app.get("/read")
def read_file(path: str):
    if not path.startswith(DATA_DIR):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        with open(path, "r") as f:
            return f.read()

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
