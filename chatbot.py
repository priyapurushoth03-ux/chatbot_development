
import csv
import os
import uuid
from datetime import datetime

import ollama

# Project folders and files
os.makedirs("dataset", exist_ok=True)

QUESTIONS_FILE = "dataset/sample_questions.csv"
HISTORY_FILE = "dataset/chat_history.csv"
RESPONSES_FILE = "dataset/responses.csv"
LOG_FILE = "dataset/chatbot_logs.txt"

MODEL = "llama3.2:1b"

# 20 sample questions for the Education domain
sample_questions = [
    "What is Python?",
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is a Large Language Model?",
    "What is a database?",
    "What is DBMS?",
    "What is SQL?",
    "What is a primary key?",
    "What is a foreign key?",
    "What is normalization in DBMS?",
    "What is a Python variable?",
    "What is a Python function?",
    "What is a loop in Python?",
    "What is a list in Python?",
    "What is an algorithm?",
    "What is a data structure?",
    "What is cloud computing?",
    "What is cybersecurity?",
    "Explain the difference between AI and ML.",
    "Write a simple Python program to add two numbers."
]

# Create the sample questions CSV
if not os.path.exists(QUESTIONS_FILE):
    with open(QUESTIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question"])
        for question in sample_questions:
            writer.writerow([question])

# Create CSV files with column headings
for file_name in [HISTORY_FILE, RESPONSES_FILE]:
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "session_id", "user_question",
                "chatbot_response"
            ])

session_id = str(uuid.uuid4())[:8]
conversation = []

system_prompt = """
You are an Education Assistant chatbot for college students.
Answer questions about education, Python, AI, and DBMS.
Use simple English and clear examples.
For coding questions, provide correct and readable code.
If you are unsure, say that you are unsure instead of inventing facts.
Keep answers relevant to the student's question.
"""

print("=" * 50)
print("       EDUCATION AI CHATBOT")
print("       Powered by Python and Ollama")
print("=" * 50)
print("Ask questions about Python, AI, DBMS and education.")
print("Type 'exit' to stop the chatbot.\n")

with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"\nNew session: {session_id}\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Chatbot: Thank you! Goodbye.")
        break

    if not user_input:
        print("Please enter a question.")
        continue

    timestamp = datetime.now().isoformat(timespec="seconds")

    conversation.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation
            ]
        )

        answer = response["message"]["content"]
        print(f"\nChatbot: {answer}\n")

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        record = [
            timestamp, session_id, user_input, answer
        ]

        for file_name in [HISTORY_FILE, RESPONSES_FILE]:
            with open(
                file_name, "a", newline="", encoding="utf-8"
            ) as f:
                csv.writer(f).writerow(record)

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] You: {user_input}\n")
            log.write(f"[{timestamp}] Bot: {answer}\n")

    except Exception as error:
        print("Error: Could not get a response from Ollama.")
        print("Check whether Ollama is running and the model is installed.")
        print("Details:", error)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] ERROR: {error}\n")
