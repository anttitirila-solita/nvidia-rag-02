import requests
import time
import json
from dotenv import load_dotenv
import os

# https://docs.openwebui.com/getting-started/api-endpoints/

# Configuration
# Load API key from .env file
load_dotenv("../.env")
API_KEY = os.getenv("OPEN_WEBUI_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in ../.env file")
BASE_URL = "http://localhost:3000"

MODEL = "gemma3:27b-it-qat"
FILE_PATH = "./data/nokia_results_2025_q1.pdf"  # <-- Replace this with whatever file you want to test
QUESTIONS = [
    "What is the document about?",
    "List key financial figures mentioned.",
    "Does it mention future business strategy?",
    "Summarize the document in 3 sentences.",
]

# Upload the file to Open WebUI and return file_id
def upload_file(file_path):
    print("[*] Uploading document...")
    with open(file_path, "rb") as f:
        files = {'file': (file_path, f)}
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json"
        }
        response = requests.post(f"{BASE_URL}/api/v1/files/", headers=headers, files=files)
    response.raise_for_status()
    file_info = response.json()
    file_id = file_info.get("id")
    print(f"[+] Uploaded with file_id: {file_id}")
    return file_id

# Ask a question referencing the uploaded file
def chat_with_file(question, file_id):
    url = f"{BASE_URL}/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "files": [{"type": "file", "id": file_id}]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# List uploaded files
def list_uploaded_files():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    response = requests.get(f"{BASE_URL}/api/v1/files/", headers=headers)
    response.raise_for_status()

    return response.json()


# Main execution
if __name__ == "__main__":

    files = list_uploaded_files()
    print("Available uploaded files:")
    for f in files:
        print(f"- ID: {f['id']}")
        print(f"  Name: {f['filename']}")
        print(f"  Size: {f['meta']['size']} bytes")
        content_preview = f['data']['content']
        if content_preview.strip():
            print(f"Content preview: {content_preview[:100]!r}")
        else:
            print("No content extracted")
        print()

    try:
        file_id = upload_file(FILE_PATH)

        print("[*] Asking questions...")
        for question in QUESTIONS:
            answer = chat_with_file(question, file_id)
            print(f"[Q] {question}\n[A] {answer}\n")
            time.sleep(1)

    except Exception as e:
        print(f"[!] Error: {e}")