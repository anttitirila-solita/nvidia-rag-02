import os
import shutil
import requests

# Configurable values
WEBUI_API_URL = "http://open-webui:8080"
LOCAL_UPLOADS_PATH = "/app/backend/data/uploads"  # Adjust to match your volume mount
SQLITE_DB_PATH = "/app/backend/data/webui.db"     # Adjust accordingly

# Delete local files
def delete_local_files(path):
    print(f"Deleting files in {path}...")
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)

# Delete vector DB
# TODO: Deleting the DB file altogether breaks the app and resets
# settings including users. Need a script to do cleanup instead.
# Clear at least these:
# - chat (chat history)
# - file (contents of vectorized files)
def reset_sqlite_db(path):
    print(f"Resetting vector DB at {path}...")
    if os.path.exists(path):
        os.remove(path)

# TODO: Clear Milvus collections
def reset_milvus_db():
    pass 

# Delete chat history via API
def clear_chat_history():
    print("Clearing all chat history...")
    # You must have admin-level access or user tokens if auth is required
    users = requests.get(f"{WEBUI_API_URL}/users").json()
    for user in users:
        user_id = user["id"]
        print(f"Clearing chats for user {user_id}")
        chats = requests.get(f"{WEBUI_API_URL}/users/{user_id}/chats").json()
        for chat in chats:
            chat_id = chat["id"]
            requests.delete(f"{WEBUI_API_URL}/users/{user_id}/chats/{chat_id}")

# Main execution
if __name__ == "__main__":
    delete_local_files(LOCAL_UPLOADS_PATH)
    reset_sqlite_db(VECTOR_DB_PATH)
    # clear_chat_history()
