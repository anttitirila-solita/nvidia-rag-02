import os
import shutil
import requests
import sqlite3
from pymilvus import connections, Collection, utility

# Configuration - Update if needed
# WEBUI_API_URL is reserved for future use (e.g., API calls to Open WebUI)
WEBUI_API_URL = "http://open-webui:8080"
LOCAL_UPLOADS_PATH = "/app/backend/data/uploads"  # Adjust to match your volume mount
SQLITE_DB_PATH = "/app/backend/data/webui.db"     # Adjust accordingly

MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
VECTOR_DB_NAME = "default"
COLLECTION_PREFIX = "open_webui_file"

# Delete local files
def delete_local_files(path):
    print(f"Deleting files in {path}...")
    shutil.rmtree(path)
    os.makedirs(path)

# Delete relevant parts of the SQLite db
# NOTE: Currently, only the 'file' and 'chat' tables are being reset.
# If there are other tables in the database that may contain sensitive information,
# please review the database schema and update this function accordingly to prevent data leaks.
def reset_sqlite_db(db_path):
    print(f"Resetting Open WebUI database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Delete all rows from 'file' and 'chat' tables
        cursor.execute("DELETE FROM file;")
        cursor.execute("DELETE FROM chat;")
        
        # Commit the changes
        conn.commit()
        print("All rows removed from 'files' and 'chat' tables.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()

    finally:
        # Close the connection
        conn.close()

def reset_milvus_db(prefix, vector_db_name="default"):
    print(f"Resetting Milvus vector database...")

    # Connect to Milvus
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        db_name=vector_db_name  # Specify the database
    )

    # List all collections in the specified DB
    all_collections = utility.list_collections()
    
    # Filter collections that start with the given prefix
    target_collections = [name for name in all_collections if name.startswith(prefix)]

    if not target_collections:
        print(f"No collections found with prefix '{prefix}'.")
        return

    # Drop each matching collection
    for name in target_collections:
        print(f"Dropping collection: {name}")
        utility.drop_collection(name)
        print(f" → Collection '{name}' dropped.")

# Main execution
if __name__ == "__main__":
    delete_local_files(LOCAL_UPLOADS_PATH)
    reset_sqlite_db(SQLITE_DB_PATH)
    reset_milvus_db(COLLECTION_PREFIX, VECTOR_DB_NAME)
