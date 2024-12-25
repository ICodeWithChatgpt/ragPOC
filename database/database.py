import sqlite3
import hashlib
import json
from datetime import datetime
import numpy as np
from API.openai_utils import generate_embedding
from helpers.helper_functions import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

DB_NAME = "content_store.db"

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT,
        url TEXT,
        raw_content TEXT,
        metadata TEXT,
        tags TEXT,
        summary TEXT,
        normalized_content TEXT,
        vectorized_content TEXT,
        created_at TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT,
        chunk TEXT,
        vectorized_content TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def add_normalized_content_column():
    """Add normalized_content column to the content table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(content)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'normalized_content' not in columns:
        cursor.execute("ALTER TABLE content ADD COLUMN normalized_content TEXT")
    conn.commit()
    conn.close()


def add_document_id_column():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Check if the column already exists
    cursor.execute("PRAGMA table_info(content)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'document_id' not in columns:
        cursor.execute("ALTER TABLE content ADD COLUMN document_id TEXT")
        conn.commit()

    conn.close()

def store_in_db(url, raw_content, metadata, tags, summary, normalized_content, vectorized_content):
    """Store processed content in SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Generate a unique ID based on content
    document_id = hashlib.sha256((url or raw_content).encode()).hexdigest()

    # Ensure vectorized_content is a numpy array
    if not isinstance(vectorized_content, np.ndarray):
        vectorized_content = np.array(vectorized_content)

    # Store the whole document in the original database
    cursor.execute("""
    INSERT INTO content (document_id, url, raw_content, metadata, tags, summary, normalized_content, vectorized_content, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (document_id, url, raw_content, metadata, tags, summary, normalized_content, json.dumps(vectorized_content.tolist()), datetime.now().isoformat()))

    conn.commit()

    # Tokenize the normalized content
    tokens = word_tokenize(normalized_content)
    chunk_size = 200
    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    # Split the vectorized content into chunks
    vectorized_chunks = [vectorized_content[i:i + chunk_size] for i in range(0, len(vectorized_content), chunk_size)]

    # Store each chunk as a separate entry
    for chunk, chunk_vector in zip(chunks, vectorized_chunks):
        vectorized_content_json = json.dumps(chunk_vector.tolist())
        cursor.execute("""
            INSERT INTO chunks (document_id, chunk, vectorized_content, created_at)
            VALUES (?, ?, ?, ?)
            """, (document_id, chunk, vectorized_content_json, datetime.now().isoformat()))

    conn.commit()
    conn.close()

# Function: Search Vectorized Content
def search_vectorized_content(query, metadata_similarity_threshold=0.80, vectorized_similarity_threshold=0.80):
    """Search DB for content matching the query using semantic similarity."""
    query_embedding = generate_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return None

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        # Fetch all metadata from the original database
        cursor.execute("SELECT document_id, tags, summary FROM content")
        metadata_results = cursor.fetchall()

        relevant_context = {}
        for row in metadata_results:
            document_id, tags, summary = row
            metadata = f"{tags} {summary}"
            metadata_embedding = generate_embedding(metadata)
            metadata_similarity = cosine_similarity(query_embedding, metadata_embedding)

            # Print the metadata similarity
            print(f"Document ID: {document_id}, Metadata Similarity: {metadata_similarity:.2f}")

            # Threshold for metadata relevance (adjust as needed)
            if metadata_similarity > metadata_similarity_threshold:
                # Fetch chunks related to the document_id from the chunks table
                cursor.execute("SELECT chunk, vectorized_content FROM chunks WHERE document_id = ?", (document_id,))
                chunk_results = cursor.fetchall()

                for chunk, vectorized_content_json in chunk_results:
                    try:
                        # Load the vectorized content (stored as JSON)
                        chunk_embedding = np.array(json.loads(vectorized_content_json))
                        similarity = cosine_similarity(query_embedding, chunk_embedding)
                        print(f"Chunk similarity: {float(similarity):.2f}")

                        if float(similarity) >= vectorized_similarity_threshold:
                            if document_id not in relevant_context:
                                relevant_context[document_id] = f"Tags: {tags}, Summary: {summary}\n"
                            relevant_context[document_id] += f"Similarity: {float(similarity):.2f}, Sentence: {chunk}\n"
                    except Exception as e:
                        print(f"Error processing vectorized content: {e}")
                        continue

        if relevant_context:
            return "\n\n".join(relevant_context.values())
        return None
    finally:
        conn.close()