import os
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
    # Delete the existing database file
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)

    # Create a new database with the updated structure
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create the content table with the updated structure
    cursor.execute("""
    CREATE TABLE content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT,
        url TEXT,
        raw_content TEXT,
        metadata TEXT,
        tags TEXT,
        summary TEXT,
        normalized_content TEXT,
        created_at TEXT
    )
    """)

    # Create the chunks table (unchanged)
    cursor.execute("""
    CREATE TABLE chunks (
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

DB_NAME = "content_store.db"

def store_in_db(url, raw_content, metadata, tags, summary, normalized_content, chunks, vectorized_chunks):
    """Store processed content in SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Generate a unique ID based on content
    document_id = hashlib.sha256((url or raw_content).encode()).hexdigest()

    # Store the whole document in the original database
    cursor.execute("""
    INSERT INTO content (document_id, url, raw_content, metadata, tags, summary, normalized_content, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (document_id, url, raw_content, metadata, tags, summary, normalized_content, datetime.now().isoformat()))

    # Store each chunk in the chunks table
    for chunk, vector in zip(chunks, vectorized_chunks):
        vector_array = np.array(vector)  # Convert list to NumPy array
        cursor.execute("""
           INSERT INTO chunks (document_id, chunk, vectorized_content)
           VALUES (?, ?, ?)
           """, (document_id, chunk, json.dumps(vector_array.tolist())))

    conn.commit()
    conn.close()

    return document_id  # Return the document ID

def search_vectorized_content(query, metadata_similarity_threshold=0.80, vectorized_similarity_threshold=0.80):
    """Search DB for content matching the query using semantic similarity."""
    query_embedding = generate_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return None

    # Ensure the query embedding has the correct dimensions
    query_embedding = np.array(query_embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    print(f"Query Embedding: {query_embedding}")

    # Log the thresholds being used
    print(f"Searching with metadata similarity threshold: {metadata_similarity_threshold} and vectorized similarity threshold: {vectorized_similarity_threshold}")

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
            metadata_embedding = np.array(metadata_embedding)
            metadata_similarity = cosine_similarity(query_embedding, metadata_embedding)

            # Print the metadata similarity
            print(f"Document ID: {document_id}, Metadata Similarity: {metadata_similarity:.2f}")

            # Threshold for metadata relevance (adjust as needed)
            if metadata_similarity > metadata_similarity_threshold:
                # Fetch the vectorized content related to the document_id from the chunks table
                cursor.execute("SELECT chunk, vectorized_content FROM chunks WHERE document_id = ?", (document_id,))
                chunk_results = cursor.fetchall()

                for chunk, vectorized_content_json in chunk_results:
                    try:
                        # Load the vectorized content (stored as JSON)
                        chunk_embedding = np.array(json.loads(vectorized_content_json))

                        # Ensure the chunk embedding has the correct dimensions
                        chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
                        print(f"Chunk Embedding: {chunk_embedding}")

                        if query_embedding.shape == chunk_embedding.shape:
                            similarity = cosine_similarity(query_embedding, chunk_embedding)
                            print(f"Chunk similarity: {float(similarity):.2f}")

                            if float(similarity) >= vectorized_similarity_threshold:
                                if document_id not in relevant_context:
                                    relevant_context[document_id] = f"Tags: {tags}, Summary: {summary}\n"
                                relevant_context[document_id] += f"Similarity: {float(similarity):.2f}, Chunk: {chunk}\n"
                        else:
                            print(f"Dimension mismatch: Query {query_embedding.shape}, Chunk {chunk_embedding.shape}")
                    except Exception as e:
                        print(f"Error processing vectorized content: {e}")
                        continue

        if relevant_context:
            return "\n\n".join(relevant_context.values())
        return None
    finally:
        conn.close()

def update_tags(document_id, tags):
    """
    Update the tags in the database for a specific document ID.

    :param document_id: The ID of the document to update (str).
    :param tags: List of tags to update (list of str).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Convert the list of tags to a comma-separated string
    tags_str = ", ".join(tags)

    cursor.execute("""
        UPDATE content
        SET tags = ?
        WHERE document_id = ?
    """, (tags_str, document_id))

    conn.commit()
    conn.close()