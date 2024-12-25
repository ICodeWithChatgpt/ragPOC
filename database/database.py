import sqlite3
import hashlib
import json
import textwrap
from datetime import datetime

from API.openai_utils import generate_embedding
from helpers.helper_functions import cosine_similarity
import nltk

DB_NAME = "content_store.db"

def setup_database():
    """Create the database and necessary table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS content (
        id TEXT PRIMARY KEY,
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

def store_in_db(url, raw_content, metadata, tags, summary, normalized_content, vectorized_content):
    """Store processed content in SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Generate a unique ID based on content
    content_id = hashlib.sha256((url or raw_content).encode()).hexdigest()

    # Serialize vectorized content (embedding) as JSON
    vectorized_content_json = json.dumps(vectorized_content or [])

    cursor.execute("""
    INSERT OR REPLACE INTO content (id, url, raw_content, metadata, tags, summary, normalized_content, vectorized_content, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (content_id, url, raw_content, metadata, tags, summary, normalized_content, vectorized_content_json, datetime.now().isoformat()))

    conn.commit()
    conn.close()

# Ensure you have the necessary NLTK data
nltk.download('punkt')

# Function: Search Vectorized Content
def search_vectorized_content(query, metadata_similarity_threshold=0.80, vectorized_similarity_threshold=0.80):
    """Search DB for content matching the query using semantic similarity."""
    query_embedding = generate_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return None

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Fetch all metadata and vectorized content from the database
    cursor.execute("SELECT id, tags, summary, normalized_content, vectorized_content FROM content")
    results = cursor.fetchall()
    conn.close()

    relevant_context = []
    for row in results:
        doc_id, tags, summary, normalized_content, vectorized_content_json = row
        metadata = f"{tags} {summary}"
        metadata_embedding = generate_embedding(metadata)
        metadata_similarity = cosine_similarity(query_embedding, metadata_embedding)

        # Print the metadata similarity
        print(f"Document ID: {doc_id}, Metadata Similarity: {metadata_similarity:.2f}")

        # Threshold for metadata relevance (adjust as needed)
        if metadata_similarity > metadata_similarity_threshold:  # Adjust threshold as needed
            try:
                # Load the vectorized content (stored as JSON)
                vectorized_content = json.loads(vectorized_content_json)
                print(f"Vectorized content loaded: {len(vectorized_content)} vectors")

                # Split normalized content into smaller chunks
                chunks = textwrap.wrap(normalized_content, width=250)  # Adjust width as needed
                candidate_sentences = []
                for chunk in chunks:
                    chunk_embedding = generate_embedding(chunk)
                    similarity = cosine_similarity(query_embedding, chunk_embedding)

                    # Threshold for relevance (adjust as needed)
                    if similarity > vectorized_similarity_threshold:
                        candidate_sentences.append(f"Similarity: {similarity:.2f}, Sentence: {chunk}")

                if candidate_sentences:
                    relevant_context.append(
                        f"Tags: {tags}, Summary: {summary}\n" + "\n".join(candidate_sentences))
            except Exception as e:
                print(f"Error processing vectorized content: {e}")
                continue

    if relevant_context:
        return "\n\n".join(relevant_context)
    return None

