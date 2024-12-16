import json
import re
from os.path import exists

import openai
import requests
from bs4 import BeautifulSoup
import sqlite3
import hashlib
import os
from datetime import datetime
from dotenv import load_dotenv


load_dotenv(".env.local")

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your environment variables

# SQLite Database Setup
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
        vectorized_content TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def scrape_url(url):
    """Scrape a webpage and return its raw text content."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Error scraping URL: {e}")
        return None


def process_with_openai(content, url=None):
    """Send content to OpenAI for processing."""
    try:
        prompt = f"""
        Analyze the following content and return the following:
        1. Metadata (e.g., title, author if available, source type).
        2. Tags or labels to categorize the content.
        3. A concise summary of the content.
        4. A normalized version ready for vectorization.
        
        The format of the response must always be a valid JSON because I am going to parse it into an object with the following structure:
            "metadata": "Example Metadata",
            "tags": "Example Tags",
            "summary": "Example Summary",
            "normalized_version": "Normalized for Vectorization"

        Content: {content}
        """
        # Make the OpenAI API request
        response = openai.chat.completions.create(
            model="gpt-4",  # Use the correct model
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract the response content
        result = response.choices[0].message.content
        result_json = parse_openai_response(result)
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None
    return result_json


def parse_openai_response(raw_content):
    parsed_response = json.loads(raw_content)
    # Check if the fields exists( metadata, tags, summary, normalized_version)
    if not all(field in parsed_response for field in ["metadata", "tags", "summary", "normalized_version"]):
        print("Missing fields in the response.")
        return None
    return parsed_response


def store_in_db(url, raw_content, metadata, tags, summary, vectorized_content):
    """Store processed content in SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Generate a unique ID based on content
    content_id = hashlib.sha256((url or raw_content).encode()).hexdigest()

    cursor.execute("""
    INSERT OR REPLACE INTO content (id, url, raw_content, metadata, tags, summary, vectorized_content, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (content_id, url, raw_content, metadata, tags, summary, vectorized_content, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def main():
    setup_database()

    # Input: Either a URL or raw content
    print("Enter a URL or paste raw content below:")
    input_data = input("> ").strip()

    url = None
    if input_data.startswith("http"):
        url = input_data
        raw_content = scrape_url(url)
        if not raw_content:
            print("Failed to scrape URL.")
            return
    else:
        raw_content = input_data

    print("\nProcessing content with OpenAI...")
    result = process_with_openai(raw_content, url=url)
    print("\nRAW RESULT:")
    print(result)

    if result:
        # Simulate extracting the result into meaningful pieces
        print("\n[OpenAI Result]")
        print(result)

        # Simulating parsing of OpenAI response
        metadata = "Example Metadata"
        tags = "Example Tags"
        summary = "Example Summary"
        vectorized_content = "Normalized for Vectorization"

        # Store in database
        store_in_db(url, raw_content, metadata, tags, summary, vectorized_content)
        print("\nContent successfully processed and stored!")
    else:
        print("Failed to process content with OpenAI.")


if __name__ == "__main__":
    main()