from flask import Flask, request, render_template, jsonify
import sqlite3
import numpy as np
import openai
import os
import json
from dotenv import load_dotenv
import textwrap
from openai_utils import generate_embedding

# Load environment variables
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
DB_NAME = "content_store.db"

# Helper: Compute Cosine Similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function: Search Vectorized Content
def search_vectorized_content(query):
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
        if metadata_similarity > 0.75:  # Adjust threshold as needed
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
                    if similarity > 0.75:
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
# Helper: Query OpenAI API
def query_openai(final_prompt):
    """Query OpenAI LLM."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying OpenAI: {e}"


# Route: Show the Prompt Page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Route: Handle Prompt Submission
@app.route("/prompt", methods=["POST"])
@app.route("/prompt", methods=["POST"])
def handle_prompt():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    search_db_first = data.get("searchDB", False)

    initial_prompt = prompt
    final_prompt = prompt
    response_text = "No response generated."

    if search_db_first:
        relevant_context = search_vectorized_content(prompt)
        if relevant_context:
            final_prompt = f"""
        Use the following retrieved context to answer the user's query.

        ### Retrieved Context:

        {relevant_context}

        ### User Query:

        {prompt}
        """
        else:
            final_prompt = f"No relevant content found. Proceeding with user query:\n{prompt}"

    response_text = query_openai(final_prompt)

    return jsonify({
        "initial_prompt": initial_prompt,
        "final_prompt": final_prompt,
        "response": response_text
    })


if __name__ == "__main__":
    app.run(debug=True)
