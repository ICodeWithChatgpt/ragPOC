from flask import Flask, request, render_template, jsonify
import openai
import os
from dotenv import load_dotenv

from API.openai_utils import query_openai, process_with_openai, generate_embedding
from scraper import scrape_url
import database.database as db

# Load environment variables
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
DB_NAME = "content_store.db"

# Route: Show the Prompt Page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route: Fetch Content
@app.route("/fetch-content", methods=["POST"])
def fetch_content():
    print("STEP 1: Fetching content......................")
    data = request.get_json()
    input_data = data.get("input_data", "").strip()

    url = None
    if input_data.startswith("http"):
        url = input_data
        raw_content = scrape_url(url)
        if not raw_content:
            return jsonify({"error": "Failed to scrape URL."}), 400
    else:
        raw_content = input_data

    return jsonify({"raw_content": raw_content})

# Route: Process Content
@app.route("/process-content", methods=["POST"])
def process_content():
    print("STEP 2: Processing content......................")
    data = request.get_json()
    edited_content = data.get("edited_content", "").strip()

    result = process_with_openai(edited_content)
    if result:
        normalized_content = result.get("normalized_version", "")
        vectorized_content = generate_embedding(normalized_content)


        db.store_in_db(
            url=None,
            raw_content=edited_content,
            metadata=result.get("metadata"),
            tags=result.get("tags"),
            summary=result.get("summary"),
            normalized_content=normalized_content,
            vectorized_content=vectorized_content
        )
        return jsonify(result)
    else:
        return jsonify({"error": "Failed to process content."}), 500

# Route: Handle Prompt Submission
@app.route("/prompt", methods=["POST"])
def handle_prompt():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    search_db_first = data.get("searchDB", False)

    initial_prompt = prompt
    final_prompt = prompt
    response_text = "No response generated."

    if search_db_first:
        relevant_context = db.search_vectorized_content(prompt)
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
    db.setup_database()
    app.run(debug=True)