from flask import Flask, request, render_template, jsonify
import openai
import os
from dotenv import load_dotenv
from API.openai_utils import query_openai, process_with_openai, generate_embedding
from scraper import scrape_url
import database.database as db
import docx2txt


# Load environment variables
load_dotenv(".env.local")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
DB_NAME = "content_store.db"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/fetch-content", methods=["POST"])
def fetch_content():
    if 'file-input' in request.files:
        file = request.files['file-input']
        if file.filename.endswith(('.doc', '.docx')):
            raw_content = docx2txt.process(file)
        else:
            return jsonify({"error": "Unsupported file type."}), 400
    else:
        input_data = request.form.get("input_data", "").strip()
        if input_data.startswith("http"):
            raw_content = scrape_url(input_data)
            if not raw_content:
                return jsonify({"error": "Failed to scrape URL."}), 400
        else:
            raw_content = input_data

    return jsonify({"raw_content": raw_content})

# Route: Process Content
@app.route("/process-content", methods=["POST"])
def process_content():
    print("Processing content......................")
    data = request.get_json()
    edited_content = data.get("edited_content", "").strip()
    metadata_similarity = data.get("metadata_similarity", 0.8)
    vectorized_similarity = data.get("vectorized_similarity", 0.8)
    chunk_size = int(data.get("chunk_size", 250))

    result = process_with_openai(edited_content, metadata_similarity, vectorized_similarity, chunk_size)
    if result:
        normalized_content = result.get("normalized_version", "")
        chunks = result.get("chunks", [])
        vectorized_chunks = result.get("vectorized_chunks", [])

        document_id = db.store_in_db(
            url=None,
            raw_content=edited_content,
            metadata=result.get("metadata"),
            tags=result.get("tags"),
            summary=result.get("summary"),
            normalized_content=normalized_content,
            chunks=chunks,
            vectorized_chunks=vectorized_chunks
        )
        result["document_id"] = document_id
        return jsonify(result)
    else:
        return jsonify({"error": "Failed to process content."}), 500

@app.route("/update-tags", methods=["POST"])
def update_tags():
    data = request.get_json()
    document_id = data.get("document_id")
    tags = data.get("tags", [])
    print(f"Updating tags for document ID {document_id}: {tags}")  # Debug log
    db.update_tags(document_id, tags)
    return jsonify({"status": "success"})

@app.route("/prompt", methods=["POST"])
def submit_propmt():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    search_db_first = data.get("searchDB", False)
    metadata_similarity = data.get("metadata_similarity", 0.8)
    vectorized_similarity = data.get("vectorized_similarity", 0.8)
    print("Metadata Similarity:", metadata_similarity)
    print("Vectorized Similarity:", vectorized_similarity)

    initial_prompt = prompt
    final_prompt = prompt
    response_text = "No response generated."

    if search_db_first:
        relevant_context = db.search_vectorized_content(prompt, metadata_similarity, vectorized_similarity)
        if relevant_context:
            #EDIT THIS PROMPT TO INFLUENCE HOW WE WANT THE RESPONSE --------------------
            #FINAL PROMPT EASY1: Use the following retrieved context to answer the user's query.
            # Maintain a serious and professional tone in your responses.
            #        Limit yourself to maintain the humorous tone and style of the provided context.




            final_prompt = f"""
        You are a helpful assistant providing responses to user queries.
        Use the following retrieved context to answer the user's query.
        If the query requires creativity or ideation, use the context as inspiration to generate innovative suggestions.
        Ensure that the response is relevant, engaging, and informative.
        

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