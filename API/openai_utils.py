import textwrap
import openai
import json

from helpers.helper_functions import pad_or_truncate_embedding


def process_with_openai(content, metadata_similarity=0.8, vectorized_similarity=0.8, chunk_size=250):
    """Send content to OpenAI API for processing. Only during the content processing phase."""
    try:
        # Extract metadata and tags from the entire content first
        instructions = f"""
        You are tasked with analyzing the following content and returning a JSON object with the required fields.
        The response **must** strictly follow the JSON structure provided below.
        The keys must appear exactly as shown, and no additional keys or extra text should be included.

        The JSON structure is as follows:
        {{
            "metadata": "Title: Example Title, Author: Example Author Name, SourceType: Example SourceType",
            "tags": "Example Tags",
            "summary": "Example Summary"
        }}

        ### Instructions:
        1. **metadata**: Provide relevant metadata about the content, such as title, author, and source type.
            - If no title or heading exists, create one based on the content itself.
            - Avoid generic phrases like 'Content from a website' or 'Scraped from [source]'.
            - Include the author's name if available.
            - Specify the source type, e.g., 'Blog', 'News Article', 'Research Paper'.

        2. **tags**: Provide comma-separated tags to categorize the content. Use concise, relevant terms.
        3. **summary**: Write a brief and coherent summary of the content in 1 or 2 sentences. Always in 3rd person and present tense. Just a clinical description.
        """

        metadata_prompt = f"""
        ### Content:
        {content[:2000]} #Limit the content for the first 2000 characters
        """

        metadata_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": metadata_prompt}
            ]
        )
        metadata_result = json.loads(metadata_response.choices[0].message.content)

        print("Metadata and tags extracted successfully.")
        print("Metadata:", metadata_result.get("metadata"))
        print("Summary:", metadata_result.get("summary"))
        print("Tags:", metadata_result.get("tags"))

        normalization_prompt = f"""
        Normalize the following content into a clean, lowercased version suitable for vectorization.
        Remove unnecessary formatting, special characters, and excessive whitespace.
        Normalize this content fully without skipping sections. Remove formatting and redundant words but retain all key sentences and structure.

        ### Content:
        {content}
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": normalization_prompt}
            ]
        )
        normalized_content = response.choices[0].message.content


        # Split normalized content into chunks of specified size
        tokens = normalized_content.split()
        chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        print(chunks)
        print(chunk_size)

        # Vectorize each chunk separately and ensure the correct dimensions
        vectorized_chunks = [pad_or_truncate_embedding(generate_embedding(chunk), 1536) for chunk in chunks]

        return {
            "metadata": metadata_result.get("metadata"),
            "tags": metadata_result.get("tags"),
            "summary": metadata_result.get("summary"),
            "normalized_version": normalized_content,
            "chunks": chunks,
            "vectorized_chunks": vectorized_chunks,
            "metadata_similarity": metadata_similarity,
            "vectorized_similarity": vectorized_similarity
        }
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None

def query_openai(final_prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying OpenAI: {e}"

def generate_embedding(text):
    try:
        print(f"Vectorizing content: {text[:200]}...")  # Log the first X characters of the content

        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None