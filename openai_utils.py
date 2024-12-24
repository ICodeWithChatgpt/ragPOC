import textwrap
import openai
import json

def process_with_openai(content):
    """Send content to OpenAI API for processing."""
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

        print(content)
        print("Metadata and tags extracted successfully.")
        print("Metadata:", metadata_result.get("metadata"))
        print("Summary:", metadata_result.get("summary"))
        print("Tags:", metadata_result.get("tags"))
        # Split raw content into smaller chunks for normalization
        chunks = textwrap.wrap(content, width=2000)  # Adjust width as needed
        print(chunks)
        normalized_content = []

        print("Starting the content normalization process...")
        for chunk in chunks:
            print(chunk)
            normalization_prompt = f"""
            Normalize the following content into a clean, lowercased version suitable for vectorization.
            Remove unnecessary formatting, special characters, and excessive whitespace.
            Normalize this content fully without skipping sections. Remove formatting and redundant words but retain all key sentences and structure.

            ### Content:
            {chunk}
            """

            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": normalization_prompt}
                ]
            )
            result = response.choices[0].message.content
            normalized_content.append(result)

        # Combine all normalized content chunks
        combined_normalized_content = " ".join(normalized_content)
        print("Content normalized successfully.")
        return {
            "metadata": metadata_result.get("metadata"),
            "tags": metadata_result.get("tags"),
            "summary": metadata_result.get("summary"),
            "normalized_version": combined_normalized_content
        }
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None

# Helper: Query OpenAI API
def query_openai(final_prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying OpenAI: {e}"

 # Helper: Parse OpenAI Response
def parse_openai_response(raw_content):
    try:
        parsed_response = json.loads(raw_content)
        if not all(k in parsed_response for k in ["metadata", "tags", "summary", "normalized_version"]):
            print("Missing fields in OpenAI response.")
            return None
        return parsed_response
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return None

# Helper: Generate Embedding
def generate_embedding(text):
    try:
        # Log the content being vectorized
        print(f"Vectorizing content: {text[:200]}...")  # Log the first X characters of the content

        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None