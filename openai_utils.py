import openai
import json

def process_with_openai(content):
    """Send content to OpenAI API for processing."""
    try:
        prompt = f"""
           You are tasked with analyzing the following content and returning a JSON object with the required fields. 
           The response **must** strictly follow the JSON structure provided below. 
           The keys must appear exactly as shown, and no additional keys or extra text should be included.

           The JSON structure is as follows:
           {{
               "metadata": "Title: Example Title, Author: Example Author Name, SourceType: Example SourceType",
               "tags": "Example Tags",
               "summary": "Example Summary",
               "normalized_version": "Normalized version of the content suitable for vectorization"
           }}

           ### Instructions:
           1. **metadata**: Provide relevant metadata about the content, such as title, author, and source type. 
               - If no title or heading exists, create one based on the content itself.
               - Avoid generic phrases like 'Content from a website' or 'Scraped from [source]'.
               - Include the author's name if available.
               - Specify the source type, e.g., 'Blog', 'News Article', 'Research Paper'.

           2. **tags**: Provide comma-separated tags to categorize the content. Use concise, relevant terms.
           3. **summary**: Write a brief and coherent summary of the content.
           4. **normalized_version**: Normalize the content into a clean, lowercased version suitable for vectorization. Remove unnecessary formatting, special characters, and excessive whitespace.

           ### Rules:
           - The response must be valid JSON. Do not include any other text outside the JSON.
           - Avoid generic metadata or tags like 'website content' or 'scraped text'. Be specific.
           - Ensure all fields are filled out properly.

           ### Example Response:
           {{
               "metadata": "Title: The history of artificial intelligence in healthcare, SourceType: Blog, URL: https://example.com/article",
               "tags": "healthcare, artificial intelligence, technology, history",
               "summary": "This article discusses the evolution of AI technologies in the healthcare sector, including their applications and impact on patient care.",
               "normalized_version": "the history of artificial intelligence in healthcare discusses the evolution of ai technologies in the healthcare sector including their applications and impact on patient care"
           }}

           ### Content:
           {content}
           """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        return parse_openai_response(result)
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None

def parse_openai_response(raw_content):
    """Parse OpenAI response and validate fields."""
    try:
        parsed_response = json.loads(raw_content)
        if not all(k in parsed_response for k in ["metadata", "tags", "summary", "normalized_version"]):
            print("Missing fields in OpenAI response.")
            return None
        return parsed_response
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return None

def generate_embedding(text):
    """Generate an embedding for the given text using OpenAI's embedding model."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
