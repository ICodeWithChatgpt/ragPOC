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
           3. **summary**: Write a brief and coherent summary of the content in 1 or 2 sentences. Always in 3rd person and present tense. Just a clinical description.
           4. **normalized_version**: Normalize ONLY the relevant content of the site, into a clean, lowercased version suitable for vectorization. 
           Remove unnecessary formatting, special characters, and excessive whitespace.
           Normalize this content fully without skipping sections. Remove formatting and redundant words but retain all key sentences and structure.


           ### Rules:
           - The response must be valid JSON. Do not include any other text outside the JSON.
           - Avoid generic metadata or tags like 'website content' or 'scraped text'. Be specific.
           - Ensure all fields are filled out properly.

           ### Example Response:
           {{
               "metadata": "Title: The history of artificial intelligence in healthcare, SourceType: Blog",
               "tags": "healthcare, artificial intelligence, technology, history",
               "summary": "This article discusses the evolution of AI technologies in the healthcare sector, including their applications and impact on patient care.",
               "raw_content": "The history of artificial intelligence in healthcare... is filled with discussions of the evolution of AI technologies in the healthcare sector, including their applications and impact on patient care.",
               "normalized_version": "the history of artificial intelligence in healthcare is filled with discusses of the evolution of ai technologies in the healthcare sector including their applications and impact on patient care"
           }}

           ### Content:
           {content}
           """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        print("Raw Response from OpenAI:")
        print(response)
        print("--------------------")
        result = response.choices[0].message.content
        print("Result after choosing only the content:")
        print(result)
        print("--------------------")
        final = parse_openai_response(result)
        print("Final Result after parsing:")
        print(final)
        return final
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
