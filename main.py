import os
from dotenv import load_dotenv
from scraper import scrape_url
from openai_utils import process_with_openai, generate_embedding
from database import setup_database, store_in_db, add_normalized_content_column

# Load environment variables
load_dotenv(".env.local")
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    setup_database()
    add_normalized_content_column()  # Add this line
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

    print("\nRaw content extracted:")
    print(raw_content)
    print("\nProcessing content with OpenAI...")
    result = process_with_openai(raw_content)
    print("Raw content:")
    print(result)
    if result:
        normalized_content = result.get("normalized_version", "")
        print("\nNormalized content:")
        print(normalized_content)
        vectorized_content = generate_embedding(normalized_content)

        store_in_db(
            url=url,
            raw_content=raw_content,
            metadata=result.get("metadata"),
            tags=result.get("tags"),
            summary=result.get("summary"),
            normalized_content=normalized_content,
            vectorized_content=vectorized_content
        )
        print("\nContent successfully processed and stored!")
    else:
        print("Failed to process content.")

if __name__ == "__main__":
    main()
