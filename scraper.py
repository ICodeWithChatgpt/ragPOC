import requests
from bs4 import BeautifulSoup
import re


def clean_text(text):
    """Clean and format extracted text."""
    # Remove extra whitespace, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = text.strip()  # Strip leading/trailing whitespace
    return text


def scrape_url(url):
    """Scrape a webpage and return cleaner text content."""
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements like <script>, <style>, and comments
        for tag in soup(["script", "style"]):
            tag.decompose()  # Remove the tag entirely

        # Extract text content from key tags only
        cleaned_content = []
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div']):
            text = element.get_text(separator=' ', strip=True)  # Extract and clean text
            if text:  # Ignore empty text
                cleaned_content.append(text)

        # Combine the cleaned content with proper spacing
        raw_cleaned_text = "\n".join(cleaned_content)
        return clean_text(raw_cleaned_text)

    except requests.exceptions.RequestException as e:
        print(f"Error scraping URL: {e}")
        return None
