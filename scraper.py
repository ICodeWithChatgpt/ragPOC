import requests
from bs4 import BeautifulSoup

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
