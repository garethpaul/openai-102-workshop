import requests
from bs4 import BeautifulSoup


def get_text(url):
    # crawl the url with requests and get the text back
    # Crawl the URL with requests and get the text back
    # create fake browser for chrome to send the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    html_content = response.text

    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the text from the parsed HTML
    text = soup.get_text()
    return text
