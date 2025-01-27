from googlesearch import search
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from langchain_core.documents import Document
import re
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, num_results: int = 5):
        self.num_results = num_results
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from document text using TF-IDF-like scoring."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:num_keywords]
        logger.info(f"Extracted keywords: {sorted_keywords}")
        return sorted_keywords

    def search_links(self, query: str) -> List[str]:
        """Search Google for relevant links with error handling and retries."""
        links = []
        try:
            logger.info(f"Searching Google for: {query}")
            for result in search(query, num_results=self.num_results, sleep_interval=2):
                links.append(result)
                if len(links) >= self.num_results:
                    break
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
        return links

    def scrape_page(self, url: str) -> Tuple[str, str]:
        """Scrape webpage content with error handling."""
        try:
            logger.info(f"Scraping page: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            return title, text
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return url, f"Error scraping page: {str(e)}"

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Main processing method with enhanced logging."""
        results = []
        for doc in documents:
            try:
                keywords = self.extract_keywords(doc.page_content)
                search_query = f"{doc.metadata['source']} {' '.join(keywords)}"
                
                # Search for relevant links
                links = self.search_links(search_query)
                logger.info(f"Found {len(links)} links for document: {doc.metadata['source']}")
                
                # Scrape found links
                for link in links:
                    title, content = self.scrape_page(link)
                    results.append(Document(
                        page_content=f"URL: {link}\nTitle: {title}\nContent: {content}",
                        metadata={"source": "web", "url": link}
                    ))
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata['source']}: {str(e)}")
        return results

# Common English stop words
stop_words = set([
    "the", "and", "of", "to", "in", "a", "is", "that", "it", "with", "as", "for", 
    "this", "are", "on", "be", "by", "or", "which", "an", "from", "not", "at", 
    "but", "they", "have", "has", "was", "were", "will", "would", "their"
])