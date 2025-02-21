import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizer
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
from urllib.parse import urljoin, urlparse

# Load BERT model and tokenizer for NER
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = TFBertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Seed URLs for testing
seed_urls = [
    "https://en.wikipedia.org/wiki/Bill_Gates",
]

visited_urls = set()  # To track visited URLs
max_depth = 2  # Maximum depth for crawling

def extract_names_with_ner(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs).logits
    predictions = tf.argmax(outputs, axis=2)

    # Convert token predictions to readable names
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())
    predicted_entities = [tokens[i] for i in range(len(predictions[0])) if predictions[0][i] != 0]  # Skip non-entity labels
    return predicted_entities

def clean_name(name):
    return re.sub(r'\s+', ' ', name.strip()).replace("\n", " ").replace("\r", "").replace("\t", " ")

def extract_links(soup, base_url):
    links = set()
    for tag in soup.find_all("a", href=True):
        link = tag['href']
        # Ensure the link is an internal link and is valid
        absolute_url = urljoin(base_url, link)
        if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
            links.add(absolute_url)
    return links

def scrape_person_info(url):
    response = requests.get(url)
    if response.status_code != 200:
        return [], []

    soup = BeautifulSoup(response.content, "html.parser")
    page_text = soup.get_text()
    ner_names = extract_names_with_ner(page_text)

    cleaned_names = [clean_name(name) for name in ner_names]

    name_counts = Counter(cleaned_names)
    final_names = [name for name in cleaned_names if name_counts[name] < 3]

    links = extract_links(soup, url)  # Extract internal links for further crawling

    print(f"Extracted names from {url}: {final_names}")
    return final_names, links

def crawl(urls, depth=2):
    to_crawl = set(urls)
    crawled = set()
    all_names = []
    current_depth = 0

    while to_crawl and current_depth < depth:
        current_level_urls = list(to_crawl)
        to_crawl.clear()  # Clear the set for the next depth level

        # Using multiprocessing to scrape URLs in parallel for the current depth level
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(scrape_person_info, current_level_urls)

        # Flatten the results and update the URLs for further crawling
        for names, links in results:
            all_names.extend(names)
            to_crawl.update(links)  # Add the new links to crawl further

        # Increase the depth after finishing the current level
        current_depth += 1
        print(f"Depth: {current_depth}, Remaining URLs to crawl: {len(to_crawl)}")

    return all_names


if __name__ == "__main__":
    result = crawl(seed_urls)
    print(f"Total extracted names: {result}")
