"""
Filename: extractor.py
Author: Jayendra Matarage
Created on: 9/20/2026 10:09 AM
Description: 
"""
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

punkt_params = PunktParameters()
punkt_params.abbrev_types = {
    "rs",
    "mr",
    "mrs",
    "dr",
    "prof",
    "st",
    "vs",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
}
custom_sent_tokenizer = PunktSentenceTokenizer(punkt_params)

def get_web_site_content(url):
    print("Loading web page")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        html_content = driver.page_source
    finally:
        driver.quit()
    print("Web page content loaded!")
    return BeautifulSoup(html_content, "html.parser")

def get_article_urls(url,limit=50):
    soup = get_web_site_content(url)
    urls = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/article/" in href or "/news/" in href or "/breaking-news/" in href:
            full_url = href if href.startswith("http") else f"https:{href}"
            if full_url not in urls:
                urls.append(full_url)
                print(f"URL found {full_url}")
            if len(urls) >= limit:
                break

    return urls


def extract_sentences_from_url(url):
    soup = get_web_site_content(url)

    article_body = soup.find("div", class_="a-content") or soup.find("div", class_="mmfpmf")

    if not article_body:
        article_body = soup

    paragraphs = article_body.find_all("p")
    raw_text = " ".join([p.get_text().replace("\xa0", " ").strip() for p in paragraphs])

    sentences = custom_sent_tokenizer.tokenize(raw_text)

    clean_sents = [
        s.strip()
        for s in sentences
        if 5 <= len(s.split()) <= 35
    ]
    return clean_sents


def build_dataset(extracted_urls, target_count=150):
    print("Building dataset..")
    collected_sentences = []

    for url in extracted_urls:
        try:
            sents = extract_sentences_from_url(url)
            print(f"extracting sentences.. {len(sents)}")
            for s in sents:
                if s not in collected_sentences:
                    collected_sentences.append(s)
                if len(collected_sentences) >= target_count:
                    break
        except Exception as e:
            continue

        if len(collected_sentences) >= target_count:
            break

    with open(
        "daily_mirror_data.raw", "w", encoding="utf-8"
    ) as f:
        for line in collected_sentences[:target_count]:
            f.write(f"{line}\n")

    print(
        f"Successfully saved {len(collected_sentences[:target_count])} sentences to daily_mirror.raw"
    )


def main():
    print("Running data extractor")
    target_url = "https://www.dailymirror.lk/top-story/155"
    extracted_urls =  get_article_urls(target_url)
    print(f"Extracted url count {len(extracted_urls)}")
    build_dataset(extracted_urls)

if __name__ == "__main__":
    main()