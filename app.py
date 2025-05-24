import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime, timedelta
import yfinance as yf

@st.cache_resource
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
def search_news(ticker, max_articles=10):
    today = datetime.today()
    after_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+after:{after_date}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        # 優先抓取 <article> 或 <div class="main-content">
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            main_div = soup.find('div', class_='main-content')
            paragraphs = main_div.find_all('p') if main_div else soup.find_all('p')

        text = '\n'.join([p.get_text() for p in paragraphs])
        if len(text) < 200:
            text = soup.get_text()
        return text.strip()
    except Exception:
        return None


    links = []
    for entry in feed.entries:
        st.write('🧾 找到新聞：', entry.title, entry.link)
        links.append(entry.link)
        if len(links) >= max_articles:
            break

    return links