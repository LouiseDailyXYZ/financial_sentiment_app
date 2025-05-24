import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datetime import datetime, timedelta
import yfinance as yf

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

nlp = load_model()

st.set_page_config(page_title="股票新聞情感分析器", layout="centered")
st.title("📈 美股新聞情感分析器")
st.markdown("輸入美股代碼（如 AAPL、TSLA），系統將擷取最近 14 日內的新聞並分析情感傾向。")

# Bing News Search API (免費方式以爬蟲為主)
import feedparser

def search_news(ticker, max_articles=10):
    today = datetime.today()
    after_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+after:{after_date}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    links = []
    for entry in feed.entries:
        st.write('🧾 找到新聞：', entry.title, entry.link)
        links.append(entry.link)
        if len(links) >= max_articles:
            break
    return links

    headers = {'User-Agent': 'Mozilla/5.0'}
    query = f"{ticker} stock site:reuters.com OR site:bloomberg.com OR site:finance.yahoo.com"
    search_url = f'https://www.bing.com/news/search?q={query}&qft=sortbydate="1"&FORM=HDRSC6'
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    results = soup.find_all('a', href=True)
    links = []
    for r in results:
        href = r['href']
        if any(domain in href for domain in ['reuters.com', 'bloomberg.com', 'finance.yahoo.com']):
            if href not in links:
                links.append(href)
        if len(links) >= max_articles:
            break
    return links

def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])
        if len(text) < 200:
            text = soup.get_text()
        return text.strip()
    except Exception as e:
        return None