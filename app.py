import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from datetime import datetime, timedelta
import yfinance as yf
import feedparser

st.set_page_config(page_title="股票新聞情感分析器", layout="centered")

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

nlp = load_model()

def search_news(ticker, max_articles=10):
    today = datetime.today()
    after_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+after:{after_date}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    links = []
    for entry in feed.entries:
        st.write('📄 找到新聞：', entry.title)
        links.append(entry.link)
        if len(links) >= max_articles:
            break

    return links

def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
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

st.title("📈 美股新聞情感分析器")
st.markdown("輸入美股代碼（如 AAPL、TSLA），系統將擷取最近 14 日內的新聞並分析情感傾向。")

ticker = st.text_input("請輸入美股代碼 (如 AAPL、TSLA)").upper()

if ticker:
    st.info("📡 正在搜尋相關新聞...")
    news_links = search_news(ticker)

    sentiments = []
    summaries = []

    for link in news_links:
        text = extract_article_text(link)
        if text:
            short_text = text[:512]
            result = nlp(short_text)
            sentiments.append(result[0]['score'] * (1 if result[0]['label'] == 'positive' else -1 if result[0]['label'] == 'negative' else 0))
            summaries.append((link, result[0]['label'], round(result[0]['score'] * 100, 2)))

    if summaries:
        st.subheader("📰 最新新聞情緒分析結果：")
        for i, (link, label, score) in enumerate(summaries, 1):
            st.markdown(f"**{i}. [{label}] ({score}%)** ➜ [查看新聞]({link})")

        avg_sentiment = sum(sentiments) / len(sentiments)
        st.subheader("📊 整體平均情緒分數：")
        if avg_sentiment > 0.1:
            st.success(f"偏正面：{round(avg_sentiment, 2)}")
        elif avg_sentiment < -0.1:
            st.error(f"偏負面：{round(avg_sentiment, 2)}")
        else:
            st.warning(f"情緒中立：{round(avg_sentiment, 2)}")
    else:
        st.warning("未擷取到有效新聞或情感無法分析。")