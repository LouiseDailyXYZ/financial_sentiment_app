import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from newspaper import Article

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

nlp = load_model()

st.set_page_config(page_title="金融情感分析器", layout="centered")
st.title("📈 金融文章情感分析器")
st.markdown("輸入一篇金融新聞連結，自動分析其情緒是正面、負面或中立。")

url = st.text_input("請輸入文章連結（例如 Bloomberg、Reuters、Yahoo Finance）")

if url:
    try:
        with st.spinner("⏳ 正在擷取文章內容..."):
            article = Article(url)
            article.download()
            article.parse()
            text = article.text

        if not text:
            st.error("❌ 無法擷取內容，請確認連結是否正確")
        else:
            st.subheader("📄 文章內容")
            st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

            st.subheader("📊 情感分析結果")
            result = nlp(text[:512])
            label = result[0]['label']
            score = round(result[0]['score'] * 100, 2)
            st.success(f"**{label}**（信心值：{score}%）")

    except Exception as e:
        st.error(f"🚨 發生錯誤：{e}")
