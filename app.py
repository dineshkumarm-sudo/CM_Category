import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(
    page_title="Product Category Accuracy Auditor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    .main-header { color: #1E3A8A; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #64748B; font-size: 1rem; margin-bottom: 2rem; }
    .css-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    [data-testid="stMetricValue"] { font-weight: 700; color: #1E3A8A; }
    .stButton>button {
        background-color: #EA580C !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Taxonomy Master
@st.cache_data
def load_taxonomy():
    if os.path.exists("taxonomy.csv"):
        df = pd.read_csv("taxonomy.csv")
    elif os.path.exists("CM_Path 1.xlsx"):
        df = pd.read_excel("CM_Path 1.xlsx")
    else:
        st.error("Taxonomy master file (taxonomy.csv or CM_Path 1.xlsx) not found!")
        st.stop()
    
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['path', 'Fullpath']).copy()
    df['path'] = df['path'].astype(str)
    df['Fullpath'] = df['Fullpath'].astype(str)
    return df

tax_df = load_taxonomy()

# Pre-build Search Index
@st.cache_resource
def build_search_index(taxonomy_df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(taxonomy_df['Fullpath'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_search_index(tax_df)

def get_candidate_categories(product_title, team_fullpath, top_n=15):
    query = f"{product_title} {team_fullpath}"
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    candidates = tax_df.iloc[top_indices][['path', 'Fullpath']].to_dict(orient='records')
    team_matches = tax_df[tax_df['Fullpath'] == team_fullpath].to_dict(orient='records')
    if team_matches and team_matches[0] not in candidates:
        candidates.insert(0, team_matches[0])
        
    return candidates

# UI Header
st.markdown('<h1 class="main-header">📦 Product Categorization Accuracy Auditor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated e-commerce taxonomy audit powered by AI</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
if not api_key:
    api_key = st.sidebar.text_input("API Key (OpenAI or NVIDIA)", type="password")

st.sidebar.subheader("Taxonomy Master")
st.sidebar.info(f"Loaded **{len(tax_df):,}** categories.")

# Function to wipe state on new upload
def reset_session_
