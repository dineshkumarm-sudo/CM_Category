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

# Custom CSS for UI Aesthetics
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Header Styling */
    .main-header {
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Container */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #1E3A8A;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #EA580C !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #C2410C !important;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.3);
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
        st.error("Taxonomy master file (taxonomy.csv or CM_Path 1.xlsx) not found in directory!")
        st.stop()
    
    # Clean column names and drop missing values
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['path', 'Fullpath']).copy()
    df['path'] = df['path'].astype(str)
    df['Fullpath'] = df['Fullpath'].astype(str)
    return df

tax_df = load_taxonomy()

# Pre-build Search Index for Large Taxonomy
@st.cache_resource
def build_search_index(taxonomy_df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(taxonomy_df['Fullpath'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_search_index(tax_df)

def get_candidate_categories(product_title, team_fullpath, top_n=15):
    """Retrieve top candidates from taxonomy to stay well within API token limits."""
    query = f"{product_title} {team_fullpath}"
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    candidates = tax_df.iloc[top_indices][['path', 'Fullpath']].to_dict(orient='records')
    
    # Ensure team's mapped category is included in candidate options if it exists
    team_matches = tax_df[tax_df['Fullpath'] == team_fullpath].to_dict(orient='records')
    if team_matches and team_matches[0] not in candidates:
        candidates.insert(0, team_matches[0])
        
    return candidates

# UI Header
st.markdown('<h1 class="main-header">📦 Product Categorization Accuracy Auditor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated e-commerce taxonomy audit powered by AI</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Retrieve API Key from Streamlit Secrets or Sidebar Input
api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

if not api_key:
    api_key = st.sidebar.text_input("API Key (OpenAI or NVIDIA)", type="password", help="Enter key starting with nvapi- or sk-")

st.sidebar.subheader("Taxonomy Master")
st.sidebar.info(f"Loaded **{len(tax_df):,}** categories from master file.")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Product File to Audit (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.read_excel(uploaded_file)

    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("### 1. Select Column Mappings")
    c1, c2 = st.columns(2)
    with c1:
        title_col = st.selectbox("Select Product Title Column", input_df.columns)
    with c2:
        team_path_col = st.selectbox("Select Team Mapped Path Column", input_df.columns)
    st.markdown('</div>', unsafe_allow_html=True)

    # Path to Fullpath Lookup dictionary
    path_to_fullpath = dict(zip(tax_df['path'], tax_df['Fullpath']))
    
    # Auto-map Team Full Path
    input_df['Team Full Path'] = input_df[team_path_col].astype(str).map(path_to_fullpath).fillna("UNMAPPED / INVALID PATH")

    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.write("### 2. Input Data Preview")
    st.dataframe(input_df[[title_col, team_path_col, 'Team Full Path']].head(5), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Categorization Audit"):
        if not api_key:
            st.error("API Key missing! Please configure Streamlit secrets or supply an API key in the sidebar.")
        else:
            api_key = api_key.strip()
            
            # Detect whether key is NVIDIA or OpenAI and set up client + model
            if api_key.startswith("nvapi-"):
                client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key
                )
                model_name = "meta/llama-3.3-70b-instruct"
            else:
                client = OpenAI(api_key=api_key)
                model_name = "gpt-4o-mini"

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_items = len(input_df)

            for idx, row in input_df.iterrows():
                title = str(row[title_col])
                team_path = str(row[team_path_col])
                team_fullpath = str(row['Team Full Path'])

                candidates = get_candidate_categories(title, team_fullpath, top_n=15)

                prompt = f"""
                You are an e-commerce taxonomy expert. Evaluate if the team mapped category is accurate for the product.

                Product Title: "{title}"
                Team Mapped Path Code: "{team_path}"
                Team Mapped Full Path: "{team_fullpath}"

                Candidate Approved Categories:
                {json.dumps(candidates, indent=2)}

                Instructions:
                1. Identify the exact product type from the title.
                2. Check if Team Mapped Full Path accurately describes the product.
                3. Select the SINGLE BEST fitting category from the candidate list provided.
                
                You MUST return ONLY a raw JSON object with these exact keys:
                {{
                    "product_type": "Brief extracted product type",
                    "is_match": true or false,
                    "suggested_path": "exact short path code from candidate list",
                    "suggested_fullpath": "exact Fullpath string from candidate list",
                    "reasoning": "Brief 1-sentence explanation"
                }}
                """

                try:
                    kwargs = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0
                    }
                    
                    if not api_key.startswith("nvapi-"):
                        kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**kwargs)
                    raw_content = response.choices[0].message.content.strip()

                    # Clean markdown formatting if returned
                    if raw_content.startswith("```json"):
                        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                    elif raw_content.startswith("```"):
                        raw_content = raw_content.replace("```", "").strip()

                    res = json.loads(raw_content)

                    results.append({
                        "Product Title": title,
                        "Team Path": team_path,
                        "Team Full Path": team_fullpath,
                        "Identified Product Type": res.get("product_type", ""),
                        "Match Status": "MATCH" if res.get("is_match") else "MISMATCH",
                        "AI Suggested Path": res.get("suggested_path", ""),
                        "AI Suggested Full Path": res.get("suggested_fullpath", ""),
                        "AI Reasoning": res.get("reasoning", "")
                    })
                except Exception as e:
                    results.append({
                        "Product Title": title,
                        "Team Path": team_path,
                        "Team Full Path": team_fullpath,
                        "Identified Product Type": "Error",
                        "Match Status": "ERROR",
                        "AI Suggested Path": "",
                        "AI Suggested Full Path": "",
                        "AI Reasoning": str(e)
                    })

                progress_bar.progress((idx + 1) / total_items)
                status_text.text(f"Auditing item {idx + 1} of {total_items}...")

            status_text.empty()
            progress_bar.empty()

            audit_df = pd.DataFrame(results)
            st.session_state['audit_results'] = audit_df

    # Display Results Dashboard
    if 'audit_results' in st.session_state:
        audit_df = st.session_state['audit_results']

        st.markdown('### 3. Audit Dashboard')
        
        total = len(audit_df)
        matches = len(audit_df[audit_df["Match Status"] == "MATCH"])
        mismatches = len(audit_df[audit_df["Match Status"] == "MISMATCH"])
        accuracy = (matches / total * 100) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Items Audited", total)
        m2.metric("Total Matches", matches)
        m3.metric("Mismatches Found", mismatches, delta=f"{mismatches} issues", delta_color="inverse")
        m4.metric("Mapping Accuracy", f"{accuracy:.1f}%")

        filter_option = st.radio(
            "Filter Results Table:",
            ["Show All", "Show Mismatches Only", "Show Matches Only"],
            horizontal=True
        )

        display_df = audit_df.copy()
        if filter_option == "Show Mismatches Only":
            display_df = display_df[display_df["Match Status"] == "MISMATCH"]
        elif filter_option == "Show Matches Only":
            display_df = display_df[display_df["Match Status"] == "MATCH"]

        st.dataframe(display_df, use_container_width=True, height=400)

        # Download Report Buttons
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            csv_data = audit_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Audit Report (CSV)",
                data=csv_data,
                file_name="category_audit_report.csv",
                mime="text/csv"
            )
        with c_dl2:
            mismatch_df = audit_df[audit_df["Match Status"] == "MISMATCH"]
            csv_mismatch = mismatch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⚠️ Download Mismatches Only (CSV)",
                data=csv_mismatch,
                file_name="mismatches_only_report.csv",
                mime="text/csv"
            )
