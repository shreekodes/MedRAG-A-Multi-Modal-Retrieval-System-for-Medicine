import streamlit as st
import requests

st.set_page_config(page_title="MedRAG: Multi-Modal Retrieval System", layout="wide")

# --- CSS for pastel design ---
st.markdown("""
<style>
    body {
        background-color: #f7f4f2;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 42px;
        color: #4e4e4e;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6d6d6d;
        margin-bottom: 40px;
    }
    .stContainer {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease-in-out;
    }
    .stContainer:hover {
        transform: translateY(-3px);
    }
    .stButton>button {
        background-color: #c7e4e0;
        color: #333333;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6em 1.4em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #aedbd4;
        color: #000000;
    }
    .scroll-box {
        height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title">MedRAG: Multi-Modal Retrieval System for Medicine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Retrieve medical insights from research papers and imaging data</div>', unsafe_allow_html=True)

# --- LAYOUT ---
col1, col2, col3 = st.columns([1.2, 1, 1.2])

# --- INPUT BOX ---
with col1:
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    st.subheader("Search Query")
    query = st.text_input("Enter your medical query (e.g., 'MRI glioblastoma findings'):")
    submit = st.button("Search")
    st.markdown('</div>', unsafe_allow_html=True)

# --- IMAGE BOX ---
with col2:
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    st.subheader("Relevant Medical Images")
    if submit and query:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966488.png", caption="Sample MRI Result", use_container_width=True)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", caption="Awaiting Query", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- ABSTRACTS BOX ---
with col3:
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    st.subheader("Top Retrieved Abstracts")

    if submit and query:
        try:
            res = requests.post("http://127.0.0.1:8000/docs", json={"query": query})
            if res.status_code == 200:
                data = res.json()
                st.markdown(f"**AI Summary:**\n\n{data['generated_answer']}")
                st.markdown("---")
                st.markdown("**Top Research Abstracts:**")
                st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
                for i, abs_data in enumerate(data['abstracts'], 1):
                    st.markdown(f"**{i}. {abs_data['title']}**")
                    st.markdown(abs_data['abstract'])
                    st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Backend returned an error.")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
    else:
        st.info("Enter a query and click Search to fetch results.")
    st.markdown('</div>', unsafe_allow_html=True)

