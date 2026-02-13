import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("CRITICAL ERROR: GOOGLE_API_KEY not found.")
    st.stop()

genai.configure(api_key=api_key)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CleverDocs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THE "GLASSMORPHISM" CSS ---
st.markdown("""
<style>
    /* 1. FORCE DARK BACKGROUND */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 50%, #1c2331 0%, #0e1117 100%);
        color: #ffffff;
    }

    /* 2. SIDEBAR STYLE */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* 3. TITLE GRADIENT */
    .neon-text {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 242, 96, 0.3);
        margin-bottom: 30px;
    }

    /* 4. CHAT BUBBLES (GLASS STYLE) */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* User Bubble Accent */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 4px solid #0575E6;
    }
    /* AI Bubble Accent */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 4px solid #00f260;
    }

    /* 5. INPUT BOX */
    .stTextInput>div>div>input {
        background-color: #161b22;
        color: white;
        border: 1px solid #30363d;
        border-radius: 10px;
    }

    /* 6. BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #00f260 0%, #0575E6 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    # --- UPDATED TITLE ---
    st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)
    # (Caption removed as requested)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        # --- UPDATED HEADER ---
        st.title("Welcome 👋")
        st.markdown("---")
        pdf_docs = st.file_uploader("Upload Files", accept_multiple_files=True, type=['pdf'])
        
        if st.button("⚡ Initialize System"):
            if not pdf_docs:
                st.warning("No files detected.")
            else:
                with st.status("System Booting...", expanded=True):
                    st.write("Reading Binary Data...")
                    raw_text = get_pdf_text(pdf_docs)
                    st.write("Chunking Vectors...")
                    text_chunks = get_text_chunks(raw_text)
                    st.write("Embedding Neural Net...")
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                st.success("System Online")

        if st.session_state.vectorstore:
            if st.button("🛑 Reset Memory"):
                st.session_state.messages = []
                st.rerun()

    # Chat UI
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

    if prompt := st.chat_input("Input command or query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                st.error("⚠️ SYSTEM OFFLINE: Please upload documents.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Processing..."):
                    docs = st.session_state.vectorstore.similarity_search(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    final_prompt = f"Context:\n{context_text}\n\nQuestion:\n{prompt}"
                    
                    try:
                        # USING THE FREE MODEL
                        model = genai.GenerativeModel('gemini-flash-latest')
                        response = model.generate_content(final_prompt)
                        full_response = response.text
                    except Exception as e:
                        full_response = f"Error: {str(e)}"
                
                # Neon Typing Effect
                displayed_response = ""
                for chunk in full_response.split():
                    displayed_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(displayed_response + "▌")
                message_placeholder.markdown(displayed_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()