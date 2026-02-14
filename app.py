import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from gtts import gTTS
from io import BytesIO

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass 

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

# --- CSS STYLES ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .neon-text {
        font-size: 3.5rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00f260, #0575E6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 30px;
    }
    .stChatMessage { background: rgba(255, 255, 255, 0.05); border-radius: 15px; }
    
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .status-success { background-color: #0e4429; border: 1px solid #00f260; color: #00f260; }
    .status-error { background-color: #4c1d1d; border: 1px solid #ff4b4b; color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# --- LOGIC FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    file_status = [] 
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            file_text = ""
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    file_text += content
            if not file_text.strip():
                file_status.append({"name": pdf.name, "status": "failed", "msg": "Empty/Scanned"})
            else:
                text += file_text + "\n"
                file_status.append({"name": pdf.name, "status": "success", "msg": "Read Successfully"})
        except Exception as e:
            file_status.append({"name": pdf.name, "status": "failed", "msg": "Error Reading"})
    return text, file_status

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore(text_chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def text_to_audio(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.sidebar.error(f"Audio Generation Failed: {e}")
        return None

def main():
    st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_status" not in st.session_state:
        st.session_state.file_status = []

    with st.sidebar:
        st.title("Welcome 👋")
        tab1, tab2 = st.tabs(["📄 Upload", "💾 History"])
        
        with tab1:
            pdf_docs = st.file_uploader("Upload Files", accept_multiple_files=True, type=['pdf'])
            if st.button("⚡ Initialize System"):
                if not pdf_docs:
                    st.warning("No files detected.")
                else:
                    with st.status("Processing...", expanded=True) as status:
                        raw_text, file_report = get_pdf_text(pdf_docs)
                        st.session_state.file_status = file_report 
                        if not raw_text.strip():
                            status.update(label="⚠️ Critical Failure", state="error")
                            st.error("No readable text found.")
                            st.stop()
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vectorstore = get_vectorstore(text_chunks)
                        status.update(label="✅ System Online", state="complete")

            if st.session_state.file_status:
                st.markdown("### 📊 File Status")
                for f in st.session_state.file_status:
                    if f["status"] == "success":
                        st.markdown(f'<div class="status-box status-success">✅ {f["name"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-box status-error">❌ {f["name"]}<br><small>{f["msg"]}</small></div>', unsafe_allow_html=True)

        with tab2:
            st.header("Manage Chat")
            
            # --- DYNAMIC DOWNLOAD FIX ---
            if st.session_state.messages:
                chat_text = "=== ⚡ CleverDocs History ⚡ ===\n\n"
                for msg in st.session_state.messages:
                    role = "👤 USER" if msg["role"] == "user" else "🤖 AI"
                    chat_text += f"{role}:\n{msg['content']}\n\n"
                    chat_text += "-" * 30 + "\n\n"
                
                # Unique key ensures button data is always fresh
                st.download_button(
                    label="📄 Download Chat (.txt)", 
                    data=chat_text, 
                    file_name="cleverdocs_chat.txt", 
                    mime="text/plain",
                    key=f"download_btn_{len(st.session_state.messages)}"
                )
            else:
                st.info("No chat history found.")
                
            st.markdown("---")
            if st.button("🛑 Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

    # Chat UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                st.error("⚠️ Please upload and process documents first.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    docs = st.session_state.vectorstore.similarity_search(prompt)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    history = "".join([f"{m['role'].upper()}: {m['content']}\n" for m in st.session_state.messages[-3:]])
                    
                    final_prompt = f"History:\n{history}\nContext:\n{context_text}\nQuestion:\n{prompt}"
                    
                    try:
                        model = genai.GenerativeModel('gemini-flash-latest')
                        response = model.generate_content(final_prompt)
                        full_response = response.text
                    except Exception as e:
                        full_response = f"Error: {str(e)}"
                
                message_placeholder.markdown(full_response)
                
                # Audio Generation
                audio_bytes = text_to_audio(full_response)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                
                # Final save and UI refresh
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()

if __name__ == '__main__':
    main()