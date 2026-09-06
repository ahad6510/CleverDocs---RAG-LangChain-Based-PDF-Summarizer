import streamlit as st
import os
import jwt
import tempfile
import json
import time
import urllib.parse
from datetime import date
from dotenv import load_dotenv
import pdfplumber
import fitz  # PyMuPDF
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from groq import Groq
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from streamlit_oauth import OAuth2Component
from streamlit_cookies_controller import CookieController 

# --- GLOBAL SHARED DATABASE FUNCTIONS ---
DB_FILE = "global_quota.json"

def get_global_quota():
    today = str(date.today())
    if not os.path.exists(DB_FILE):
        return 50
    with open(DB_FILE, "r") as f:
        try:
            data = json.load(f)
            if data.get("date") != today:
                return 50
            return data.get("quota", 50)
        except json.JSONDecodeError:
            return 50

def update_global_quota(new_quota):
    today = str(date.today())
    data = {"date": today, "quota": new_quota}
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# --- CONFIGURATION ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client_id = os.getenv("GOOGLE_CLIENT_ID")
client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8501/component/streamlit_oauth.authorize_button")

if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client_id = st.secrets.get("GOOGLE_CLIENT_ID", client_id)
    client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET", client_secret)
    redirect_uri = st.secrets.get("REDIRECT_URI", redirect_uri)

if not groq_api_key:
    st.error("CRITICAL ERROR: GROQ_API_KEY not found in environment variables.")
    st.stop()

try:
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq Client: {e}")
    st.stop()

@st.cache_resource
def get_ocr_reader():
    return RapidOCR()

ocr_reader = get_ocr_reader()

# --- PAGE CONFIG ---
st.set_page_config(page_title="CleverDocs", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- CSS STYLES ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .neon-text { font-size: 3.5rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #00f260, #0575E6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 30px; }
    .login-container { max-width: 400px; margin: 0 auto; padding: 40px 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
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
            file_text = ""
            pdf.seek(0)
            pdf_bytes = pdf.read()
            
            # 1. Native Digital Text Extraction
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        file_text += content + "\n"
            
            # 2. Local RapidOCR Fallback (for scans/images)
            if not file_text.strip():
                try:
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page_num in range(len(pdf_document)):
                        page = pdf_document.load_page(page_num)
                        pix = page.get_pixmap()
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        
                        ocr_results, _ = ocr_reader(img_array)
                        if ocr_results:
                            for result in ocr_results:
                                file_text += result[1] + " "
                            file_text += "\n"
                except Exception as ocr_e:
                    print(f"RapidOCR Fallback Failed: {ocr_e}")
            
            if not file_text.strip():
                file_status.append({"name": pdf.name, "status": "failed", "msg": "Completely Unreadable"})
            else:
                text += file_text + "\n"
                file_status.append({"name": pdf.name, "status": "success", "msg": "Read Successfully"})
                
        except Exception as e:
            file_status.append({"name": pdf.name, "status": "failed", "msg": "Error Processing File"})
            
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

def main():
    try:
        cookies = CookieController()
    except Exception:
        cookies = None

    if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if "file_status" not in st.session_state: st.session_state.file_status = []
    if "user_email" not in st.session_state: st.session_state.user_email = None
    if "user_name" not in st.session_state: st.session_state.user_name = None
    if "user_picture" not in st.session_state: st.session_state.user_picture = None

    # Restore session from cookies if available
    if st.session_state.user_email is None and cookies is not None:
        try:
            raw_cookie = cookies.get("cleverdocs_session")
            if raw_cookie:
                decoded_data = json.loads(urllib.parse.unquote(raw_cookie))
                st.session_state.user_email = decoded_data.get("email")
                st.session_state.user_name = decoded_data.get("name", "Authorized User")
                st.session_state.user_picture = decoded_data.get("picture", "https://www.w3schools.com/howto/img_avatar.png")
        except Exception:
            pass

    # --- 1. GOOGLE OAUTH GATEKEEPER ---
    if st.session_state.user_email is None:
        is_oauth_callback = "code" in st.query_params and "state" in st.query_params
        
        if "cookie_sync_done" not in st.session_state and not is_oauth_callback:
            st.session_state.cookie_sync_done = True
            st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)
            with st.spinner("Verifying secure connection..."):
                time.sleep(1.2)  
            st.rerun()  
        
        else:
            st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)
            
            if not client_id or not client_secret:
                st.error("⚠️ System Offline: Missing Google OAuth Credentials in Environment Variables.")
                st.stop()
                
            AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
            TOKEN_URL = "https://oauth2.googleapis.com/token"
            REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
            
            oauth2 = OAuth2Component(client_id, client_secret, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, None)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                st.markdown("### Secure Login Required")
                st.markdown("<p style='color: #a1a1aa; margin-bottom: 20px;'>Sign in to access document analysis tools.</p>", unsafe_allow_html=True)
                
                result = None
                try:
                    result = oauth2.authorize_button(
                        name="Continue with Google",
                        icon="https://www.google.com/favicon.ico",
                        redirect_uri=redirect_uri,
                        scope="openid email profile",
                        key="google_login",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"⚠️ Auth Error: {str(e)}")
                    st.query_params.clear()
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
                if result and 'token' in result:
                    id_token = result['token']['id_token']
                    decoded = jwt.decode(id_token, options={"verify_signature": False})
                    
                    st.session_state.user_email = decoded.get('email', 'Authorized User')
                    st.session_state.user_picture = decoded.get('picture', 'https://www.w3schools.com/howto/img_avatar.png')
                    
                    google_name = decoded.get('name', 'Authorized User')
                    if st.session_state.user_email == "khanahad6510@gmail.com":
                        st.session_state.user_name = "Abdul Ahad Khan (24BCD002)"
                    else:
                        st.session_state.user_name = google_name
                    
                    if cookies is not None:
                        try:
                            session_data = {
                                "email": st.session_state.user_email,
                                "name": st.session_state.user_name,
                                "picture": st.session_state.user_picture
                            }
                            encoded_data = urllib.parse.quote(json.dumps(session_data))
                            cookies.set("cleverdocs_session", encoded_data, max_age=604800, path="/")
                            time.sleep(0.5)
                        except Exception:
                            pass
                    
                    st.query_params.clear()
                    st.rerun()

    # --- 2. MAIN APPLICATION WORKSPACE ---
    else:
        if "quota_left" not in st.session_state: 
            st.session_state.quota_left = get_global_quota()
            
        st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)

        with st.sidebar:
            profile_html = f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <img src="{st.session_state.user_picture}" width="45" height="45" style="border-radius: 50%; border: 2px solid #00f260; object-fit: cover;" referrerpolicy="no-referrer">
                <div style="line-height: 1.2; overflow: hidden;">
                    <div style="font-size: 0.75rem; color: #a1a1aa;">Welcome back,</div>
                    <div style="font-weight: 600; color: #ffffff; font-size: 0.95rem; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{st.session_state.user_name}</div>
                </div>
            </div>
            """
            st.markdown(profile_html, unsafe_allow_html=True)
            
            if st.button("🚪 Secure Logout", use_container_width=True):
                if cookies is not None:
                    try:
                        cookies.remove("cleverdocs_session")
                    except Exception:
                        pass
                
                st.session_state.user_email = None
                st.session_state.user_name = None
                st.session_state.user_picture = None
                st.session_state.messages = []
                st.session_state.vectorstore = None
                st.session_state.file_status = []
                
                with st.spinner("Closing session..."):
                    time.sleep(0.5)
                st.rerun()
                
            st.markdown("---")
            st.markdown("<div style='font-size: 0.85rem; color: #a1a1aa; margin-bottom: 5px;'>⚡ Daily API Quota</div>", unsafe_allow_html=True)
            
            progress_val = max(0.0, min(1.0, st.session_state.quota_left / 50.0))
            st.progress(progress_val)
            
            quota_color = "#ff4b4b" if st.session_state.quota_left < 5 else "#00f260"
            st.markdown(f"<div style='text-align: right; font-size: 0.8rem; color: {quota_color}; font-weight: bold;'>{st.session_state.quota_left} / 50 Requests</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            tab1, tab2 = st.tabs(["📄 Upload", "💾 History"])
            
            with tab1:
                pdf_docs = st.file_uploader("Upload Files", accept_multiple_files=True, type=['pdf'])
                if st.button("⚡ Initialize System"):
                    if pdf_docs:
                        with st.status("Processing Documents...", expanded=True) as status:
                            raw_text, file_report = get_pdf_text(pdf_docs)
                            st.session_state.file_status = file_report 
                            
                            if raw_text.strip():
                                text_chunks = get_text_chunks(raw_text)
                                st.session_state.vectorstore = get_vectorstore(text_chunks)
                                status.update(label="✅ System Online", state="complete")
                                st.rerun()
                            else:
                                status.update(label="⚠️ Critical Failure", state="error")
                                st.error("No readable text found.")
                                st.stop()
                    else:
                        st.warning("No files detected.")

                if st.session_state.file_status:
                    st.markdown("### 📊 File Status")
                    for f in st.session_state.file_status:
                        if f["status"] == "success":
                            st.markdown(f'<div class="status-box status-success">✅ {f["name"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-box status-error">❌ {f["name"]}<br><small>{f["msg"]}</small></div>', unsafe_allow_html=True)

            with tab2:
                st.header("Manage Chat")
                if st.session_state.messages:
                    chat_text = "=== ⚡ CleverDocs History ⚡ ===\n\n"
                    for msg in st.session_state.messages:
                        role = "👤 USER" if msg["role"] == "user" else "🤖 AI"
                        chat_text += f"{role}:\n{msg['content']}\n\n" + ("-" * 30) + "\n\n"
                    
                    st.download_button(
                        label="📄 Download Chat (.txt)",
                        data=chat_text,
                        file_name="cleverdocs_chat.txt",
                        mime="text/plain",
                        key=f"download_btn_{len(st.session_state.messages)}"
                    )
                else:
                    st.info("No chat history to download.")
                    
                st.markdown("---")
                if st.button("🛑 Clear Conversation"):
                    st.session_state.messages = []
                    st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.quota_left <= 0:
            st.chat_input("⚠️ Daily API Limit Reached. Check back tomorrow!", disabled=True)
        else:
            if prompt := st.chat_input("Ask about your documents..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            if st.session_state.vectorstore is None:
                with st.chat_message("assistant"):
                    st.error("⚠️ Please upload and process documents first.")
            else:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        user_prompt = st.session_state.messages[-1]["content"]
                        
                        docs = st.session_state.vectorstore.similarity_search(user_prompt)
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        history_str = ""
                        for m in st.session_state.messages[-5:-1]:
                            history_str += f"{m['role'].upper()}: {m['content']}\n"
                        
                        system_prompt = """You are an elite document analysis AI embedded in the CleverDocs system. 
Your objective is to extract information and answer the user's question based STRICTLY on the provided Document Context.

CRITICAL RULES:
1. NO HALLUCINATIONS: You must ONLY use the information explicitly found in the Document Context.
2. IF UNKNOWN: If the answer cannot be found in the context, you must explicitly state: "I could not find the answer to this in the uploaded documents." Do NOT use outside knowledge.
3. PROFESSIONAL FORMATTING: Structure your response beautifully using Markdown. Use bold text for key terms, bullet points for lists, and keep paragraphs concise."""
                        
                        user_payload = f"""=== Chat History ===\n{history_str}\n\n=== Document Context ===\n{context_text}\n\n=== User Question ===\n{user_prompt}"""
                        
                        try:
                            response = groq_client.chat.completions.create(
                                model="openai/gpt-oss-120b", 
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_payload}
                                ],
                                temperature=0.2
                            )
                            full_response = response.choices[0].message.content
                            
                            if st.session_state.quota_left > 0:
                                st.session_state.quota_left -= 1
                                update_global_quota(st.session_state.quota_left)
                                
                        except Exception as e:
                            full_response = f"⚠️ System Error: {str(e)}"
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()

if __name__ == '__main__':
    main()