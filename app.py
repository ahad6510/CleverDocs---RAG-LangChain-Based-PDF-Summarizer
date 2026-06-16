import streamlit as st
import os
import jwt
import tempfile
import json
import time
from datetime import date
from dotenv import load_dotenv
import pdfplumber
from google import genai
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
        return 20
    with open(DB_FILE, "r") as f:
        try:
            data = json.load(f)
            if data.get("date") != today:
                return 20
            return data.get("quota", 20)
        except json.JSONDecodeError:
            return 20

def update_global_quota(new_quota):
    today = str(date.today())
    data = {"date": today, "quota": new_quota}
    with open(DB_FILE, "w") as f:
        json.dump(data, f)

# --- CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client_id = os.getenv("GOOGLE_CLIENT_ID")
client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI", "http://localhost:8501/component/streamlit_oauth.authorize_button")

if not api_key:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        client_id = st.secrets.get("GOOGLE_CLIENT_ID", client_id)
        client_secret = st.secrets.get("GOOGLE_CLIENT_SECRET", client_secret)
        redirect_uri = st.secrets.get("REDIRECT_URI", redirect_uri)

if not api_key:
    st.error("CRITICAL ERROR: GOOGLE_API_KEY not found.")
    st.stop()

try:
    gemini_client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")
    st.stop()

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

def main():
    # --- SAFE COOKIE INITIALIZATION ---
    try:
        cookies = CookieController()
    except Exception:
        cookies = None

    # Initialize Session States
    if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if "file_status" not in st.session_state: st.session_state.file_status = []
    if "user_email" not in st.session_state: st.session_state.user_email = None

    # --- CHECK BROWSER COOKIES ---
    if st.session_state.user_email is None and cookies is not None:
        try:
            saved_email = cookies.get("user_email")
            if saved_email:
                st.session_state.user_email = saved_email
                st.session_state.user_name = cookies.get("user_name")
                st.session_state.user_picture = cookies.get("user_picture")
        except Exception:
            pass

    # --- OAUTH GATEKEEPER ---
    if st.session_state.user_email is None:
        is_oauth_callback = "code" in st.query_params and "state" in st.query_params
        
        if "cookie_sync_done" not in st.session_state and not is_oauth_callback:
            st.session_state.cookie_sync_done = True
            st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)
            with st.spinner("Verifying secure connection..."):
                time.sleep(0.6)
            st.rerun()
        
        else:
            st.markdown('<div class="neon-text">⚡ CleverDocs</div>', unsafe_allow_html=True)
            oauth2 = OAuth2Component(client_id, client_secret, "https://accounts.google.com/o/oauth2/v2/auth", "https://oauth2.googleapis.com/token", "https://oauth2.googleapis.com/token", "https://oauth2.googleapis.com/revoke")
            
            with st.columns([1, 2, 1])[1]:
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                result = oauth2.authorize_button(name="Continue with Google", icon="https://www.google.com/favicon.ico", redirect_uri=redirect_uri, scope="openid email profile", key="google_login", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if result and 'token' in result:
                    id_token = result['token']['id_token']
                    decoded = jwt.decode(id_token, options={"verify_signature": False})
                    st.session_state.user_email = decoded.get('email')
                    st.session_state.user_picture = decoded.get('picture', 'https://www.w3schools.com/howto/img_avatar.png')
                    st.session_state.user_name = "Abdul Ahad Khan (24BCD002)" if st.session_state.user_email == "khanahad6510@gmail.com" else decoded.get('name')
                    
                    if cookies:
                        cookies.set("user_email", st.session_state.user_email, max_age=604800)
                        cookies.set("user_name", st.session_state.user_name, max_age=604800)
                        cookies.set("user_picture", st.session_state.user_picture, max_age=604800)
                    st.query_params.clear()
                    st.rerun()
    else:
        # --- MAIN APP UI ---
        # (Include your existing Upload/Chat logic here)
        st.write(f"Welcome back, {st.session_state.user_name}")

if __name__ == '__main__':
    main()