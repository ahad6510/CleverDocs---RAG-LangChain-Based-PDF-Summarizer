import os
from dotenv import load_dotenv
from google import genai # <-- NEW SDK IMPORT

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    exit()

print("⚡ Initializing Gemini Client...")
try:
    # Initialize the new client
    client = genai.Client(api_key=api_key)
    
    print("✅ Connection Successful! Checking available models...\n")
    
    # Fetch and print the model list
    for m in client.models.list():
        print(f"- {m.name}")
        
except Exception as e:
    print(f"❌ Error: {e}")