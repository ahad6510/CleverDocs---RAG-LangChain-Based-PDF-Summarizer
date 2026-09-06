import os
from groq import Groq
from dotenv import load_dotenv

# This forces Python to read your .env file
load_dotenv()

# Initialize the Groq client 
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# Fetch the list of models using the SDK method
model_response = client.models.list()

print("Available Groq Models:")
print("-" * 22)

# Loop through the response data to extract just the model IDs
for model in model_response.data:
    print(model.id)