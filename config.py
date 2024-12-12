import os

# Google Gemini API Key
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  

# Database configuration
LANCEDB_URI = os.path.join(os.getcwd(), "lancedb")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
