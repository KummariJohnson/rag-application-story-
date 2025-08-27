import os
from dotenv import load_dotenv
from pathlib import Path

# New LlamaIndex v0.10+ imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables from a .env file
load_dotenv("main.env")

# --- Environment Variable Validation ---
# It's good practice to ensure all required keys are present.
required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
if not all(key in os.environ for key in required_keys):
    # You can be more specific about which keys are missing if you want
    raise ValueError("One or more required environment variables are missing.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LlamaIndex Global Settings ---
# In new versions of LlamaIndex, we use a global Settings object
# to configure the LLM and embedding model.
print("Configuring LlamaIndex Settings...")

# *** MODIFICATION HERE ***
# Changed model_name from "gemini-pro" to a currently stable and supported version.
# gemini-2.5-pro is the most advanced.
Settings.llm = Gemini(model_name="gemini-2.5-pro", api_key=GOOGLE_API_KEY)

# The embedding model name "models/embedding-001" is generally stable and correct.
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)

# --- Document Loading ---
# SimpleDirectoryReader is the recommended way to load documents from a folder.
# It can handle various file types automatically if the required libraries are installed.
print("Loading documents...")
reader = SimpleDirectoryReader(input_files=[Path("docs/story.txt")])
docs = reader.load_data()
print(f"Loaded {len(docs)} document(s).")

# --- Pinecone Setup ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# The StorageContext defines where the vectors and index metadata are stored.
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Indexing ---
# The index is now created using the StorageContext.
# The LLM and embedding model are pulled from the global Settings.
print("Indexing documents... This may take a moment.")
index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    show_progress=True # Helpful for seeing progress on large document sets
)
print("Indexing complete.")

# --- Query Engine ---
# The query engine is created from the index.
query_engine = index.as_query_engine()

def get_answer(question: str) -> str:
    """
    Queries the engine with a question and returns the response.
    """
    print(f"Querying engine with: '{question}'")
    response = query_engine.query(question)
    return str(response)

# Example of how to use it (optional, for direct script execution)
if __name__ == "__main__":
    print("\n--- RAG Engine Ready ---")
    test_question = "What is the main theme of the story?"
    answer = get_answer(test_question)
    print("\n--- Example Query ---")
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")