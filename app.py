from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma # Updated import for Chroma
from langchain_community.document_loaders import TextLoader # Updated import for TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: Replace 'https://your-netlify-frontend-url.netlify.app' with your ACTUAL Netlify URL.
# This explicitly allows your Netlify frontend to make requests to this backend.
# You can find your Netlify URL in your Netlify dashboard (e.g., https://random-name-12345.netlify.app)
# If you are testing locally, you might temporarily use origins=["http://localhost:3000"] or origins=["*"]
# For production, it's best to specify your exact frontend domain.
CORS(app, resources={r"/*": {"origins": "https://resonant-zuccutto-bb7023.netlify.app"}})


# --- Global Variables for LLM and Vector Store ---
llm = None
vectorstore = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Function to Initialize Knowledge Base and LLM ---
def initialize_knowledge_base():
    global llm, vectorstore

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return False

    try:
        # Initialize the LLM (Large Language Model)
        # Using gemini-1.0-pro as it's a stable and widely available model for chat
        llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        print("LLM (Gemini) initialized successfully.")

        # Load documents from the knowledge_base directory
        documents = []
        knowledge_base_path = 'knowledge_base'
        if not os.path.exists(knowledge_base_path):
            print(f"Error: Knowledge base directory '{knowledge_base_path}' not found.")
            return False

        for filename in os.listdir(knowledge_base_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(knowledge_base_path, filename)
                print(f"Loading document: {filepath}")
                loader = TextLoader(filepath)
                documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents from knowledge base.")

        # Split documents into chunks for processing
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

        # Create embeddings for the document chunks
        # GoogleGenerativeAIEmbeddings is used to convert text into numerical vectors
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        # Create a Chroma vector store from the document chunks and embeddings
        # This allows for efficient similarity search later
        vectorstore = Chroma.from_documents(docs, embeddings)
        print("Chroma vector store created successfully.")
        return True

    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return False

# Initialize knowledge base on app startup
# This ensures the LLM and vector store are ready when the server starts
if not initialize_knowledge_base():
    print("Failed to initialize knowledge base on startup. API will not function correctly.")

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint to check the health and status of the backend service.
    """
    return jsonify({"status": "ok", "message": "Sheelaa Chatbot Backend is running!"}), 200

# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to handle chat messages from the frontend.
    Receives a message, processes it using the LLM and knowledge base,
    and returns a response.
    """
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"Received message: {user_message}")

    if llm is None or vectorstore is None:
        print("Error: LLM or vector store not initialized.")
        return jsonify({"error": "Knowledge base or LLM not loaded. Please check backend logs."}), 500

    try:
        # Create a RetrievalQA chain
        # This chain will:
        # 1. Retrieve relevant documents from the vectorstore based on the user's query.
        # 2. Pass the retrieved documents and the user's query to the LLM.
        # 3. The LLM will generate a response based on the provided context.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 'stuff' combines all retrieved docs into one prompt
            retriever=vectorstore.as_retriever(),
            return_source_documents=False # Set to False for production to reduce log verbosity
        )

        # Execute the QA chain with the user's message
        result = qa_chain({"query": user_message})

        # Extract the answer from the result
        bot_response = result.get("result", "An error occurred while processing your request. Please try again.")

        print(f"Sending response: {bot_response}")
        return jsonify({"response": bot_response}), 200

    except Exception as e:
        print(f"Error processing chat message: {e}")
        # Return a generic error message to the frontend for security/user experience
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# This block ensures the Flask development server runs only when the script is executed directly
# On Render, Gunicorn will manage the app, so this block is primarily for local development.
if __name__ == "__main__":
    # In a production environment like Render, Gunicorn will serve the app.
    # The host and port here are for local development testing.
    app.run(host="0.0.0.0", port=5000)
