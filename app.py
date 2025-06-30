import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    print("Please get your API key from https://aistudio.google.com/app/apikey and add it to the .env file.")
    exit(1)

# --- Initialize Google Gemini LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# --- Knowledge Base Setup (RAG) ---
KB_DIR = "knowledge_base"
vectorstore = None # Initialize vectorstore as None

def load_knowledge_base():
    """
    Loads text files from the knowledge_base directory, splits them into chunks,
    and creates a Chroma vector store for RAG.
    This function is called once at startup.
    """
    global vectorstore
    documents = []
    try:
        for filename in os.listdir(KB_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(KB_DIR, filename)
                print(f"Loading document: {filepath}")
                loader = TextLoader(filepath)
                documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents from knowledge base.")

        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        # Create a Chroma vector store from the document chunks
        # This will embed the chunks and store them for efficient similarity search
        vectorstore = Chroma.from_documents(chunks, embeddings)
        print("Chroma vector store created successfully.")

    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        vectorstore = None # Ensure vectorstore is None on error

# Load knowledge base on application startup
with app.app_context():
    load_knowledge_base()

# --- Chatbot Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if vectorstore is None:
        return jsonify({"error": "Knowledge base not loaded. Please check backend logs."}), 500

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
            return_source_documents=True # Optionally return the source documents for debugging/transparency
        )

        # Get response from the QA chain
        response = qa_chain({"query": user_message})
        bot_response = response["result"]

        # Optionally, extract and return source documents
        source_docs = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        print(f"User: {user_message}")
        print(f"Bot: {bot_response}")
        # print(f"Sources: {source_docs}") # Uncomment for debugging source documents

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "message": "Sheelaa Chatbot Backend is running!"}), 200

if __name__ == '__main__':
    # Ensure the knowledge_base directory exists
    if not os.path.exists(KB_DIR):
        print(f"Error: Knowledge base directory '{KB_DIR}' not found.")
        print("Please create the 'knowledge_base' directory inside the 'backend' folder and add your .txt files.")
        exit(1)
    app.run(debug=True, port=5000) # Run Flask app on port 5000
