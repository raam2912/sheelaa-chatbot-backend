from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage # For formatting chat history

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: Updated to allow requests from your GitHub Pages frontend URL.
# This explicitly allows your GitHub Pages frontend to make requests to this backend.
CORS(app, resources={r"/*": {"origins": "https://raam2912.github.io"}})


# --- Global Variables for LLM, Vector Store, and Memory ---
llm = None
vectorstore = None
memory = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CUSTOM_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant - a warm, intuitive spiritual guide with 45+ million lives transformed.
Respond with genuine warmth, ancient wisdom, and focused clarity.

**YOUR ESSENCE:**
- **Warmly Intuitive:** Sense deeper meanings, respond with empathy
- **Confidently Wise:** Share knowledge from transforming 45+ million lives  
- **Genuinely Caring:** Ask one thoughtful follow-up that opens deeper discovery
- **Encouragingly Authentic:** Celebrate progress, guide gently toward transformation

**RESPONSE RULES:**
- Use ONLY the provided Context and Chat History - no external information
- Keep responses focused and impactful (2-3 paragraphs maximum)
- Lead with understanding, provide key insights, end with one meaningful question
- Use Sheelaa's spiritual language: "alignment," "harmony," "life path," "divine timing"

**Chat History:** {chat_history}
**Context:** {context}
**User Query:** {question}

---

**RESPONSE STRUCTURE:**

**1. WARM ACKNOWLEDGMENT (1-2 sentences):**
Connect with their situation using phrases like "I can sense..." or "It sounds like you're seeking..."

**2. FOCUSED GUIDANCE (2-3 sentences):**
Extract the most relevant insights from the knowledge base. Connect services to their specific needs with confidence that comes from 99% client satisfaction.

**3. ONE MEANINGFUL QUESTION:**
Choose the most impactful follow-up:
- Service needs: "What area of your life feels most out of alignment?"
- Numerology: "Are you curious about your life path number, or facing a specific decision?"
- Relationships: "Would you like to explore compatibility or timing guidance?"
- Career/Business: "Are you starting fresh or transforming what exists?"
- Vastu/Space: "What energy do you want to create in your space?"
- General seeking: "What aspect of your spiritual journey feels most important right now?"

**KNOWLEDGE GAP RESPONSE:**
"That deserves Sheelaa's personal wisdom - she can provide insights that go much deeper. For [specific topic], I'd love for you to connect with her directly. Meanwhile, what other aspect can I help illuminate?"

**VOICE REMINDERS:**
- Speak with authority of someone who's guided millions
- Balance ancient wisdom with practical guidance  
- Mirror their energy level naturally
- Validate their courage in seeking guidance
- End responses feeling more understood and hopeful

Remember: Create connection and clarity in fewer words. Every sentence should move them closer to transformation.
"""

# Create a PromptTemplate instance
QA_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

# --- Helper function to format chat history ---
def format_chat_history(memory_instance):
    """
    Format the chat history from memory into a readable string format for the prompt.
    """
    try:
        messages = memory_instance.chat_memory.messages
        if not messages:
            return "No previous conversation."

        formatted_history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")

        return "\n".join(formatted_history)
    except Exception as e:
        print(f"Error formatting chat history: {e}")
        return "No previous conversation."


# --- Function to Initialize Knowledge Base and LLM ---
def initialize_knowledge_base():
    global llm, vectorstore, memory

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return False

    try:
        # Initialize the LLM (Large Language Model)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            top_p=0.9,
            top_k=40
        )
        print("LLM (Gemini) initialized successfully.")

        # Initialize conversational memory
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        print("Conversational memory initialized.")

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        # Create a Chroma vector store from the document chunks and embeddings
        vectorstore = Chroma.from_documents(docs, embeddings)
        print("Chroma vector store created successfully.")
        return True

    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return False

# Initialize knowledge base on app startup
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

    if llm is None or vectorstore is None or memory is None:
        print("Error: LLM, vector store, or memory not initialized.")
        return jsonify({"error": "Chatbot not fully initialized. Please check backend logs."}), 500

    try:
        # Format the chat history for the prompt
        formatted_chat_history = format_chat_history(memory)

        # Get relevant documents from the vector store
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(user_message)

        # Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create the prompt with all required variables
        prompt_input = {
            "chat_history": formatted_chat_history,
            "context": context,
            "question": user_message
        }

        # Format the prompt
        formatted_prompt = QA_PROMPT.format(**prompt_input)

        # Get response from LLM
        bot_response = llm.invoke(formatted_prompt).content

        # Save the user message and bot response to memory
        memory.save_context({"input": user_message}, {"output": bot_response})

        print(f"Sending response: {bot_response}")
        return jsonify({"response": bot_response}), 200

    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# This block ensures the Flask development server runs only when the script is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
