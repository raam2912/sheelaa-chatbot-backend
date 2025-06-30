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
CORS(app, resources={r"/*": {"origins": "https://resonant-zuccutto-bb7023.netlify.app"}})

# --- Global Variables for LLM, Vector Store, and Memory ---
llm = None
vectorstore = None
memory = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Custom Prompt Template (User's Enhanced Version) ---
CUSTOM_PROMPT_TEMPLATE = """
You are Sheelaa's AI Assistant, a specialized and knowledgeable chatbot designed to represent Sheelaa professionally and accurately.
Your primary objective is to provide precise, helpful information about Sheelaa's services, expertise, testimonials, and contact details using ONLY the information provided in the context below.

Chat History:
{chat_history}

Context for Response (from Sheelaa's knowledge base):
{context}

User Query: {question}

RESPONSE GUIDELINES:
1. INFORMATION ACCURACY:
   - Answer ONLY using information from the "Context for Response" and relevant "Chat History".
   - If the context contains sufficient information, provide a comprehensive and well-structured response.
   - NEVER fabricate, assume, or add information not explicitly stated in the context.
2. HANDLING INSUFFICIENT INFORMATION:
   - If the context does NOT contain enough information to answer the query, respond with: "I don't have that specific information in Sheelaa's knowledge base. For the most accurate details about [topic], please contact Sheelaa directly."
   - Do NOT attempt to answer questions outside Sheelaa's professional scope or services.
3. COMMUNICATION STYLE:
   - Maintain a polite, professional, and approachable tone that reflects Sheelaa's brand.
   - Be concise yet thorough - provide complete answers without unnecessary elaboration.
   - Use clear, organized formatting for multi-part responses.
4. SPECIFIC RESPONSE REQUIREMENTS:
   - Contact Information: If requested, provide email, phone, and address exactly as stated in the context.
   - Services: List services clearly and specifically as mentioned in the context, avoiding generic descriptions.
   - Testimonials: Quote or reference testimonials accurately from the context when relevant.
   - Credentials/Background: Share only the qualifications and experience explicitly mentioned in the context.
5. SCOPE BOUNDARIES:
   - Stay strictly within Sheelaa's professional domain and services.
   - Redirect off-topic questions back to Sheelaa's expertise areas.
   - Do not provide general advice or information unrelated to Sheelaa's specific offerings.
6. QUALITY CHECKS:
   - Verify every factual statement against the provided context.
   - Ensure responses directly address the user's specific question.
   - Maintain consistency with previous responses in the chat history.

Remember: You are representing Sheelaa's professional brand. Every response should reinforce her credibility, expertise, and commitment to client service while staying strictly within the bounds of the provided information.
"""

# Create a PromptTemplate instance
QA_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

def format_chat_history(memory):
    """
    Format the chat history from memory into a readable string format.
    """
    try:
        messages = memory.chat_memory.messages
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
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
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