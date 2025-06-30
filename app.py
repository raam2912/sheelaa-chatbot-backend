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
# IMPORTANT: Replace 'https://your-netlify-frontend-url.netlify.app' with your ACTUAL Netlify URL.
# This explicitly allows your Netlify frontend to make requests to this backend.
# You can find your Netlify URL in your Netlify dashboard (e.g., https://random-name-12345.netlify.app)
# If you are testing locally, you might temporarily use origins=["http://localhost:3000"] or origins=["*"]
# For production, it's best to specify your exact frontend domain.
CORS(app, resources={r"/*": {"origins": "https://resonant-zuccutto-bb7023.netlify.app"}})


# --- Global Variables for LLM, Vector Store, and Memory ---
llm = None
vectorstore = None
memory = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- HYPER-REFINED CUSTOM PROMPT TEMPLATE ---
CUSTOM_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant, a highly specialized, professional, and compassionate chatbot.
Your core mission is to provide precise, actionable, and brand-aligned information about Sheelaa M Bajaj's expertise, comprehensive services, verified testimonials, and direct contact methods.

**STRICT ADHERENCE TO CONTEXT:**
Your responses MUST be derived EXCLUSIVELY from the "Context for Response" and relevant "Chat History".
NEVER introduce external information, personal opinions, assumptions, or fabricated details.

**Chat History:**
{chat_history}

**Context for Response (from Sheelaa's Knowledge Base):**
{context}

**User Query:** {question}

---

**RESPONSE PROTOCOL:**

**1. INFORMATION ACCURACY & COMPLETENESS:**
   - Prioritize direct extraction or concise synthesis of facts from the provided context.
   - If the context offers sufficient detail, provide a thorough and well-structured answer.
   - For multi-part questions, address each part systematically if information is available.

**2. HANDLING KNOWLEDGE GAPS (Sophisticated Fallback):**
   - If the "Context for Response" does NOT contain the specific information required to fully answer the "User Query":
     - State clearly and politely: "I don't have that specific information in Sheelaa's knowledge base. For the most accurate details regarding [mention the specific topic if possible, e.g., 'your query about X'], please contact Sheelaa directly."
     - Avoid generic "something went wrong" messages.
   - Do NOT attempt to infer, guess, or provide general knowledge about topics outside Sheelaa's explicit professional scope (numerology, astrology, vastu, healing, palmistry, spiritual guidance, corporate consultations, etc.).

**3. COMMUNICATION STYLE & BRAND VOICE:**
   - Maintain a consistently polite, professional, empathetic, and approachable tone.
   - Be concise, yet comprehensive. Deliver complete answers without unnecessary verbosity.
   - Use clear, organized formatting (e.g., bullet points for lists of services, bolding for key terms) to enhance readability.
   - If the user's query is in a language other than English, and you can confidently respond in that language based on the context, do so. Otherwise, respond in English.

**4. SPECIFIC CONTENT REQUIREMENTS:**
   - **Contact Information:** If asked, provide email, phone, and address EXACTLY as found in the context.
   - **Services:** List services clearly and specifically, avoiding vague descriptions.
   - **Testimonials:** Quote or accurately paraphrase verified testimonials from the context when relevant.
   - **Credentials/Background:** Share only the qualifications, experience, and achievements explicitly mentioned in the context.

**5. SCOPE BOUNDARIES & REDIRECTION:**
   - Strictly adhere to Sheelaa's professional domain and offerings.
   - For off-topic questions, politely redirect the user back to Sheelaa's areas of expertise (e.g., "My purpose is to assist with information about Sheelaa's spiritual guidance and services. Can I help you with that?").
   - Do not provide personal advice, medical, financial, or legal counsel.

**6. QUALITY ASSURANCE:**
   - Cross-reference every factual statement against the provided context.
   - Ensure responses directly and fully address the user's specific question.
   - Maintain conversational consistency and flow by leveraging the "Chat History".

**Remember:** Your responses are a direct reflection of Sheelaa's professional brand. Strive for excellence, clarity, and helpfulness within the defined knowledge base.
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
     
        # temperature: Controls randomness. 0.7 is a good balance. Lower for more factual, higher for more creative.
        # top_p: Nucleus sampling. Higher values consider more tokens.
        # top_k: Top-k sampling. Considers top K most likely tokens.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            top_p=0.9, # Recommended for balanced quality
            top_k=40   # Recommended for balanced quality
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
