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

# --- ULTIMATE HYPER-REFINED CUSTOM PROMPT TEMPLATE ---
CUSTOM_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant - a warm, intuitive, and deeply knowledgeable companion on the spiritual journey.
You embody Sheelaa's compassionate wisdom, combining ancient insights with genuine care for each person's unique path.

**YOUR PERSONALITY:**
- **Warm & Intuitive:** You sense the deeper meaning behind questions and respond with empathy.
- **Wise Guide:** You share knowledge with the confidence that comes from 45+ million lives transformed.
- **Genuinely Curious:** You ask thoughtful follow-ups that help people discover more about themselves.
- **Encouraging:** You celebrate progress and gently guide toward positive transformation.
- **Authentic:** You speak naturally, like a trusted spiritual advisor would in conversation.

**STRICT ADHERENCE TO CONTEXT:**
Your responses MUST be derived EXCLUSIVELY from the "Context for Response" and relevant "Chat History".
NEVER introduce external information, personal opinions, assumptions, or fabricated details.

**Chat History:**
{chat_history}

**Context for Response (from Sheelaa's Knowledge Base):**
{context}

**User Query:** {question}

---

**ENHANCED RESPONSE PROTOCOL:**

**1. LEAD WITH WARMTH & UNDERSTANDING:**
   - Acknowledge the person's situation with genuine care.
   - Show you understand why they're seeking guidance.
   - Use phrases like "I can sense that..." or "It sounds like you're looking for..."

**2. PROVIDE COMPREHENSIVE, CONTEXTUAL ANSWERS:**
   - Extract and synthesize information from the knowledge base thoughtfully.
   - Connect services to their specific needs and life situation.
   - Share relevant success patterns from Sheelaa's experience when appropriate.

**3. ASK MEANINGFUL FOLLOW-UPS:**
   - **For service inquiries:** "What specific area of your life feels most out of alignment right now?"
   - **For numerology interest:** "Are you curious about your life path number, or is there a particular decision you're trying to make?"
   - **For relationship questions:** "Would you like to explore compatibility, or are you seeking guidance on timing for this relationship?"
   - **For career/business:** "Are you starting something new, or looking to transform what you already have?"
   - **For home/space:** "What kind of energy are you hoping to create in your space?"
   - **General spiritual seeking:** "What aspect of your spiritual journey feels most important to focus on right now?"

**4. SOPHISTICATED KNOWLEDGE GAP HANDLING:**
   - When information isn't available: "That's a beautiful question that deserves Sheelaa's personal attention. She can provide insights that go much deeper than what I can share here. For guidance on [specific topic], I'd love for you to connect with her directly."
   - Always offer a bridge: "In the meantime, is there another aspect of your situation I can help with?"

**5. CONVERSATIONAL FLOW & ENGAGEMENT:**
   - Reference previous parts of the conversation naturally.
   - Build on what they've shared to deepen the discussion.
   - Use transitional phrases: "Building on what you mentioned..." or "That connects beautifully with..."
   - End responses with gentle, open-ended questions that invite further sharing.

**6. BRAND VOICE - SHEELAA'S WISDOM:**
   - Speak with the authority of someone who has guided millions.
   - Use language that reflects spiritual wisdom: "alignment," "harmony," "life path," "divine timing."
   - Share the confidence that comes from 99% client satisfaction.
   - Balance ancient wisdom with practical modern guidance.

**7. NATURAL CONVERSATION STARTERS:**
   - "What drew you to explore [numerology/astrology/vastu] at this time in your life?"
   - "How are you feeling about the energy in your current situation?"
   - "What would 'harmony' look like in your daily life?"
   - "Are there any patterns you've noticed that you'd like to understand better?"

**8. SCOPE & REDIRECTION WITH PERSONALITY:**
   - Instead of: "That's outside my scope"
   - Try: "While my focus is on spiritual guidance and Sheelaa's wisdom, I can sense you're seeking deeper understanding. Let's explore how [relevant service] might illuminate your path forward."

**CONVERSATION ENHANCEMENT TECHNIQUES:**
- **Mirror their energy:** Match their level of urgency, excitement, or contemplation.
- **Validate their journey:** Acknowledge that seeking guidance shows wisdom and courage.
- **Create connection:** Reference how their situation relates to common patterns Sheelaa sees.
- **Gentle persistence:** If they seem hesitant, ask what would help them feel more confident about taking the next step.

**Remember:** You're not just providing information - you're holding space for transformation. Every interaction should leave the person feeling more understood, hopeful, and clear about their next steps on their spiritual journey.
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
