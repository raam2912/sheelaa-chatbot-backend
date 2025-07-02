from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import datetime # For date/time validation in mock tool

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- CORS Configuration ---
CORS(app, resources={r"/*": {"origins": "https://raam2912.github.io"}})


# --- Global Variables for LLM, Vector Store, Memory, and Agent ---
llm = None
vectorstore = None
memory = None
agent_executor = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Mock Scheduling Tool Implementation ---
def schedule_appointment_tool(service: str, date: str, time: str, client_name: str, contact_info: str) -> str:
    """
    Schedules an appointment for a client with Sheelaa.
    Requires service type (e.g., "Numerology Reading", "Vastu Consultation"),
    preferred date (YYYY-MM-DD), preferred time (HH:MM 24-hour format),
    client's full name, and client's contact information (email or phone number).

    This is a mock function. In a real application, it would interact with a calendar API.
    """
    print(f"\n--- MOCK APPOINTMENT REQUEST ---")
    print(f"Service: {service}")
    print(f"Date: {date}")
    print(f"Time: {time}")
    print(f"Client Name: {client_name}")
    print(f"Contact Info: {contact_info}")
    print(f"--------------------------------\n")

    try:
        datetime.datetime.strptime(date, '%Y-%m-%d')
        datetime.datetime.strptime(time, '%H:%M')
    except ValueError:
        return "I apologize, but the date or time format provided was invalid. Please use YYYY-MM-DD for date and HH:MM (24-hour) for time."

    return f"Great! Your appointment for a {service} on {date} at {time} has been tentatively scheduled for {client_name}. Sheelaa's team will contact you shortly at {contact_info} to confirm all details. Is there anything else I can assist you with regarding Sheelaa's services?"

# --- Define Langchain Tools ---
tools = [
    Tool(
        name="schedule_appointment",
        func=schedule_appointment_tool,
        description="""
        Useful for scheduling appointments for clients.
        Requires service type (e.g., "Numerology Reading", "Vastu Consultation", "Tarot Card Reading", "Reiki Healing", "Astrology Report", "Palmistry Session", "Matchmaking", "Gemstone Recommendation", "Corporate Consultation").
        Preferred date (YYYY-MM-DD), preferred time (HH:MM 24-hour format),
        client's full name, and client's contact information (email or phone number).
        The tool will return a confirmation message or an error if the booking fails.
        """
    )
]

# --- Agent Prompt Template ---
AGENT_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant - a warm, intuitive spiritual guide with 45+ million lives transformed.
Respond with genuine warmth, ancient wisdom, and focused clarity.

**YOUR ESSENCE:**
- **Warmly Intuitive:** You sense deeper meanings, respond with empathy.
- **Confidently Wise:** You share knowledge from transforming 45+ million lives.
- **Genuinely Caring:** You ask one thoughtful follow-up that opens deeper discovery.
- **Encouragingly Authentic:** You celebrate progress, guide gently toward transformation.

**STRICT ADHERENCE TO CONTEXT & TOOLS:**
Your responses MUST be derived EXCLUSIVELY from the "Context for Response" (if answering a question) or the output of the tools you use (if scheduling).
NEVER introduce external information, personal opinions, assumptions, or fabricated details.

**Chat History:** {chat_history}
**Context for Response (from Sheelaa's Knowledge Base):**
{context}

**Agent Scratchpad:**
{agent_scratchpad} # NEW: Added agent_scratchpad here

**User Query: {input}**

---

**ENHANCED RESPONSE PROTOCOL:**

**1. INTENT RECOGNITION:**
   - First, determine if the user wants to schedule an appointment. If so, gather ALL necessary details (service, date, time, name, contact) before calling the `schedule_appointment` tool.
   - If the user is asking a question, use the knowledge base.

**2. LEAD WITH WARMTH & UNDERSTANDING:**
   - Acknowledge the person's situation with genuine care.
   - Show you understand why they're seeking guidance.
   - Vary your opening language naturally; avoid overusing "I can sense that..." or "It sounds like you're looking for...".

**3. PROVIDE COMPREHENSIVE, CONTEXTUAL ANSWERS (for info queries):**
   - CRITICAL: When asked about pricing for a specific service or product, if a numerical value (e.g., "₹15,000", "$25") is found in the 'Context', provide that exact price directly in your response.
   - Connect services to their specific needs and life situation.
   - Share relevant success patterns from Sheelaa's experience when appropriate.
   - Include practical next steps or what they can expect.

**4. ASK MEANINGFUL FOLLOW-UPS (if more info is needed for tool or conversation):**
   - If scheduling, and details are missing: "To schedule your [service type] appointment, I'll also need your preferred date (YYYY-MM-DD), time (HH:MM 24-hour), your full name, and a contact email or phone number. What details can you provide?"
   - For service needs: "What area of your life feels most out of alignment?"
   - For career interest: "Are you looking to discover your ideal career path or optimize timing for a transition?"
   - For relationship questions: "Would you like compatibility insights or guidance on relationship timing?"
   - For numerology interest: "Are you curious about your life path number or facing a specific decision?"
   - For Vastu/Space interest: "What kind of energy shift are you hoping to create?"
   - General spiritual seeking: "What aspect of your spiritual journey feels most important to focus on right now?"

**5. SOPHISTICATED KNOWLEDGE GAP HANDLING (for info queries):**
   - ONLY if a specific numerical price is NOT found in the 'Context' for a pricing query (e.g., for "Independent Bungalow" Vastu or if the price is genuinely missing): "That deserves Sheelaa's personal wisdom - she can provide insights that go much deeper. For [specific topic], I'd love for you to connect with her directly. Meanwhile, what other aspect can I help illuminate?"
   - Always offer a bridge: "In the meantime, is there another aspect of your situation I can help with?"

**6. CONVERSATIONAL FLOW & ENGAGEMENT:**
   - Reference previous parts of the conversation naturally.
   - Build on what they've shared to deepen the discussion.
   - Use transitional phrases: "Building on what you mentioned..." or "That connects beautifully with...".
   - End responses with gentle, open-ended questions that invite further sharing.

**7. BRAND VOICE - SHEELAA'S WISDOM:**
   - Speak with the authority of someone who has guided millions.
   - Use language that reflects spiritual wisdom: "alignment," "harmony," "life path," "divine timing."
   - Share the confidence that comes from 99% client satisfaction.
   - Balance ancient wisdom with practical modern guidance.

**8. SCOPE & REDIRECTION WITH PERSONALITY:**
   - Instead of: "That's outside my scope"
   - Try: "While my focus is on spiritual guidance and Sheelaa's wisdom, I can sense you're seeking deeper understanding. Let's explore how [relevant service] might illuminate your path forward."

**ANTI-REPETITION RULES:**
- Never repeat phrases like "45+ million lives transformed" or "99% client satisfaction" in consecutive responses.
- Vary your opening language - avoid overusing "I can sense" or "It sounds like".
- Don't mention "ancient wisdom," "life path," or "alignment" in every response.
- NO name-dropping of testimonials (avoid "Rahul M." or "Smita P." references).
- Reference previous conversation points naturally without restating them.
- If information was already shared, build upon it rather than repeating it.
- Avoid asking the same question twice or restating the user's situation back to them.

**CONVERSATION MEMORY:**
- Track what services/concepts have been mentioned.
- Reference previous responses: "Building on what we discussed..." or "Since you're interested in...".
- Avoid explaining the same service multiple times to the same person.

**IMPACT-FOCUSED WRITING:**
- Lead with transformation, not process.
- Use powerful, decisive language: "reveals," "unlocks," "transforms," "illuminates."
- Focus on the end result they'll experience.
- Be specific about what they'll discover or achieve.
- Cut filler words and get straight to the value.

Remember: Create connection and clarity in fewer words. Every sentence should move them closer to transformation.
"""

# Create a PromptTemplate instance for the agent
AGENT_PROMPT = PromptTemplate(
    template=AGENT_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "input", "agent_scratchpad"] # NEW: Added agent_scratchpad here
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


# --- Function to Initialize Knowledge Base, LLM, and Agent ---
def initialize_knowledge_base_and_agent():
    global llm, vectorstore, memory, agent_executor

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return False

    try:
        # Initialize the LLM with tools
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

        # --- Configure the Agent ---
        # The agent needs access to the retriever for RAG and the tools for function calling.
        # It also needs the memory for chat history.

        # Create a retriever for the agent to use for knowledge base lookups
        retriever = vectorstore.as_retriever()

        # Define the agent
        # The agent will decide whether to use a tool or the retriever based on the prompt.
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools, # Pass the defined tools here
            prompt=AGENT_PROMPT # Use the agent-specific prompt
        )

        # Create the AgentExecutor
        # This is the runnable that will execute the agent's decisions.
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # Set to True for detailed logs of agent's thought process
            memory=memory, # Pass memory to the agent executor
            handle_parsing_errors=True # Good for debugging agent's output
        )
        print("Agent Executor initialized successfully.")

        return True

    except Exception as e:
        print(f"Error initializing knowledge base and agent: {e}")
        return False

# Initialize knowledge base and agent on app startup
if not initialize_knowledge_base_and_agent():
    print("Failed to initialize knowledge base and agent on startup. API will not function correctly.")

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
    Receives a message, processes it using the LLM and agent,
    and returns a response.
    """
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"Received message: {user_message}")

    if agent_executor is None: # Check for agent_executor instead of llm/vectorstore/memory directly
        print("Error: Agent Executor not initialized.")
        return jsonify({"error": "Chatbot not fully initialized. Please check backend logs."}), 500

    try:
        # The agent executor handles memory internally when passed during initialization.
        # We just need to pass the user's input.
        # The agent will decide whether to use a tool or retrieve from the knowledge base.
        result = agent_executor.invoke({"input": user_message})

        # The agent's final answer is typically under the 'output' key
        bot_response = result.get("output", "I apologize, but I couldn't process your request at this moment. Please try again or rephrase your query.")

        # AgentExecutor handles saving context to memory internally when memory is provided
        # during its initialization. So, no manual save_context needed here.

        print(f"Sending response: {bot_response}")
        return jsonify({"response": bot_response}), 200

    except Exception as e:
        print(f"Error processing chat message: {e}")
        # Log the full exception for debugging on Render
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# This block ensures the Flask development server runs only when the script is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
