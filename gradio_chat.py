import gradio as gr
import logging
import os
from query_data import build_chatbot
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
# API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma"

if not API_KEY:
    raise EnvironmentError(
        """API key is not set.
        Either export it in your shell or place it in a .env file."""
    )

# Global variables for bot functions
bot_fn = None
reset_fn = None

def initialize_bot():
    global bot_fn, reset_fn
    try:
        # Get embedding function
        embedding_function = get_embedding_function()
        
        # Build chatbot with api, chroma db, and embeddings
        bot_fn, reset_fn = build_chatbot(
            api_key=API_KEY,
            chroma_path=CHROMA_PATH,
            embedding_function=embedding_function
        )
        
        # Initialize
        if reset_fn:
            greeting = reset_fn()
            # on console
            logger.info("Bot initialized successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize bot: {str(e)}")
        return False

def chat_fn(message: str, history):
    """Gradio chat function that interfaces with the LangChain pipeline."""
    global bot_fn, reset_fn
    
    # Log the received message to console
    logger.info(f"User message: {message}")
    logger.info(f"History length: {len(history) if history else 0}")
    
    # Check if bot is initialized. potential api crash
    if not bot_fn:
        if not initialize_bot():
            return "Sorry, I'm having trouble starting up. Please try again later."
    
    # Handle empty messages
    if not message or not message.strip():
        return "Please ask me a question about KAFD and I'll be happy to help!"
    
    try:
        # Get response from the chatbot
        response = bot_fn(message.strip())
        logger.info(f"Bot response: {response[:100]}...")  # Log first 100 chars only
        return response
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return "I apologize, but I'm experiencing technical difficulties. Please try asking your question again."

def on_clear():
    """Handle chat clear/reset."""
    global reset_fn
    logger.info("Chat cleared. Resetting llm memory")
    try:
        if reset_fn:
            greeting = reset_fn()
            return [], [] 
            # return None
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return [], []
        # return None


css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}

.chat-message {
    font-size: 16px;
    line-height: 1.5;
}

/* Support for RTL text (Arabic) */
.message p {
    unicode-bidi: plaintext;
    text-align: start;
}

/* Make buttons more prominent */
.primary {
    background-color: #2563eb !important;
}

.primary:hover {
    background-color: #1d4ed8 !important;
}

/* Better spacing for chat messages */
.message-wrap {
    margin-bottom: 10px;
}
"""

# Initialize the bot on startup
logger.info("Starting KAFD Concierge application...")
if not initialize_bot():
    logger.error("Failed to initialize bot on startup")

# Create the Gradio interface
with gr.Blocks(title="🏢 KAFD Concierge Assistant", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("""
    # 🏢 KAFD Concierge Assistant
    ### Your intelligent guide to the King Abdullah Financial District
    
    I can help you with:
    - 📍 Distances between locations
    - 🍽️ Restaurant information and dining options
    - 🏗️ Building and landmark details
    - ℹ️ General KAFD facilities and services
    
    *I speak both English and Arabic!*
                
    # 🏢 مساعد كونسيرج كافد  
    ### دليلك الذكي في مركز الملك عبدالله المالي  

    أستطيع مساعدتك في:  
    - 📍 المسافات بين المواقع  
    - 🍽️ معلومات المطاعم وخيارات الطعام  
    - 🏗️ تفاصيل المباني والمعالم  
    - ℹ️ المرافق والخدمات العامة في كافد  

    *أتحدث الإنجليزية والعربية!*
    """)
    
    chatbot = gr.Chatbot(
        value=[],
        height=500,
        show_copy_button=True,
        bubble_full_width=False,
        render_markdown=True
    )
    
    msg = gr.Textbox(
        label="Your message",
        placeholder="Ask me anything about KAFD... (English/Arabic)",
        lines=2,
        max_lines=5,
        show_label=False
    )
    
    with gr.Row():
        submit = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("🗑️ Clear Chat", scale=1)
    
    # Examples to help users get started
    gr.Examples(
        examples=[
            "What restaurants are available in KAFD?",
            "How far is it from the Conference Center to KAFD Academy?",
            "Tell me about the Prayer Hall",
            "ما هي المطاعم المتوفرة؟",
            "كم المسافة من مركز المؤتمرات إلى أكاديمية كافد؟",
        ],
        inputs=msg,
        label="Try these examples:"
    )
    
    # Event handlers
    def user_submit(message, history):
        """Handle user submission."""
        return "", history + [[message, None]]
    
    def bot_response(history):
        """Generate bot response."""
        if history and history[-1][1] is None:
            user_message = history[-1][0]
            bot_message = chat_fn(user_message, history[:-1])
            history[-1][1] = bot_message
        return history
    
    # Connect events
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    submit.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    clear.click(on_clear, None, [chatbot, msg])
    
    # Add footer
    gr.Markdown("""
    ---
    💡 **Tips:** 
    - Ask specific questions for better results
    - I can calculate distances between any two locations in KAFD
    - Feel free to ask in English or Arabic
    """)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    try:
        demo.launch(
            share=False,  # Set to True if you want a public link
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default Gradio port
            show_error=True,        # Show errors in the interface
            quiet=False             # Show startup logs
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error launching application: {str(e)}")
        print(f"Failed to start the application: {str(e)}")