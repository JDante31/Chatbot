import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Page config
st.set_page_config(
    page_title="Mental Health Support Chat",
    page_icon="🤗",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
    }
    .stTextInput {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_id = "microsoft/DialoGPT-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_response(prompt, model, tokenizer, max_length=100):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = very random)
            top_k=50,        # Controls diversity
            top_p=0.9,       # Nucleus sampling
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("Mental Health Support Chat 🤗")
    st.markdown("""
    Welcome! I'm here to listen and chat with you. While I'm not a replacement for 
    professional help, I'm happy to discuss your thoughts and feelings.
    """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load model
    tokenizer, model = load_model()
    
    if not tokenizer or not model:
        st.error("Failed to load the model. Please try again later.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Share your thoughts..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, model, tokenizer)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Add helpful resources at the bottom
    with st.expander("📚 Helpful Resources"):
        st.markdown("""
        - Emergency: Call 911 (US) or your local emergency number
        - National Suicide Prevention Lifeline (US): 988 or 1-800-273-8255
        - Crisis Text Line: Text HOME to 741741
        
        Remember: This is an AI chatbot and not a substitute for professional help. 
        If you're experiencing a mental health crisis, please reach out to professionals.
        """)

if __name__ == "__main__":
    main()