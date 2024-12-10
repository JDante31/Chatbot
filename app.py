import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Page config
st.set_page_config(
    page_title="Mental Health Support Chat",
    page_icon="🤗",
    layout="centered"
)

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
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            # Add these parameters to prevent repetition
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    
    # Decode and clean the response
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    
    # Check for crisis keywords
    crisis_keywords = ['suicide', 'kill', 'die', 'hurt', 'harm', 'end my life']
    if any(keyword in prompt.lower() for keyword in crisis_keywords):
        return ("""I notice you're expressing thoughts of harm. Please know that you're not alone, and help is available:

1. Call 988 (US Suicide & Crisis Lifeline) - Available 24/7
2. Text HOME to 741741 (Crisis Text Line)
3. Call your local emergency number
4. Reach out to a trusted friend, family member, or mental health professional

Would you like to talk more about what's troubling you? I'm here to listen, but please remember I'm an AI and not a replacement for professional help.""")
    
    # If response is empty or just whitespace, provide a default response
    if not response.strip():
        return "I hear you. Can you tell me more about what's troubling you? Remember, I'm here to listen, though I'm not a replacement for professional help."
    
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
        
    # Initialize conversation history for the model
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

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

        # Update conversation history
        st.session_state.conversation_history.append(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, model, tokenizer)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_history.append(response)

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