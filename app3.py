

import streamlit as st
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.set_page_config(page_title="RAG Chat Application", layout="wide")
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #a8e6cf, #dcedc1);  /* Fresh Mint + Soft Yellow Gradient */
        font-family: 'Georgia', serif;
    }
    .css-1v0mbdj {
        background-color: #FFFFFF;  /* White for the card-like appearance */
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background: linear-gradient(to bottom, #A1C6E6, #6F99A9);  /* Darker Lavender to Aqua gradient */
    }
    .css-1544g2n {
        font-size: 18px;
        font-weight: bold;
        color: #4C4F6C;  /* Dark Charcoal */
    }
    .css-1d391kg {
        font-size: 16px;
        margin: 5px 0;
    }
    .stButton>button {
        background-color: #FF9AA2;  /* Fresh Coral */
        color: #FFFFFF;
        border-radius: 5px;
        border: none;
        padding: 10px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #FF6F7A;  /* Darker Coral on hover */
    }
    
    /* Aesthetic changes for the chat input box */
    .stTextInput>div>input {
        background: linear-gradient(to right, #E6F7FF, #A1E6D3);  /* Soft gradient from light blue to mint */
        color: #4C4F6C;  /* Dark Charcoal */
        border-radius: 12px;
        padding: 15px;
        font-size: 18px;
        border: none;
        outline: none;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .stTextInput>div>input:focus {
        background: linear-gradient(to right, #A1E6D3, #E6F7FF);  /* Reverse gradient on focus */
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);  /* More prominent shadow on focus */
        border-radius: 12px;
    }

    .stTextInput>div>input::placeholder {
        color: #A1C6E6;  /* Light Blue placeholder text */
    }

    /* Aesthetic changes for the chat container */
    .chat-container {
        background-color: #FFFFFF; /* White background */
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        max-height: 500px;
        overflow-y: auto;
    }

    .user-message {
        background-color: #FFB6B9;  /* Soft Coral for user messages */
        color: #4C4F6C; /* Dark Charcoal */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }

    .assistant-message {
        background-color: #D1C4E9;  /* Soft Lavender for assistant responses */
        color: #4C4F6C; /* Dark Charcoal */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Chat Application with RAG ")

# Load the document
loader = PyPDFLoader("Plugin_Virtual_Assistance_Chat.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create a vector store and retriever
vectorstore = Chroma.from_documents(
    documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
)

# Define the system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Function to initialize fresh chats
def initialize_chats():
    return {"Default": []}  # Start with a default empty chat

# Load chats into session state
if "chats" not in st.session_state:
    st.session_state.chats = initialize_chats()

if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Default"

# Sidebar for chat management
st.sidebar.title("💬 Chat Management")
chat_names = list(st.session_state.chats.keys())

# Add new chat button
if st.sidebar.button("➕ New Chat"):
    new_chat_name = f"Chat {len(chat_names) + 1}"
    st.session_state.chats[new_chat_name] = []
    st.session_state.active_chat = new_chat_name
    chat_names = list(st.session_state.chats.keys())

# Select chat
active_chat = st.sidebar.radio("Select Chat", options=chat_names, index=chat_names.index(st.session_state.active_chat))
st.session_state.active_chat = active_chat

# Delete chat button
if st.sidebar.button("❌ Delete Selected Chat"):
    if active_chat != "Default":
        del st.session_state.chats[active_chat]
        st.session_state.active_chat = "Default"
        chat_names = list(st.session_state.chats.keys())

# Chat input
query = st.chat_input("Type your message here:")

if query:
    # Initialize chat if not exists
    if active_chat not in st.session_state.chats:
        st.session_state.chats[active_chat] = []

    # Add user query to the active chat
    st.session_state.chats[active_chat].append({"query": query})

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Generate the response
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    # Store the response in the active chat
    st.session_state.chats[active_chat][-1]["response"] = answer

# Display the chat history for the active chat
st.subheader(f"Chat: {active_chat}")
if active_chat in st.session_state.chats:
    chat_container = st.container()
    for entry in st.session_state.chats[active_chat]:
        with chat_container:
            # Displaying user and assistant messages with specific styles
            st.markdown(f'<div class="user-message"><strong>You:</strong> {entry["query"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {entry["response"]}</div>', unsafe_allow_html=True)
else:
    st.write("No chat history available.")
