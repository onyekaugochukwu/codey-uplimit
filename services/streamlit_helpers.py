import streamlit as st
from typing import List
from . import code_analyzer
import lancedb
import config

def apply_custom_css():
    """Applies custom CSS with theme support."""
    theme = getattr(st.session_state, 'theme', 'light')
    
    if theme == 'dark':
        background_color = "#1E1E1E"
        text_color = "#FFFFFF"
        border_color = "#333333"
    else:
        background_color = "#f0f2f6"
        text_color = "#000000"
        border_color = "#E0E0E0"

    st.markdown(f"""
    <style>
    .main {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton button {{
        background-color: #4CAF50;
        color: white;
    }}
    .stTextInput input {{
        background-color: {background_color};
        color: {text_color};
        border: 0.51px solid {border_color};
    }}
    .stMarkdown h3 {{
        color: #4CAF50;
    }}
    .chat-message {{
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid {border_color};
    }}
    .user-message {{
        background-color: #E3F2FD;
    }}
    .assistant-message {{
        background-color: #F5F5F5;
    }}
    .timestamp {{
        font-size: 0.8rem;
        color: #666;
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_state():
    """Initializes session state with additional features."""
    if 'code_index' not in st.session_state:
        st.session_state.code_index = None
    if 'code_chunks' not in st.session_state:
        st.session_state.code_chunks = None
    if 'table_initialized' not in st.session_state:
        st.session_state.table_initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

def reset_state():
    """Resets all session state variables."""
    st.session_state.code_index = None
    st.session_state.code_chunks = None
    st.session_state.table_initialized = False
    st.session_state.messages = []
    st.session_state.current_repo = None

def get_db_table():
    """Gets or creates the LanceDB table."""
    try:
        db = lancedb.connect(config.LANCEDB_URI)
        table_name = "code_embeddings"
        if table_name in db.table_names():
            return db.open_table(table_name)
        return None
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def extract_code(github_link: str, temp_folder: str) -> None:
    """Extracts code, creates embeddings, and stores in vector db."""
    code_analyzer.clone_repo(github_link, temp_folder)
    code_index, code_chunks = code_analyzer.extract_and_chunk_code(temp_folder)

    st.session_state.code_index = code_index
    st.session_state.code_chunks = code_chunks
    st.session_state.current_repo = github_link

    # Create embeddings and store in vector database
    with st.spinner("Creating embeddings and storing in vector database..."):
        code_analyzer.create_and_populate_db(code_chunks)
        st.session_state.table_initialized = True

async def generate_response(prompt: str) -> str:
    """Generates a response using Gemini and context from vector db."""
    if not st.session_state.table_initialized:
        return "Please extract code first before asking questions."
    
    table = get_db_table()
    if table is None:
        return "Unable to connect to the database. Please try again."
    
    context = code_analyzer.get_context_from_db(prompt, table)
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-4:]
    ]
    response = await code_analyzer.get_gemini_response(prompt, context, chat_history)
    return response

def clear_chat_history():
    """Clears the chat history."""
    st.session_state.messages = []


# import streamlit as st
# from typing import List
# from . import code_analyzer
# import lancedb
# import config


# def apply_custom_css():
#     """Applies custom CSS."""
#     st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;
#     }
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#     }
#     .stTextInput input {
#         background-color: #fff;
#         color: #000;
#         border: 0.51px solid #333333;
#     }
#     .stMarkdown h3 {
#         color: #4CAF50;
#     }
#     .stSelectbox>div[data-baseweb="select"] {
#         background-color: #ffffff;
#         color: #000000;
#         border: 0.51px solid #333333;
#     }
#     .stSelectbox>div[data-baseweb="select"] .css-1wa3eu0-placeholder {
#         color: #ffffff;
#     }
#     .stSelectbox>div[data-baseweb="select"] .css-1hb7zxy-IndicatorsContainer {
#         color: #ffffff;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def initialize_state():
#     """Initializes session state."""
#     if 'code_index' not in st.session_state:
#         st.session_state.code_index = None
#     if 'code_chunks' not in st.session_state:
#         st.session_state.code_chunks = None
#     if 'table_initialized' not in st.session_state:
#         st.session_state.table_initialized = False
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

# def get_db_table():
#     """Gets or creates the LanceDB table."""
#     try:
#         db = lancedb.connect(config.LANCEDB_URI)
#         table_name = "code_embeddings"
#         if table_name in db.table_names():
#             return db.open_table(table_name)
#         return None
#     except Exception as e:
#         print(f"Error connecting to database: {str(e)}")
#         return None

# def extract_code(github_link: str, temp_folder: str) -> None:
#     """Extracts code, creates embeddings, and stores in vector db."""
#     code_analyzer.clone_repo(github_link, temp_folder)
#     code_index, code_chunks = code_analyzer.extract_and_chunk_code(temp_folder)

#     st.session_state.code_index = code_index
#     st.session_state.code_chunks = code_chunks

#     # Create embeddings and store in vector database
#     with st.spinner("Creating embeddings and storing in vector database..."):
#         code_analyzer.create_and_populate_db(code_chunks)
#         st.session_state.table_initialized = True

# async def generate_response(prompt: str) -> str:
#     """Generates a response using Gemini and context from vector db."""
#     if not st.session_state.table_initialized:
#         return "Please extract code first before asking questions."
    
#     table = get_db_table()
#     if table is None:
#         return "Unable to connect to the database. Please try again."
    
#     context = code_analyzer.get_context_from_db(prompt, table)
#     # Get chat history from session state
#     chat_history = [
#         {"role": msg["role"], "content": msg["content"]}
#         for msg in st.session_state.messages[-4:]  # Get last 4 messages for context
#     ]
#     response = await code_analyzer.get_gemini_response(prompt, context, chat_history)
#     return response

# def clear_chat_history():
#     """Clears the chat history."""
#     st.session_state.messages = []


# # import streamlit as st
# # import asyncio
# # from . import code_analyzer
# # import lancedb
# # import config
# # from typing import List
# # from langchain.memory import ConversationBufferMemory
# # from langchain_core.messages import HumanMessage, AIMessage

# # def apply_custom_css():
# #     """Applies custom CSS."""
# #     st.markdown("""
# #     <style>
# #     .main {
# #         background-color: #f0f2f6;
# #     }
# #     .stButton button {
# #         background-color: #4CAF50;
# #         color: white;
# #     }
# #     .stTextInput input {
# #         background-color: #fff;
# #         color: #000;
# #         border: 0.51px solid #333333;
# #     }
# #     .stMarkdown h3 {
# #         color: #4CAF50;
# #     }
# #     .stSelectbox>div[data-baseweb="select"] {
# #         background-color: #ffffff;
# #         color: #000000;
# #         border: 0.51px solid #333333;
# #     }
# #     .stSelectbox>div[data-baseweb="select"] .css-1wa3eu0-placeholder {
# #         color: #ffffff;
# #     }
# #     .stSelectbox>div[data-baseweb="select"] .css-1hb7zxy-IndicatorsContainer {
# #         color: #ffffff;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)

# # def initialize_state():
# #     """Initializes session state."""
# #     if 'code_index' not in st.session_state:
# #         st.session_state.code_index = None
# #     if 'code_chunks' not in st.session_state:
# #         st.session_state.code_chunks = None
# #     if 'table_initialized' not in st.session_state:
# #         st.session_state.table_initialized = False
# #     if 'messages' not in st.session_state:
# #         st.session_state.messages = []
# #     if 'memory' not in st.session_state:
# #         st.session_state.memory = ConversationBufferMemory(
# #             return_messages=True,
# #             memory_key="chat_history"
# #         )

# # async def generate_response(prompt: str) -> str:
# #     """Generates a response using Gemini and context from vector db."""
# #     if not st.session_state.table_initialized:
# #         st.error("Please extract code first before asking questions.")
# #         return ""
    
# #     table = get_db_table()
# #     if table is None:
# #         st.error("Unable to connect to the database. Please try again.")
# #         return ""
    
# #     # Get context from the database
# #     context = code_analyzer.get_context_from_db(prompt, table)
    
# #     # Add the user's message to the chat history
# #     st.session_state.messages.append(HumanMessage(content=prompt))
    
# #     # Get response from Gemini
# #     response = await code_analyzer.get_gemini_response(
# #         query=prompt,
# #         context=context,
# #         chat_history=st.session_state.messages
# #     )
    
# #     # Add the assistant's response to the chat history
# #     st.session_state.messages.append(AIMessage(content=response))
    
# #     return response

# # def display_chat_history():
# #     """Displays the chat history in a conversational format."""
# #     for message in st.session_state.messages:
# #         if isinstance(message, HumanMessage):
# #             with st.chat_message("user"):
# #                 st.markdown(message.content)
# #         elif isinstance(message, AIMessage):
# #             with st.chat_message("assistant"):
# #                 st.markdown(message.content)

# # def clear_chat_history():
# #     """Clears the chat history."""
# #     st.session_state.messages = []
# #     st.session_state.memory.clear()

# # def get_db_table():
# #     """Gets or creates the LanceDB table."""
# #     try:
# #         db = lancedb.connect(config.LANCEDB_URI)
# #         table_name = "code_embeddings"
# #         if table_name in db.table_names():
# #             return db.open_table(table_name)
# #         return None
# #     except Exception as e:
# #         print(f"Error connecting to database: {str(e)}")
# #         return None

# # def extract_code(github_link: str, temp_folder: str) -> None:
# #     """Extracts code, creates embeddings, and stores in vector db."""
# #     code_analyzer.clone_repo(github_link, temp_folder)
# #     code_index, code_chunks = code_analyzer.extract_and_chunk_code(temp_folder)

# #     st.session_state.code_index = code_index
# #     st.session_state.code_chunks = code_chunks

# #     # Create embeddings and store in vector database
# #     with st.spinner("Creating embeddings and storing in vector database..."):
# #         code_analyzer.create_and_populate_db(code_chunks)
# #         st.session_state.table_initialized = True

# # async def generate_response(prompt: str) -> str:
# #     """Generates a response using Gemini and context from vector db."""
# #     if not st.session_state.table_initialized:
# #         st.error("Please extract code first before asking questions.")
# #         return ""
    
# #     table = get_db_table()
# #     if table is None:
# #         st.error("Unable to connect to the database. Please try again.")
# #         return ""
    
# #     context = code_analyzer.get_context_from_db(prompt, table)
# #     response = await code_analyzer.get_gemini_response(prompt, context)
# #     return response