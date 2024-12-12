import streamlit as st
import os
import asyncio
import argparse
import atexit
import json
from datetime import datetime
import base64
import pandas as pd
import lancedb
import config
from services import code_analyzer, streamlit_helpers

# Initialize the Streamlit app first
st.set_page_config(page_title="GitHub Code Tutor", layout="wide")

# Initialize session state immediately after st.set_page_config
if "messages" not in st.session_state:
    st.session_state.messages = []
if "table_initialized" not in st.session_state:
    st.session_state.table_initialized = False
if "current_repo" not in st.session_state:
    st.session_state.current_repo = None

def reset_all_state():
    """Reset all session state variables to their initial values."""
    st.session_state.messages = []
    st.session_state.table_initialized = False
    st.session_state.current_repo = None
    st.session_state.code_index = None
    st.session_state.code_chunks = None
    
    # Drop the LanceDB table if it exists
    try:
        db = lancedb.connect(config.LANCEDB_URI)
        if "code_embeddings" in db.table_names():
            print("Dropping LanceDB table during reset")
            db.drop_table("code_embeddings")
    except Exception as e:
        print(f"Error dropping table: {e}")

def export_chat_history(format_type):
    """Export chat history in specified format."""
    if not st.session_state.messages:
        st.warning("No chat history to export!")
        return

    # Format the chat history
    formatted_chat = []
    for msg in st.session_state.messages:
        formatted_chat.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": msg["role"],
            "content": msg["content"]
        })

    if format_type == "json":
        # JSON export
        json_str = json.dumps(formatted_chat, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="chat_history.json",
            mime="application/json"
        )
    elif format_type == "csv":
        # CSV export
        df = pd.DataFrame(formatted_chat)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="chat_history.csv",
            mime="text/csv"
        )
    elif format_type == "text":
        # Text export
        text_content = "\n\n".join([
            f"[{msg['timestamp']}]\n{msg['role'].upper()}: {msg['content']}"
            for msg in formatted_chat
        ])
        st.download_button(
            label="Download Text",
            data=text_content,
            file_name="chat_history.txt",
            mime="text/plain"
        )

# Argument parser for temp folder
parser = argparse.ArgumentParser(description='GitHub Code Tutor')
parser.add_argument('--temp_folder', type=str, default='./temp', help='Temporary folder to store GitHub files')
args = parser.parse_args()

# Ensure temp folder exists
os.makedirs(args.temp_folder, exist_ok=True)

# Register atexit to delete temp folder on exit
def cleanup_temp_folder():
    if os.path.exists(args.temp_folder):
        import shutil
        shutil.rmtree(args.temp_folder)

atexit.register(cleanup_temp_folder)

st.title("GitHub Code Tutor")
streamlit_helpers.apply_custom_css()

# Sidebar content
with st.sidebar:
    st.header("Options")
    
    # # Clear chat button
    # if st.button("Clear Chat History"):
    #     st.session_state.messages = []
    #     st.rerun()
    
    # Export options
    st.subheader("Export Chat History")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        if st.button("Export JSON"):
            export_chat_history("json")
    with export_col2:
        if st.button("Export CSV"):
            export_chat_history("csv")
    with export_col3:
        if st.button("Export Text"):
            export_chat_history("text")
    
    # Display repository info if available
    if st.session_state.get("current_repo"):
        st.subheader("Current Repository")
        st.write(f"URL: {st.session_state.current_repo}")
        if st.button("Analyze New Repository"):
            reset_all_state()
            st.rerun()

# Main content
github_link = st.text_input("Enter GitHub Repository URL:")

# Button to trigger the analysis
if st.button("Analyze") and github_link:
    with st.spinner("Extracting code and creating embeddings..."):
        try:
            st.session_state.current_repo = github_link
            streamlit_helpers.extract_code(github_link, args.temp_folder)
            st.success("GitHub repository processed. Embeddings created and stored in vector database. You can now ask questions about the codebase.")
        except Exception as e:
            st.error(f"Error processing repository: {str(e)}")
            st.session_state.table_initialized = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the codebase"):
    if not st.session_state.table_initialized:
        st.warning("Please analyze a GitHub repository first before asking questions.")
    else:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(streamlit_helpers.generate_response(prompt))
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    if "database" in str(e).lower():
                        st.session_state.table_initialized = False
                        st.warning("Database connection lost. Please try analyzing the repository again.")



