import os
import git
import shutil
from typing import Tuple, List
from pathlib import Path
import config
import google.generativeai as genai
import lancedb
import magic
from lancedb.embeddings import with_embeddings, EmbeddingFunctionRegistry, get_registry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

def configure_lancedb():
    """Configure LanceDB settings to avoid CPU warnings."""
    import os
    # Set environment variables for LanceDB
    os.environ["LANCE_MAX_READER_THREADS"] = "1"  # Limit reader threads
    os.environ["LANCE_USE_THREADS"] = "1"         # Use threads instead of processes
    
# Add this to the beginning of your main.py file
configure_lancedb()

class GoogleEmbeddingFunction(EmbeddingFunctionRegistry):
    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def __call__(self, texts):
        # Embed the texts using the specified Google model
        if isinstance(texts, str):
            texts = [texts]
        # Convert texts into a list of dictionaries with 'text' key
        texts_dict_list = [{'text': text} for text in texts]
        embeddings = [genai.embed_content(model=self.model_name, content=text_dict, task_type="retrieval_document")['embedding'] for text_dict in texts_dict_list]
        return embeddings

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self(docs)

    def embed_query(self, query: str) -> List[float]:
        # Use retrieval_query for embedding the query
        if isinstance(query, str):
            query = [query]
        texts_dict_list = [{'text': text} for text in query]
        embeddings = [genai.embed_content(model=self.model_name, content=text_dict, task_type="retrieval_query")['embedding'] for text_dict in texts_dict_list]
        return embeddings[0]
    
    def vector_dimensionality(self) -> int:
        # This is a placeholder. You need to find the actual dimensionality of the model.
        # As of now, this information might not be directly available in the API response.
        # You might need to refer to the model's documentation or use trial and error.
        return 768

    @property
    def name(self):
        return self.model_name

    @property
    def parameters(self):
        return {"model": self.model_name}

def clone_repo(repo_url: str, repo_dir: str) -> None:
    """Clones a GitHub repository."""
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(repo_dir)
    try:
        git.Repo.clone_from(repo_url, repo_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository: {e}")

def extract_and_chunk_code(repo_dir: str) -> Tuple[List[str], List[Document]]:
    """
    Extracts code, chunks it, and prepares for embedding.
    Skips binary files and the .git directory.
    """
    code_index = []
    code_texts = []

    for root, dirs, files in os.walk(repo_dir):
        # Ignore the .git directory
        if ".git" in dirs:
            dirs.remove(".git")

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)

            # Use python-magic to detect file type
            mime_type = magic.from_file(file_path, mime=True)

            # Only process text files
            if mime_type.startswith("text/"):
                code_index.append(relative_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as file_content:
                        code_texts.append(file_content.read())
                except UnicodeDecodeError:
                    print(f"Error reading file {file_path}: Not a UTF-8 encoded text file")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
            else:
                print(f"Skipping non-text file: {file_path}")

    # Using LangChain's RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    documents = []
    for i, code_text in enumerate(code_texts):
        chunks = text_splitter.split_text(code_text)
        for chunk_num, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"source": code_index[i], "chunk": chunk_num},
            )
            documents.append(doc)

    return code_index, documents

def create_and_populate_db(documents: List[Document]):
    """Creates or opens a LanceDB table and populates it with embeddings."""
    try:
        print("Starting create_and_populate_db...")
        genai.configure(api_key=config.GOOGLE_API_KEY)

        # Convert Langchain documents to a list of dictionaries for LanceDB
        data = [
            {
                "text": doc.page_content,
                "source": doc.metadata["source"],
                "chunk": doc.metadata["chunk"],
            }
            for doc in documents
        ]

        # Create and register the Google embedding function
        registry = get_registry()
        google_ef = GoogleEmbeddingFunction(api_key=config.GOOGLE_API_KEY)
        registry.register(google_ef)

        # Generate embeddings
        texts_to_embed = [item["text"] for item in data]
        embeddings = google_ef(texts_to_embed)

        # Add embeddings to data
        for i, emb in enumerate(embeddings):
            data[i]["vector"] = emb

        # Create database directory if it doesn't exist
        db_uri = os.path.join(os.getcwd(), "lancedb")
        os.makedirs(db_uri, exist_ok=True)
        print(f"Using database directory: {db_uri}")

        # Connect to LanceDB
        db = lancedb.connect(db_uri)
        table_name = "code_embeddings"

        # Drop existing table if it exists to avoid corruption
        if table_name in db.table_names():
            print(f"Dropping existing table: {table_name}")
            db.drop_table(table_name)

        # Create new table
        print(f"Creating new table: {table_name}")
        table = db.create_table(
            table_name,
            data=data,
            mode="overwrite",
        )
        print(f"Successfully created table: {table_name}")
        return table

    except Exception as e:
        print(f"Error in create_and_populate_db: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise


async def get_gemini_response(query: str, context: List[str], chat_history: List[dict] = None) -> str:
    """Generates a response using Google Gemini, incorporating the retrieved context and chat history."""
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    if not context:
        return "I'm sorry, I couldn't find relevant information in the codebase about that. Could you rephrase your question or ask about something else?"
    
    context_str = "\n\n".join(context)
    
    # Format chat history from dict format
    chat_history_str = ""
    if chat_history and len(chat_history) > 0:
        history_messages = []
        for msg in chat_history[:-1]:  # Exclude the latest message as it's the current query
            role = msg["role"].capitalize()
            content = msg["content"]
            history_messages.append(f"{role}: {content}")
        chat_history_str = "\n".join(history_messages)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_template("""You are an expert code assistant analyzing a GitHub repository. Based on the provided code context and chat history, answer the user's question about the codebase.

Code Context:
{context}

Previous Conversation:
{chat_history}

Current Question: {question}

Instructions:
1. Your response should be based on the information provided in the code context and chat history
2. If you find relevant information, provide a detailed explanation with specific references to the code
3. Pay attention to file names and code sections as shown in the context
4. Use the chat history to maintain context and provide more relevant answers
5. If you're unable to find specific information, acknowledge what you do know and what information seems to be missing

Response:""")

    # Create chain with proper variable mapping
    chain = prompt | llm | StrOutputParser()
    
    try:
        # Execute chain with correct parameter mapping
        response = await chain.ainvoke({
            "context": context_str,
            "chat_history": chat_history_str if chat_history_str else "No previous conversation.",
            "question": query
        })
        
        # Handle generic responses
        generic_responses = [
            "i cannot find that information in the context",
            "i don't see that information in the context",
            "this code context does not contain",
            "i apologize, but i cannot find"
        ]
        
        if any(generic in response.lower() for generic in generic_responses):
            available_info = "Based on the available code context, I can tell you about:\n"
            if "README.md" in context_str:
                available_info += "- Project overview and setup instructions\n"
            if "codey.py" in context_str:
                available_info += "- Main application structure\n"
            if "code_analyzer.py" in context_str:
                available_info += "- Code analysis functionality\n"
            if "streamlit_helpers.py" in context_str:
                available_info += "- Streamlit UI helper functions\n"
                
            return f"I couldn't find specific information about your query in the current context. However, {available_info}\n\nCould you try asking about one of these aspects or rephrase your question?"
            
        return response
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"I encountered an error while analyzing the code: {str(e)}"
        s

def get_context_from_db(query: str, table) -> List[str]:
    """Retrieves relevant context from LanceDB using similarity search."""
    if table is None:
        raise ValueError("Database table is None. Please ensure the table was properly created.")

    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        google_ef = GoogleEmbeddingFunction(api_key=config.GOOGLE_API_KEY)
        
        # Expand query to include relevant keywords
        expanded_query = f"{query} codebase structure architecture implementation"
        query_embedding = google_ef.embed_query(expanded_query)
        
        # Get more potential matches and sort by relevance
        context_docs = (
            table.search(query_embedding)
            .limit(8)  # Increased from 6
            .select(["text", "source", "chunk"])
            .to_pandas()
        )

        if context_docs.empty:
            print("WARNING: No matching documents found in the database.")
            return []

        # Process and format context
        context_texts = []
        seen_content = set()  # To avoid exact duplicates
        
        for _, row in context_docs.iterrows():
            text = row['text'].strip()
            # Skip if we've seen this exact content
            if text in seen_content:
                continue
                
            source = row['source']
            
            # Special handling for README.md to ensure project overview is included
            if source == "README.md" and "# " in text:
                context_texts.insert(0, f"File: {source}\n{text}")
            else:
                context_texts.append(f"File: {source}\n{text}")
                
            seen_content.add(text)

        print(f"Found {len(context_texts)} unique code sections")
        
        # Ensure we don't exceed token limits
        MAX_SECTIONS = 5
        if len(context_texts) > MAX_SECTIONS:
            context_texts = context_texts[:MAX_SECTIONS]
        
        return context_texts

    except Exception as e:
        print(f"An error occurred during search: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise



