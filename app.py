# import streamlit as st
# import tempfile
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import time
# import requests
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv
# import os
# from langchain.schema import Document
# from langchain.schema import HumanMessage, AIMessage, SystemMessage


# load_dotenv()

# embeddings = AzureOpenAIEmbeddings(
#     model=os.getenv("EMBEDDING_MODEL"),
#     api_version=os.getenv("API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     api_key=os.getenv("API_KEY")
# )

# llm = AzureChatOpenAI(
#     model=os.getenv("COMPLETION_MODEL"),
#     api_version=os.getenv("API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     temperature=0)

# VECTORSTORE_DIR = "vectorstore"



# def chat_memory_setup(llm):
#     memory = ConversationBufferMemory(
#         llm = llm,
#         max_token_limit=1000,
#         return_messages=True,
#     )
#     formatted_messages = []
#     for msg in st.session_state.messages:
#         if isinstance(msg, dict):
#             role = msg.get("role")
#             content = msg.get("content", "")
#             if role == "user":
#                 formatted_messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 formatted_messages.append(AIMessage(content=content))
#             elif role == "system":
#                 formatted_messages.append(SystemMessage(content=content))

#     memory.chat_memory.messages = formatted_messages

#     return memory

# def llm_handler(input_text, knowledge_base):
#     system_prompt = """You are an AI assistant, designed to answer questions based on the provided documents. Stick to the provided documents always. Provide users with accurate information.
#     If you don't know the answer, say "I don't know" instead of making up an answer."""

#     chat_history = chat_memory_setup(llm = llm)
#     parser = StrOutputParser()
#     SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
#         [
#             ("human", """
#                 {system_prompt}
             
#              Relevant Knowledge Base: {knowledge_base}
            
#             Chat History: {chat_history}

#             User input: {input_text}
#             """)
#         ]
#     )
#     chain = SYSTEM_PROMPT | llm | parser

#     print("Xyz:", knowledge_base)
#     output = chain.invoke({
#         "system_prompt": system_prompt,
#         "chat_history": chat_history,
#         "input_text": input_text,
#         "knowledge_base": knowledge_base
#     })
#     return output


# # --- PAGE SETUP ---
# st.set_page_config(page_title="📚 RAG Chatbot", layout="centered")
# st.title("💬 RAG Chatbot")
# st.write("Chat with us, ask questions!")



# st.success("Ready for QnA!")


# # --- Chat UI ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # if "chain" in st.session_state:
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).markdown(msg["content"])

# user_input = st.chat_input("Ask me something from the document...")

# if user_input:
#     st.chat_message("user").markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     with st.spinner("Thinking..."):
#         docs = ""
#         try: 
#             try:
#                 vectorstore = Chroma(
#                     persist_directory=VECTORSTORE_DIR,
#                     embedding_function=embeddings,
#                     collection_name="all_documents"
#                 )

#                 relevant_docs_with_sources = vectorstore.similarity_search(
#                     user_input,
#                     k=2
#                 )
                
#                 for res in relevant_docs_with_sources:
#                     print(f"* {res.page_content} [{res.metadata}]")
#                     docs += res.page_content + "\n\n"

#             except Exception as e: 
#                 print(f"Error loading vector store: {e}")

#             result = llm_handler(user_input, knowledge_base=docs)
#         except Exception as e:
#             result = f"error: {e}"
#             st.error(f"Error processing your request: {e}")

#     st.chat_message("assistant").markdown(result)
#     st.session_state.messages.append({"role": "assistant", "content": result})


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import tempfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import requests
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Import our unified ingestion manager
try:
    from unified_ingestion import UnifiedIngestionManager
    INGESTION_AVAILABLE = True
except ImportError:
    INGESTION_AVAILABLE = False
    st.warning("Unified ingestion manager not found. Manual ingestion required.")

load_dotenv()

st.set_page_config(page_title="📚 RAG Chatbot", layout="centered")

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize embeddings and LLM - cached for performance"""
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY")
    )

    llm = AzureChatOpenAI(
        model=os.getenv("COMPLETION_MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        temperature=0
    )
    
    return embeddings, llm

embeddings, llm = initialize_components()
VECTORSTORE_DIR = "vectorstore"

@st.cache_resource
def get_vectorstore():
    """Initialize vectorstore - cached for performance"""
    try:
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name="all_documents"
        )
        # Test if vectorstore has data
        sample = vectorstore.get(limit=1)
        if len(sample['ids']) == 0:
            return None
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def check_and_run_ingestion():
    """Check if ingestion is needed and run if necessary"""
    if not INGESTION_AVAILABLE:
        return False
    
    try:
        manager = UnifiedIngestionManager(vectorstore_dir=VECTORSTORE_DIR)
        status = manager.get_ingestion_status()
        
        if not status['is_populated']:
            st.info("🔄 Vectorstore is empty. Running initial data ingestion...")
            with st.spinner("Ingesting data... This may take a few minutes."):
                success = manager.ingest_all_data()
                if success:
                    st.success("✅ Data ingestion completed successfully!")
                    st.rerun()  # Restart the app to load the new data
                else:
                    st.error("❌ Data ingestion failed!")
                return success
        elif status['needs_update']:
            if st.button("🔄 Update Data", help="New or changed files detected"):
                with st.spinner("Updating vectorstore..."):
                    success = manager.ingest_all_data()
                    if success:
                        st.success("✅ Data updated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Data update failed!")
        
        return True
    except Exception as e:
        st.error(f"Error with ingestion manager: {e}")
        return False

def chat_memory_setup(llm):
    """Set up conversation memory"""
    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
    )
    formatted_messages = []
    for msg in st.session_state.messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                formatted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_messages.append(AIMessage(content=content))
            elif role == "system":
                formatted_messages.append(SystemMessage(content=content))

    memory.chat_memory.messages = formatted_messages
    return memory

def llm_handler(input_text, knowledge_base, vectorstore):
    """Handle LLM interaction with knowledge base"""
    system_prompt = """You are an AI assistant designed to answer questions based on the provided documents. 
    Always stick to the provided documents and provide users with accurate information.
    If you don't know the answer based on the provided context, say "I don't know" instead of making up an answer.
    When possible, mention which document or source your answer comes from."""

    chat_history = chat_memory_setup(llm=llm)
    parser = StrOutputParser()
    
    SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
        ("human", """
        {system_prompt}
        
        Relevant Knowledge Base: {knowledge_base}
        
        Chat History: {chat_history}
        
        User input: {input_text}
        """)
    ])
    
    chain = SYSTEM_PROMPT | llm | parser
    
    output = chain.invoke({
        "system_prompt": system_prompt,
        "chat_history": chat_history,
        "input_text": input_text,
        "knowledge_base": knowledge_base
    })
    return output

def display_ingestion_status():
    """Display current ingestion status in sidebar"""
    if not INGESTION_AVAILABLE:
        return
    
    try:
        manager = UnifiedIngestionManager(vectorstore_dir=VECTORSTORE_DIR)
        status = manager.get_ingestion_status()
        
        with st.sidebar:
            st.subheader("📊 Data Status")
            
            if status['is_populated']:
                st.success("✅ Vectorstore populated")
                st.write(f"**Total chunks:** {status['total_chunks']}")
                st.write(f"**Files processed:** {len(status['processed_files'])}")
                st.write(f"**Web pages:** {status['processed_urls_count']}")
                
                if status['last_ingestion']:
                    from datetime import datetime
                    ingestion_date = datetime.fromisoformat(status['last_ingestion'])
                    st.write(f"**Last updated:** {ingestion_date.strftime('%Y-%m-%d %H:%M')}")
                
                if status['needs_update']:
                    st.warning("⚠️ Data needs updating")
                    if st.button("🔄 Update Now"):
                        with st.spinner("Updating..."):
                            success = manager.ingest_all_data()
                            if success:
                                st.success("Updated!")
                                st.rerun()
                            else:
                                st.error("Update failed!")
            else:
                st.error("❌ No data ingested")
                if st.button("🚀 Ingest Data"):
                    with st.spinner("Ingesting..."):
                        success = manager.ingest_all_data()
                        if success:
                            st.success("Ingested!")
                            st.rerun()
                        else:
                            st.error("Ingestion failed!")
    
    except Exception as e:
        st.sidebar.error(f"Status check failed: {e}")

# --- PAGE SETUP ---

st.title("💬 RAG Chatbot")
st.write("Chat with us, ask questions!")

# Check ingestion status and run if needed
ingestion_ok = check_and_run_ingestion()

# Get vectorstore
vectorstore = get_vectorstore()

# Display status in sidebar
display_ingestion_status()

# Main app logic
if vectorstore is None:
    st.error("⚠️ Vectorstore not available. Please run data ingestion first.")
    st.info("Make sure you have documents in the 'assets' folder and/or web scraping is configured.")
    st.stop()

st.success("Ready for Q&A!")

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me something from the documents...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        docs = ""
        try:
            # Retrieve relevant documents
            relevant_docs_with_sources = vectorstore.similarity_search(
                user_input,
                k=3  # Increased from 2 for better context
            )
            
            # Build knowledge base with source information
            sources = set()
            for res in relevant_docs_with_sources:
                docs += f"Source: {res.metadata.get('source', 'Unknown')}\n"
                docs += f"Content: {res.page_content}\n\n"
                
                # Track sources for citation
                source_info = res.metadata.get('source_file') or res.metadata.get('source', 'Unknown')
                sources.add(source_info)
            
            # Generate response
            result = llm_handler(user_input, knowledge_base=docs, vectorstore=vectorstore)
            
            # Add source information if available
            if sources:
                result += f"\n\n📚 *Sources: {', '.join(list(sources)[:3])}*"
            
        except Exception as e:
            result = f"❌ Error processing your request: {e}"
            st.error(f"Error: {e}")

    st.chat_message("assistant").markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})

# --- Footer ---
with st.sidebar:
    st.markdown("---")
    st.markdown("💡 **Tips:**")
    st.markdown("- Ask specific questions about your documents")
    st.markdown("- Reference specific topics or sections")
    st.markdown("- Use follow-up questions for clarification")
