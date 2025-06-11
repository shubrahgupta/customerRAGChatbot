import tempfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os


load_dotenv()



doc_dir = "assets"
all_docs = []
VECTORSTORE_DIR = "vectorstore"

# try:

#     for file in os.listdir(doc_dir):
#         file_path = os.path.join(doc_dir, file)
#         if file.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif file.endswith(".docx"):
#             loader = UnstructuredWordDocumentLoader(file_path)
#         else:
#             continue

#         document_name = os.path.splitext(file)[0]  # Use the filename (without extension) as the collection name
#         collection_dir = os.path.join(VECTORSTORE_DIR, document_name)


#         all_docs.extend(loader.load())


#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
#         pdf_doc_splits = text_splitter.split_documents(all_docs)

#         embeddings = AzureOpenAIEmbeddings(
#             model=os.getenv("EMBEDDING_MODEL"),
#             api_version=os.getenv("API_VERSION"),
#             azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#             api_key=os.getenv("API_KEY")
#         )
#         vectorstore = Chroma.from_documents(
#             documents=pdf_doc_splits,
#             embedding=embeddings,
#             persist_directory=VECTORSTORE_DIR,
#         )
#         print(f"✅ Embedded {len(pdf_doc_splits)} chunks into {VECTORSTORE_DIR}")

# except Exception as e:
#     print(str(e))


def create_single_collection_with_metadata():
    """Create a single collection but with rich metadata for filtering"""
    
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY")
    )
    
    all_docs = []
    
    try:
        for file in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue

            docs = loader.load()
            document_name = os.path.splitext(file)[0]
            
            # Enhance metadata for each document
            for doc in docs:
                doc.metadata.update({
                    "source_file": file,
                    "document_name": document_name,
                    "file_type": file.split('.')[-1],
                    "doc_id": f"{document_name}_{hash(doc.page_content) % 10000}"
                })
            
            all_docs.extend(docs)

        # Split all documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            add_start_index=True
        )
        all_doc_splits = text_splitter.split_documents(all_docs)

        # Create single vectorstore with all documents
        vectorstore = Chroma.from_documents(
            documents=all_doc_splits,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR,
            collection_name="all_documents"
        )
        
        print(f"✅ Created single collection with {len(all_doc_splits)} chunks from {len(set(doc.metadata['document_name'] for doc in all_doc_splits))} documents")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    

vectorstore = create_single_collection_with_metadata()
