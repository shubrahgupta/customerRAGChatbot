# unified_ingestion.py
import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path
from datetime import datetime

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class UnifiedIngestionManager:
    def __init__(self, vectorstore_dir="vectorstore"):
        self.vectorstore_dir = vectorstore_dir
        self.collection_name = "all_documents"
        self.metadata_file = os.path.join(vectorstore_dir, "ingestion_metadata.json")
        self.doc_dir = "assets"
        self.base_url = "https://www.angelone.in/support"
        self.domain = "www.angelone.in"
        self.max_pages = 15
        
        self.embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("API_KEY")
        )
        
        # Ensure directories exist
        os.makedirs(vectorstore_dir, exist_ok=True)
        os.makedirs(self.doc_dir, exist_ok=True)
    
    def _load_metadata(self):
        """Load ingestion metadata to track what's been processed"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "last_ingestion": None,
            "processed_files": {},
            "processed_urls": [],
            "total_chunks": 0,
            "version": "1.0"
        }
    
    def _save_metadata(self, metadata):
        """Save ingestion metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_file_hash(self, file_path):
        """Get hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    def _is_vectorstore_populated(self):
        """Check if vectorstore already exists and has data"""
        try:
            vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            # Try to get a sample to see if data exists
            sample = vectorstore.get(limit=1)
            return len(sample['ids']) > 0
        except Exception:
            return False
    
    def _needs_reingestion(self):
        """Check if re-ingestion is needed based on file changes"""
        metadata = self._load_metadata()
        
        # Check if any files have changed
        current_files = {}
        if os.path.exists(self.doc_dir):
            for file in os.listdir(self.doc_dir):
                if file.endswith(('.pdf', '.docx')):
                    file_path = os.path.join(self.doc_dir, file)
                    current_files[file] = {
                        'hash': self._get_file_hash(file_path),
                        'modified': os.path.getmtime(file_path)
                    }
        
        # Compare with stored metadata
        stored_files = metadata.get('processed_files', {})
        
        for file, info in current_files.items():
            if file not in stored_files or stored_files[file]['hash'] != info['hash']:
                return True
        
        # Check if files were deleted
        for file in stored_files:
            if file not in current_files:
                return True
        
        return False
    
    def _crawl_website(self):
        """Crawl website and collect documents"""
        visited = set()
        docs = []
        
        def is_valid(url):
            parsed = urlparse(url)
            return parsed.netloc == self.domain and parsed.path.startswith("/support")
        
        def extract_clean_text(html):
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        
        def crawl_recursive(url):
            if url in visited or not is_valid(url) or len(visited) >= self.max_pages:
                return
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    return
                
                text = extract_clean_text(response.text)
                if len(text.strip()) > 100:
                    docs.append(Document(
                        page_content=text, 
                        metadata={
                            "source": url,
                            "source_type": "web",
                            "domain": self.domain,
                            "ingestion_date": datetime.now().isoformat()
                        }
                    ))
                
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(url, link["href"])
                    crawl_recursive(full_url)
                
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        print(f"🌐 Crawling website from {self.base_url}...")
        crawl_recursive(self.base_url)
        print(f"✅ Crawled {len(visited)} pages, collected {len(docs)} documents")
        
        return docs, list(visited)
    
    def _load_documents(self):
        """Load documents from the assets directory"""
        all_docs = []
        processed_files = {}
        
        if not os.path.exists(self.doc_dir):
            print(f"Document directory {self.doc_dir} does not exist")
            return all_docs, processed_files
        
        print(f"📄 Loading documents from {self.doc_dir}...")
        
        for file in os.listdir(self.doc_dir):
            if not file.endswith(('.pdf', '.docx')):
                continue
                
            file_path = os.path.join(self.doc_dir, file)
            
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:
                    continue
                
                docs = loader.load()
                document_name = os.path.splitext(file)[0]
                
                # Enhanced metadata for each document
                for doc in docs:
                    doc.metadata.update({
                        "source_file": file,
                        "document_name": document_name,
                        "file_type": file.split('.')[-1],
                        "source_type": "document",
                        "doc_id": f"{document_name}_{hash(doc.page_content) % 10000}",
                        "ingestion_date": datetime.now().isoformat()
                    })
                
                all_docs.extend(docs)
                
                # Track processed file
                processed_files[file] = {
                    'hash': self._get_file_hash(file_path),
                    'modified': os.path.getmtime(file_path),
                    'chunks': len(docs)
                }
                
                print(f"Loaded {len(docs)} chunks from {file}")
                
            except Exception as e:
                print(f" Error loading {file}: {e}")
        
        return all_docs, processed_files
    
    def ingest_all_data(self, force=False):
        """Main ingestion method that handles both documents and web content"""
        
        # Check if ingestion is needed
        if not force and self._is_vectorstore_populated() and not self._needs_reingestion():
            print(" Vectorstore already populated and no changes detected. Skipping ingestion.")
            print("Use force=True to re-ingest anyway.")
            return True
        
        print("🚀 Starting unified data ingestion...")
        
        all_documents = []
        
        # Load documents from files
        doc_documents, processed_files = self._load_documents()
        all_documents.extend(doc_documents)
        
        # Crawl website
        web_documents, processed_urls = self._crawl_website()
        all_documents.extend(web_documents)
        
        if not all_documents:
            print("⚠️ No documents found to ingest")
            return False
        
        # Split documents
        print("🔧 Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(all_documents)
        
        # Create/update vectorstore
        print("💾 Creating vectorstore...")
        try:
            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=self.embeddings,
                persist_directory=self.vectorstore_dir,
                collection_name=self.collection_name
            )
            
            # Save metadata
            metadata = {
                "last_ingestion": datetime.now().isoformat(),
                "processed_files": processed_files,
                "processed_urls": processed_urls,
                "total_chunks": len(all_splits),
                "total_documents": len(all_documents),
                "version": "1.0"
            }
            self._save_metadata(metadata)
            
            print(f"✅ Successfully ingested {len(all_splits)} chunks from {len(all_documents)} documents")
            print(f"   - Document files: {len(doc_documents)} documents")
            print(f"   - Web pages: {len(web_documents)} pages")
            
            return True
            
        except Exception as e:
            print(f" Error creating vectorstore: {e}")
            return False
    
    def get_ingestion_status(self):
        """Get current ingestion status"""
        metadata = self._load_metadata()
        is_populated = self._is_vectorstore_populated()
        needs_update = self._needs_reingestion()
        
        return {
            "is_populated": is_populated,
            "needs_update": needs_update,
            "last_ingestion": metadata.get("last_ingestion"),
            "total_chunks": metadata.get("total_chunks", 0),
            "processed_files": list(metadata.get("processed_files", {}).keys()),
            "processed_urls_count": len(metadata.get("processed_urls", []))
        }

# CLI interface for running ingestion
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Data Ingestion Manager")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if data exists")
    parser.add_argument("--status", action="store_true", help="Show ingestion status")
    parser.add_argument("--vectorstore-dir", default="vectorstore", help="Vectorstore directory")
    
    args = parser.parse_args()
    
    manager = UnifiedIngestionManager(vectorstore_dir=args.vectorstore_dir)
    
    if args.status:
        status = manager.get_ingestion_status()
        print("\n📊 Ingestion Status:")
        print(f"   Vectorstore populated: {status['is_populated']}")
        print(f"   Needs update: {status['needs_update']}")
        print(f"   Last ingestion: {status['last_ingestion']}")
        print(f"   Total chunks: {status['total_chunks']}")
        print(f"   Processed files: {len(status['processed_files'])}")
        print(f"   Processed URLs: {status['processed_urls_count']}")
    else:
        success = manager.ingest_all_data(force=args.force)
        if success:
            print("\n Ingestion completed successfully!")
        else:
            print("\n Ingestion failed!")