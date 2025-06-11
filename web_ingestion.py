import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os


load_dotenv()

# CONFIG
BASE_URL = "https://www.angelone.in/support"
DOMAIN = "www.angelone.in"
VECTORSTORE_DIR = "vectorstore"
MAX_PAGES = 15  # safety limit

visited = set()
docs = []

def is_valid(url):
    parsed = urlparse(url)
    return parsed.netloc == DOMAIN and parsed.path.startswith("/support")

def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def crawl_and_collect(url):
    if url in visited or not is_valid(url) or len(visited) >= MAX_PAGES:
        return
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return

        text = extract_clean_text(response.text)
        if len(text.strip()) > 100:
            docs.append(Document(page_content=text, metadata={"source": url}))

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            crawl_and_collect(full_url)

        time.sleep(0.5)

    except Exception as e:
        print(f"Error crawling {url}: {e}")


try: 
    # START
    print(f"🌐 Crawling AngelOne support site from {BASE_URL} ...")
    crawl_and_collect(BASE_URL)
    print(f"✅ Crawled {len(visited)} pages, collected {len(docs)} documents.")

    # SPLIT & EMBED
    print("🔧 Splitting and embedding...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_version=os.getenv("API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("API_KEY")
    )
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"✅ Embedded {len(splits)} chunks into vectorstore: {VECTORSTORE_DIR}")

except Exception as e: 
    print(str(e))
