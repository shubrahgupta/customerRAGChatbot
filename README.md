# Repository Documentation: RAG Chatbot with Unified Ingestion

This repository contains a **Retrieval-Augmented Generation (RAG) Chatbot** built using Streamlit, LangChain, and Azure OpenAI. The chatbot allows users to query documents and web pages, leveraging a vectorstore for efficient similarity-based retrieval.

---

## **Table of Contents**
1. Features
2. [Setup Instructions](#setup-instructions)
3. [Usage Instructions](#usage-instructions)
4. [Folder Structure](#folder-structure)
5. [Key Components](#key-components)
6. Troubleshooting

---

## **Features**
- Unified ingestion of documents and web pages into a vectorstore.
- Streamlit-based chatbot interface for querying ingested data.
- Automatic detection of missing or outdated vectorstore data.
- Support for PDF and Word documents, as well as web scraping.
- Integration with Azure OpenAI for embeddings and LLM responses.
- Metadata tracking for document sources.

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.11
- Azure OpenAI account with API keys
- Required Python libraries (see requirements.txt)

### **2. Clone the Repository**
```bash
git clone https://github.com/your-repo-name/rag-chatbot.git
cd rag-chatbot
```

### **3. Install Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**
Create a .env file in the root directory and add the following variables:
```plaintext
EMBEDDING_MODEL=text-embedding-ada-002
COMPLETION_MODEL=gpt-4
API_VERSION=2024-02-15-preview
AZURE_ENDPOINT=https://<your-azure-endpoint>.openai.azure.com/
API_KEY=<your-azure-api-key>
```
[Modified for streamlit use by replacing with st.secrets: [https://docs.streamlit.io/develop/api-reference/connections/st.secrets]([url](https://docs.streamlit.io/develop/api-reference/connections/st.secrets))]

### **5. Prepare Data**
- Place your documents (PDFs, Word files) in the assets folder.
- Configure the base URL for web scraping in web_ingestion.py[Not in use now, replaced by unified_ingestion.py file]

 (default: `https://www.angelone.in/support`).

---

## **Usage Instructions**

### **1. Run the Chatbot**
Start the Streamlit app:
```bash
streamlit run app.py
```

### **2. Interact with the Chatbot**
- Open the app in your browser (default: `http://localhost:8501`).
- Ask questions about the ingested documents or web pages.
- The chatbot will retrieve relevant information and provide accurate responses.

### **3. Ingestion Management**
- The app automatically checks if the vectorstore is populated.
- If the vectorstore is empty or outdated, you can trigger ingestion from the sidebar.

---

## **Folder Structure**
```
rag-chatbot/
│
├── app.py                     # Main Streamlit app
├── unified_ingestion.py       # Unified ingestion manager for documents and web pages
├── doc_ingestion.py           # Handles document ingestion (PDFs, Word files),  Not in use now, replaced by unified_ingestion.py file.
├── web_ingestion.py           # Handles web scraping and ingestion, Not in use now, replaced by unified_ingestion.py file.
├── assets/                    # Folder for storing documents
├── vectorstore/               # Folder for storing vectorstore data
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not included in repo)
└── README.md                  # Repository documentation
```

---

## **Key Components**

### **1. 

app.py

**
- Main Streamlit app for interacting with the chatbot.
- Handles user input, vectorstore retrieval, and LLM responses.

### **2. 

unified_ingestion.py

**
- Manages ingestion of documents and web pages into a unified vectorstore.
- Tracks metadata for ingestion status and updates.

### **3. 

doc_ingestion.py

**
- Processes documents (PDFs, Word files) and splits them into chunks for embedding. Not in use now, replaced by unified_ingestion.py file.

### **4. 

web_ingestion.py

**
- Crawls web pages and extracts clean text for embedding. Not in use now, replaced by unified_ingestion.py file.

### **5. 

vectorstore

**
- Stores the vectorstore data for similarity-based retrieval.

---

## **Troubleshooting**

### **1. Missing Vectorstore**
- Ensure the vectorstore folder exists and is writable.
- Run the ingestion process from the sidebar or manually trigger it.

### **2. Environment Variable Issues**
- Verify that the .env file is correctly configured.
- Ensure the Azure OpenAI API key and endpoint are valid.

### **3. Dependency Errors**
- Ensure all dependencies are installed using `pip install -r requirements.txt`.
- Use a virtual environment to avoid conflicts.

### **4. Web Scraping Issues**
- Check the base URL in web_ingestion.py
- Ensure the target website allows scraping and is accessible.

---
