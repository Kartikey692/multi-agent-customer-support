# TechSolutions Support Agent Orchestrator

## Overview
This project is a multi-agent AI system designed for **TechSolutions**, a B2B SaaS company specializing in cloud infrastructure management software. It provides an intelligent customer support orchestration layer capable of routing incoming customer inquiries to specialized agents, fetching context dynamically, and synthesizing accurate responses. 

The system leverages **Mistral AI** as its core Large Language Model (LLM) and uses **ChromaDB** for Retrieval-Augmented Generation (RAG) to ground answers in real product catalogs, technical documentation, and FAQs.

## Tech Stack
- **Language**: Python 3.10
- **Framework**: FastAPI (served via Uvicorn)
- **AI/LLM**: Mistral AI (`mistralai` Python client), LangChain (for text splitting and document processing)
- **Vector Database**: ChromaDB (persistent local storage for embeddings)
- **Embeddings**: SentenceTransformers
- **Containerization**: Docker

## Architecture
The system employs an orchestrator-agent pattern, comprising the following specialized components:

### 1. Agent Orchestrator (`main.py` & `agent_implementations.py`)
Acts as the central nervous system. It receives incoming customer queries, coordinates with the Router Agent for classification, delegates tasks to the appropriate specialized agent, and maintains conversation history.

### 2. Router Agent
Responsible for analyzing and classifying user queries into specific categories: `Product`, `Technical`, `Billing`, `Account`, or `General`. It can also detect multi-part queries, split them, and route each part independently.

### 3. Product Specialist Agent
Handles inquiries related to products, pricing, features, and plans. It queries the `products` ChromaDB collection to retrieve relevant catalog information and FAQs to construct accurate, grounded responses.

### 4. Technical Support Agent
Focuses on troubleshooting and technical errors. It retrieves technical documentation from the `technical` ChromaDB collection and can simulate diagnostic API calls to resolve user issues.

### 5. Order/Billing Agent
Manages questions regarding invoices, payments, and order statuses. It interfaces with internal mock APIs (`/api/orders/{order_id}` and `/api/accounts/{account_id}`) to fetch real-time financial data.

### 6. Account Management Agent
Assists users with account configuration, user seat limits, and subscription tier upgrades/downgrades. It verifies permissions and fetches live account data from the mock APIs to guide customers.

## Complete Workflow

1. **Data Ingestion (`setup.py` & `data_utils.py`)**
   - When initialized, the `DataManager` loads source data from the `data/` directory (JSON catalogs, Markdown docs, FAQs).
   - `LangChain`'s `RecursiveCharacterTextSplitter` chunks these documents.
   - The chunks are embedded and stored persistently in **ChromaDB** (`chroma_db/`).

2. **Query Processing (`main.py`)**
   - The client sends a POST request to `/api/query` with a user prompt.
   - The **Orchestrator** sends the prompt to the **Router Agent** for classification.
   - Based on the classification, the prompt is forwarded to the designated **Specialist Agent**.

3. **Context Retrieval & Response Generation**
   - **RAG Agents** (Product & Technical) search ChromaDB for similar documents based on the user's prompt.
   - **API Agents** (Billing & Account) extract IDs (like `ORD-123` or `ACC-1111`) from the prompt and query the respective REST APIs.
   - The retrieved context and the user's prompt are wrapped in a system prompt and sent to the **Mistral LLM**.
   - The generated response is returned to the client and saved in the conversation history.

## Project Structure
```text
.
├── Dockerfile                  # Container definition for the FastAPI app
├── agent_implementations.py    # Core logic for Orchestrator and all specific Agents
├── chroma_db/                  # Persistent vector database storage (generated)
├── data/                       # Source data files
│   ├── customer_conversations.jsonl
│   ├── faq.json
│   ├── product_catalog.json
│   └── tech_documentation.md
├── data_utils.py               # DataManager logic for chunking and ChromaDB ingestion
├── main.py                     # FastAPI application and mock REST endpoints
├── requirements.txt            # Python dependencies
└── setup.py                    # Script to bootstrap the vector database
```

## How to Run

### Prerequisites
1. Python 3.10+
2. A Mistral API Key.

### Local Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your environment variables in a `.env` file at the root:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   MISTRAL_MODEL_NAME=mistral-small-latest
   ```
3. Initialize the Vector Database:
   ```bash
   python setup.py
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Docker Setup
To run the service via Docker:
```bash
docker build -t techsolutions-orchestrator .
docker run -p 8000:8000 -e MISTRAL_API_KEY=your_api_key techsolutions-orchestrator
```

### Testing the API
You can interact with the system via its `/api/query` endpoint using `curl`:
```bash
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "I am getting error e1234, how do I fix it?", "conversation_id": "session-1"}'
```
