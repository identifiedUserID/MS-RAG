# 🔍 MS-RAG: Multi-Stage Retrieval-Augmented Generation with Gemini & PostgreSQL

A fully modular Retrieval-Augmented Generation (RAG) system designed to ingest unstructured documents (e.g., PDFs), summarize and index them with embeddings, and generate refined, context-rich answers to user queries.

> ⚡ Built using PostgreSQL with `pgvector`, and powered by Google’s **Gemini** models (2.5 Flash & Pro).

---

## 🧠 Project Scope

This project implements a **multi-phase RAG pipeline** that does the following:

1. **Ingests documents** (PDF or `.txt`) and extracts their full text.
2. **Summarizes** the content using Gemini 2.5 Flash.
3. **Embeds summaries and fine-grained text chunks** using Google Embedding API (`text-embedding-004`).
4. **Stores everything in a PostgreSQL vector database** for semantic retrieval.
5. **Responds to user questions** using a refined multi-stage answer generation process:
   - Coarse-grained summary retrieval
   - LLM-powered query refinement (Gemini 2.0 Flash)
   - Fine-grained chunk retrieval
   - Final answer generation (Gemini 2.5 Pro)

---

## 🚀 Key Features

- **End-to-End RAG pipeline**: Document ingestion to final answer in a single run.
- **PostgreSQL + pgvector**: Scalable similarity search using vector embeddings.
- **Gemini LLM integration**:
  - Smart summarization
  - Dynamic query rewriting
  - Precise answer construction
- **Chunking + Metadata tracking** for traceability.
- **Structured LLM outputs** (JSON-formatted summaries with named entities and key events).
- **Ready for extension** to multi-document, multi-source ingestion.

---

## 💡 Use Cases

| Domain            | Use Case Example |
|------------------|------------------|
| Legal Research    | Upload court case PDFs and ask for precedents or rulings. |
| Academic Review   | Summarize and query scientific papers. |
| Corporate Docs    | Ingest contracts, SOPs, or technical specs. |
| Historical Analysis | Ask questions about chapters from historical archives. |
| Narrative Insights | Use on story chapters for character tracking or plot querying. |

---

## 🛠️ Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/yourname/rag-to-chatgpt.git
cd rag-to-chatgpt

```

### 2. Install Dependencies

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt

```

> `requirements.txt` should include:
> 
> -   `google-generativeai`
>     
> -   `psycopg2-binary`
>     
> -   `pypdf`
>     
> -   `python-dotenv`
>     

### 3. Setup PostgreSQL + pgvector

Install PostgreSQL and enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

```

Ensure your DB has two tables: `summaries` and `chunks` (automatically created via script).

### 4. Add Gemini API Key

Create a `.env` file in the root directory:

```
GEMINI_API_KEY="your_google_gemini_api_key"

```

----------

## 🧰 Models Used

**Purpose**

**Model Used**

Summarization

`gemini-2.5-flash`

Query Refinement

`gemini-2.0-flash`

Final Answer Generation

`gemini-2.5-pro`

Embeddings

`text-embedding-004`

----------

## 🧱 System Architecture

```mermaid
graph TD
  A[User Document (PDF/TXT)] --> B[Summarization (Gemini 2.5 Flash)]
  B --> C[Store Summary & Embedding in DB]
  A --> D[Chunk Text]
  D --> E[Store Chunks & Embeddings]
  F[User Query] --> G[Summary Embedding Search]
  G --> H[LLM Query Refinement (Gemini 2.0 Flash)]
  H --> I[Chunk Embedding Search]
  I --> J[Final Answer (Gemini 2.5 Pro)]

```

----------

## 🔍 Core Functions

| **Function**                                  | **Purpose**                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `configure_gemini()`                          | Load API key from `.env` and configure Google Generative AI SDK.                            |
| `setup_database()`                            | Initialize DB schema and ensure pgvector is enabled.                                        |
| `load_pdf(file_path)`                         | Extracts full text from PDF file.                                                           |
| `chunk_text(text)`                            | Splits text into overlapping paragraph chunks for fine-grained retrieval.                   |
| `get_embedding(text, task)`                   | Calls Gemini’s embedding API for text vectorization.                                        |
| `get_summary_from_llm(text)`                  | Uses Gemini 2.5 Flash to generate structured summaries in JSON (summary, entities, events). |
| `refine_query_with_llm(query, summaries)`     | Refines the query based on retrieved summaries using Gemini 2.0 Flash.                      |
| `generate_final_answer(query, chunks)`        | Constructs final response using Gemini 2.5 Pro.                                             |
| `process_and_store_document(path, source_id)` | Ingests document, stores both summary and chunks with embeddings.                           |
| `rag_pipeline(query)`                         | End-to-end RAG query pipeline that retrieves and responds intelligently.                    |

----------

## 🧪 Example Usage

```bash
# 1. Configure Gemini API and Database
configure_gemini()
setup_database()

# 2. Ingest a document
process_and_store_document("docs/chapter1.pdf", source_id="chapter_1")

# 3. Ask a question
response = rag_pipeline("What triggered the main character's transformation?")
print(response)

```

----------

## 📂 Folder Structure

```
rag-to-chatgpt/
│
├── rag_system.py          # Main pipeline implementation
├── dummy_novel.txt        # Example document
├── .env                   # API key for Gemini
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

```

----------

## 📌 Notes

-   Summarization input is capped (~200K chars).
    
-   Chunking uses paragraph splits with optional overlap.
    
-   Gemini output is parsed via JSON; malformed responses are handled gracefully.
    
-   PostgreSQL similarity search uses `embedding <=> %s` for cosine distance.
    

----------

## 🧱 Roadmap / To-Do

-   Add web UI (Streamlit / Flask)
    
-   Multi-document ingestion & cross-source querying
    
-   Add citation tracking (which chunks produced which parts of the answer)
    
-   Enable PDF output for responses
    

----------

## 🤝 Contributions

PRs and suggestions welcome! Please file issues for bugs, model upgrade requests, or integration ideas.

----------

## 📝 License

MIT License – see `LICENSE.md` for details.

```
