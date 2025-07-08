import os
import re
import json
import psycopg2
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()

# --- LLM and Embedding Model Configuration ---
# IMPORTANT NOTE: The model names below are based on the README.md.
# As of late 2024, 'gemini-2.0-flash' and 'gemini-2.5-...' may be speculative
# or unreleased. You may need to replace them with currently available models
# like 'gemini-1.5-flash-latest' or 'gemini-1.5-pro-latest' for the code to run.
SUMMARY_LLM_MODEL = 'gemini-2.5-flash'
REFINEMENT_LLM_MODEL = 'gemini-2.0-flash'
FINAL_ANSWER_LLM_MODEL = 'gemini-2.5-pro'
EMBEDDING_MODEL = 'models/text-embedding-004'

# --- Database Configuration ---
DB_NAME = "rag_db"
DB_USER = "postgres"
DB_PASSWORD = "your_postgres_password" # Replace with your password
DB_HOST = "localhost"
DB_PORT = "5432"


def configure_gemini():
    """Configures the Gemini API with the key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully.")

# --- DATABASE FUNCTIONS ---

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to database: {e}")
        raise

def setup_database():
    """Sets up the database tables and enables the vector extension."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Ensured 'vector' extension exists.")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id SERIAL PRIMARY KEY,
            source_id TEXT UNIQUE NOT NULL,
            source_type VARCHAR(50),
            summary_text TEXT,
            entities JSONB,
            key_events JSONB,
            embedding vector(768)
        );
        """)
        print("Table 'summaries' is ready.")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            source_id TEXT NOT NULL,
            chunk_text TEXT,
            metadata JSONB,
            embedding vector(768)
        );
        """)
        print("Table 'chunks' is ready.")

    conn.commit()
    conn.close()

# --- DATA INGESTION AND PROCESSING FUNCTIONS ---

def load_document(file_path: str) -> str:
    """Reads the text content from a PDF or TXT file."""
    if file_path.lower().endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Please use .pdf or .txt")

def chunk_text(text: str) -> list[str]:
    """Splits text into chunks. A simple paragraph splitter is used here."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p for p in paragraphs if p.strip()]

# --- LLM AND EMBEDDING FUNCTIONS ---

def get_embedding(text: str, task_type: str) -> list[float]:
    """Generates an embedding for the given text."""
    return genai.embed_content(model=EMBEDDING_MODEL,
                               content=text,
                               task_type=task_type)["embedding"]

def get_summary_from_llm(text: str) -> dict:
    """Generates a structured summary from the LLM."""
    model = genai.GenerativeModel(SUMMARY_LLM_MODEL)
    prompt = f"""
    You are a precision literary summarization engine. Your job is to process a document and return a detailed summary, key entities, and events.
    The summary must be 250 to 500 words and capture who did what, when, where, and why.
    Extract named entities (characters, locations, items, groups, abilities) and key events.
    You must output everything in the following structured JSON format:
    {{
      "summary_text": "...",
      "entities": {{
        "characters": [], "locations": [], "items": [], "groups": [], "abilities": []
      }},
      "key_events": [
        {{"title": "...", "description": "..."}}
      ]
    }}

    DOCUMENT TEXT (first 200k chars):
    ---
    {text[:200000]}
    ---
    """
    response = model.generate_content(prompt)
    try:
        json_str = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL).group(1)
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: Could not parse summary response as JSON. Trying to find JSON block.\nRaw Response:\n{response.text}")
        try:
            # A less strict fallback
            json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error: Fallback JSON parsing also failed: {e}")
            return None


def refine_query_with_llm(query: str, summaries: list[str]) -> dict:
    """Uses LLM to refine a query based on retrieved summaries."""
    model = genai.GenerativeModel(REFINEMENT_LLM_MODEL)
    context = "\n---\n".join(summaries)
    prompt = f"""
    Based on the user's query and these retrieved chapter summaries, your task is to identify the most relevant chapter(s) and refine the user's query to be more specific.

    USER QUERY: "{query}"

    RETRIEVED SUMMARIES:
    ---
    {context}
    ---

    Output a JSON object with two keys:
    1. "relevant_sources": A list of the most relevant source_id strings from the summaries.
    2. "refined_question": A more specific and detailed version of the user's query.

    JSON_OUTPUT:
    """
    response = model.generate_content(prompt)
    try:
        json_str = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL).group(1)
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: Could not parse refinement response. Trying to find JSON block.\nRaw Response:\n{response.text}")
        try:
            json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error: Fallback JSON parsing also failed for refinement: {e}")
            return None

def generate_final_answer(query: str, chunks: list[str]) -> str:
    """Generates the final answer from the LLM based on fine-grained context."""
    model = genai.GenerativeModel(FINAL_ANSWER_LLM_MODEL)
    context = "\n---\n".join(chunks)
    prompt = f"""
    You are an expert assistant. Your task is to answer the user's question using ONLY the provided context.
    If the context does not contain the answer, state that clearly.
    Be concise and directly reference information from the context.

    USER QUESTION: "{query}"

    CONTEXT:
    ---
    {context}
    ---

    ANSWER:
    """
    response = model.generate_content(prompt)
    return response.text

# --- CORE WORKFLOW FUNCTIONS ---

def process_and_store_document(file_path: str, source_id: str, source_type: str = "document"):
    """Main ingestion pipeline for a single document."""
    conn = get_db_connection()
    print(f"\n--- Processing Document: {source_id} ---")
    
    # 1. Load and Summarize (Tier 1)
    full_text = load_document(file_path)
    summary_data = get_summary_from_llm(full_text)
    
    if summary_data:
        summary_embedding = get_embedding(summary_data['summary_text'], "retrieval_document")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO summaries (source_id, source_type, summary_text, entities, key_events, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    entities = EXCLUDED.entities,
                    key_events = EXCLUDED.key_events,
                    embedding = EXCLUDED.embedding;
            """, (source_id, source_type, summary_data['summary_text'], json.dumps(summary_data['entities']), json.dumps(summary_data['key_events']), summary_embedding))
        conn.commit()
        print(f"Stored summary for {source_id}.")
    
    # 2. Chunk and Store (Tier 2)
    chunks = chunk_text(full_text)
    print(f"Document split into {len(chunks)} chunks.")
    
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = %s;", (source_id,))
        for i, chunk in enumerate(chunks):
            chunk_embedding = get_embedding(chunk, "retrieval_document")
            metadata = {"source_id": source_id, "chunk_index": i}
            cur.execute(
                "INSERT INTO chunks (source_id, chunk_text, metadata, embedding) VALUES (%s, %s, %s, %s)",
                (source_id, chunk, json.dumps(metadata), chunk_embedding)
            )
    
    conn.commit()
    conn.close()
    print(f"Stored {len(chunks)} chunks for {source_id}.")

def rag_pipeline(query: str):
    """Full RAG pipeline from user query to final answer."""
    conn = get_db_connection()
    print("\n--- Starting RAG Pipeline ---")
    print(f"User Query: {query}")
    
    # 1. Coarse-Grained Search (Summaries)
    query_embedding = get_embedding(query, "retrieval_query")
    with conn.cursor() as cur:
        cur.execute("SELECT source_id, summary_text FROM summaries ORDER BY embedding <=> %s LIMIT 5;", (str(query_embedding),))
        retrieved_summaries = [{"source_id": r[0], "summary_text": r[1]} for r in cur.fetchall()]
        
    if not retrieved_summaries:
        return "I could not find any relevant documents to answer your question."
    print(f"\n[Phase 1] Retrieved {len(retrieved_summaries)} relevant summaries.")
    
    # 2. LLM-Powered Filtering and Query Refinement
    summary_texts = [s['summary_text'] for s in retrieved_summaries]
    refinement_result = refine_query_with_llm(query, summary_texts)
    
    if not refinement_result or 'refined_question' not in refinement_result:
        return "I had trouble refining the query. Please try rephrasing."

    refined_question = refinement_result['refined_question']
    relevant_sources = refinement_result.get('relevant_sources', [s['source_id'] for s in retrieved_summaries])
    print(f"\n[Phase 2] Refined Question: {refined_question}")
    print(f"[Phase 2] Focusing search on sources: {relevant_sources}")
    
    # 3. Fine-Grained Search (Chunks)
    refined_query_embedding = get_embedding(refined_question, "retrieval_query")
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_text FROM chunks WHERE source_id = ANY(%s) ORDER BY embedding <=> %s LIMIT 5;", (relevant_sources, str(refined_query_embedding)))
        retrieved_chunks = [r[0] for r in cur.fetchall()]

    if not retrieved_chunks:
        return "I found relevant documents but could not retrieve specific details. Please try another question."
    print(f"\n[Phase 3] Retrieved {len(retrieved_chunks)} relevant chunks for final answer.")

    # 4. Final Answer Synthesis
    final_answer = generate_final_answer(refined_question, retrieved_chunks)
    
    conn.close()
    print("\n--- RAG Pipeline Complete ---")
    return final_answer

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    configure_gemini()
    setup_database()

    # Ingest the document into our RAG system
    # Make sure 'dummy_novel.txt' is in the same directory
    process_and_store_document("dummy_novel.txt", source_id="chapter_17-18")
    
    # Ask a question using the RAG pipeline
    user_question = "What happened right after Nolan touched the Ember Core?"
    answer = rag_pipeline(user_question)
    
    print(f"\n\nFINAL ANSWER:\n{'-'*20}\n{answer}")