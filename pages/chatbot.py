# Groq API key
import streamlit as st
import ast
import os
import time
import json
import unicodedata
import re
from io import StringIO
import contextlib
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import requests
from groq import Groq
import config

# Initialize Streamlit
st.set_page_config(
    page_title="LEOparts Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid='stSidebarNav'] {display: none;}
    
    /* Sidebar white background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
    }
    /* Button styling with blue theme */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #4a90e2 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part"):
        st.switch_page("pages/extraction_attributs.py")

# Add navigation button at the top
if st.sidebar.button("← Back to Main App", use_container_width=True):
    st.switch_page("app.py")

# Add blue header band at the top
st.markdown(
    """
    <style>
    .header-band {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a90e2 100%);
        color: white;
        padding: 0.7rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
    }
    .header-band h1 {
        font-size: 2.2em;
        margin: 0;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
    <div class="header-band">
        <h1>LEONI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Configuration ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
        raise ValueError("One or more secrets not found.")
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    st.stop()

# --- Model & DB Config ---
MARKDOWN_TABLE_NAME = "markdown_chunks"
ATTRIBUTE_TABLE_NAME = "Leoni_attributes"          # <<< VERIFY
RPC_FUNCTION_NAME = "match_markdown_chunks"     # <<< VERIFY
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = config.EMBEDDING_DIMENSIONS

# ░░░  MODEL SWITCH  ░░░
GROQ_MODEL_FOR_SQL = "qwen-qwq-32b"              ### <-- CHANGED
GROQ_MODEL_FOR_ANSWER = "qwen-qwq-32b"              ### <-- CHANGED
# st.write(f"Using Groq Model for SQL: {GROQ_MODEL_FOR_SQL}")
# st.write(f"Using Groq Model for Answer: {GROQ_MODEL_FOR_ANSWER}")

# --- Search Parameters ---
VECTOR_SIMILARITY_THRESHOLD = 0.4
VECTOR_MATCH_COUNT = 3

# --- Initialize Clients ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    # st.success("Supabase client initialized.")
except Exception as e:
    st.error(f"Error initializing Supabase client: {e}")
    st.stop()

# --- Embedding API Configuration ---
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://hbaananou-embedder-model.hf.space/embed")
USE_API_EMBEDDINGS = os.getenv("USE_API_EMBEDDINGS", "true").lower() == "true"

try:
    if USE_API_EMBEDDINGS:
        # Test API embeddings
        test_response = requests.post(
            EMBEDDING_API_URL,
            headers={"Content-Type": "application/json"},
            json={"texts": ["test"]},
            timeout=config.EMBEDDING_TIMEOUT
        )
        test_response.raise_for_status()
        test_result = test_response.json()
        
        # Extract test embedding
        if "embeddings" in test_result:
            test_emb = test_result["embeddings"][0]
        elif "vectors" in test_result:
            test_emb = test_result["vectors"][0]
        elif isinstance(test_result, list):
            test_emb = test_result[0]
        else:
            test_emb = test_result.get("data", test_result.get("result", test_result))[0]
        
        if len(test_emb) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(test_emb)}")
        
        st_model = None  # We'll use API calls instead
        # st.success(f"API embeddings initialized successfully.")
    else:
        # Fallback to local model
        st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        test_emb = st_model.encode("test")
        if len(test_emb) != EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimension mismatch")
        # st.success(f"Sentence Transformer model ({EMBEDDING_MODEL_NAME}) loaded.")
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")
    st.stop()

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # st.success("Groq client initialized.")
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────
# HELPER TO STRIP <think> … </think> FROM GROQ RESPONSES
# ───────────────────────────────────────────────────────────────────────────
def strip_think_tags(text: str) -> str:
    """
    Removes any <think> … </think> block (case-insensitive, single or multiline)
    that the reasoning model may prepend to its answer.
    """
    if not text:
        return text
    return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>',
                  '',
                  text,
                  flags=re.IGNORECASE | re.DOTALL).strip()

# ───────────────────────────────────────────────────────────────────────────
# Existing helper functions
# ───────────────────────────────────────────────────────────────────────────
def get_query_embedding(text):
    if not text:
        return None
    try:
        if USE_API_EMBEDDINGS and st_model is None:
            # Use API embeddings
            response = requests.post(
                EMBEDDING_API_URL,
                headers={"Content-Type": "application/json"},
                json={"texts": [text]},
                timeout=config.EMBEDDING_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding from response
            if "embeddings" in result:
                return result["embeddings"][0]
            elif "vectors" in result:
                return result["vectors"][0]
            elif isinstance(result, list):
                return result[0]
            else:
                return result.get("data", result.get("result", result))[0]
        else:
            # Use local model
            return st_model.encode(text).tolist()
    except Exception as e:
        st.error(f"    Error generating query embedding: {e}")
        return None

def find_relevant_attributes_with_sql(generated_sql: str):
    """
    Executes the LLM-generated SELECT via execute_readonly_sql().
    Always returns List[dict] rows.
    """
    if not generated_sql:
        return []

    sql_to_run = generated_sql.rstrip().rstrip(';')
    try:
        res = supabase.rpc("execute_readonly_sql", {"q": sql_to_run}).execute()

        if not res.data:
            return []

        # If each element is already a dict, just return the list as-is
        if isinstance(res.data[0], dict):
            return res.data

        # Otherwise grab the single JSON column
        first_key = next(iter(res.data[0].keys()))
        return [_to_dict(row[first_key]) for row in res.data]

    except Exception as e:
        return []

# ───────────────────────────────────────────────────────────────────────────
#  TEXT-TO-SQL GENERATION
# ───────────────────────────────────────────────────────────────────────────
def generate_sql_from_query(user_query, table_schema):
    """Uses Groq LLM with refined prompt and examples to generate SQL, attempting broad keyword matching."""
    # --- Full Original Prompt ---
    prompt = f"""Your task is to convert natural language questions into robust PostgreSQL SELECT queries for the "Leoni_attributes" table. The primary goal is to find matching rows even if the user slightly misspells a keyword or uses variations.

Strictly adhere to the following rules:
1. **Output Only SQL or NO_SQL**: Your entire response must be either a single, valid PostgreSQL SELECT statement ending with a semicolon (;) OR the exact word NO_SQL if the question cannot be answered by querying the table. Do not add explanations or markdown formatting.
2. **Target Table**: ONLY query the "Leoni_attributes" table.
3. **Column Quoting**: Use double quotes around column names ONLY if necessary (contain spaces, capitals beyond first letter, special chars). Check schema: {table_schema}
4. **SELECT Clause**:
   - Select columns explicitly asked for or implied by the user's condition.
   - Always include the columns involved in the WHERE clause conditions for verification.
   - Use `SELECT *` for requests about a specific part number.
5. **Robust Keyword Searching (CRITICAL RULE)**:
   - Identify the main descriptive keyword(s) in the user's question (e.g., colors, materials, types like 'black', 'connector', 'grey', 'terminal'). Do NOT apply this robust search to specific identifiers like part numbers unless the user query implies a pattern search (e.g., 'starts with...').
   - For the identified keyword(s), generate a comprehensive list of **potential variations**:
     - **Common Abbreviations:** (e.g., 'blk', 'bk' for black; 'gry', 'gy' for grey; 'conn' for connector; 'term' for terminal).
     - **Alternative Spellings/Regional Variations:** (e.g., 'grey'/'gray', 'colour'/'color').
     - **Different Casings:** (e.g., 'BLK', 'Gry', 'CONN').
     - ***Likely Typos/Common Misspellings:*** (e.g., for 'black', consider 'blak', 'blck'; for 'terminal', consider 'termnial', 'terminl'; for 'connector', 'conecter'). Use your knowledge of common typing errors, but be reasonable – don't include highly improbable variations.
   - Search for the original keyword AND **ALL generated variations** across **multiple relevant text-based attributes**. Relevant attributes typically include "Colour", "Name", "Material Name", "Context", "Type Of Connector", "Terminal Position Assurance", etc. – use context to decide which columns are most relevant for the specific keyword.
   - Use `ILIKE` with surrounding wildcards (`%`) (e.g., `'%variation%'`) for case-insensitive, substring matching for every term and variation.
   - Combine **ALL** these individual search conditions (original + all variations across all relevant columns) using the `OR` operator. This might result in a long WHERE clause, which is expected.
6. **LIMIT Clause**: Use `LIMIT 3` for specific part number lookups. Use `LIMIT 10` (or maybe `LIMIT 20` if many variations are generated) for broader keyword searches to provide a reasonable sample.
7. **NO_SQL**: Return NO_SQL for general knowledge questions, requests outside the table's scope, or highly ambiguous queries.

Table Schema: "Leoni_attributes"
{table_schema}

Examples:
User Question: "What is part number P00001636?"
SQL Query: SELECT * FROM "Leoni_attributes" WHERE "Number" = 'P00001636' LIMIT 3;

User Question: "Show me supplier parts containing 'connector'"
SQL Query: SELECT "Number", "Name", "Object Type Indicator", "Type Of Connector" FROM "Leoni_attributes" WHERE "Object Type Indicator" = 'Supplier Part' AND ("Name" ILIKE '%connector%' OR "Name" ILIKE '%conn%' OR "Name" ILIKE '%conecter%' OR "Type Of Connector" ILIKE '%connector%' OR "Type Of Connector" ILIKE '%conn%' OR "Type Of Connector" ILIKE '%conecter%') LIMIT 10; # Includes variation and likely typo

User Question: "Find part numbers starting with C"
SQL Query: SELECT "Number", "Name" FROM "Leoni_attributes" WHERE "Number" ILIKE 'C%' LIMIT 10; # Pattern search, not robust keyword search

User Question: "List part numbers that are black"
SQL Query: SELECT "Number", "Colour", "Name", "Material Name" FROM "Leoni_attributes" WHERE "Colour" ILIKE '%black%' OR "Colour" ILIKE '%blk%' OR "Colour" ILIKE '%bk%' OR "Colour" ILIKE '%BLK%' OR "Colour" ILIKE '%blak%' OR "Colour" ILIKE '%blck%' OR "Name" ILIKE '%black%' OR "Name" ILIKE '%blk%' OR "Name" ILIKE '%bk%' OR "Name" ILIKE '%BLK%' OR "Name" ILIKE '%blak%' OR "Name" ILIKE '%blck%' OR "Material Name" ILIKE '%black%' OR "Material Name" ILIKE '%blk%' OR "Material Name" ILIKE '%bk%' OR "Material Name" ILIKE '%BLK%' OR "Material Name" ILIKE '%blak%' OR "Material Name" ILIKE '%blck%' LIMIT 10; # Example with typos added

User Question: "Any grey parts?"
SQL Query: SELECT "Number", "Colour", "Name" FROM "Leoni_attributes" WHERE "Colour" ILIKE '%grey%' OR "Colour" ILIKE '%gray%' OR "Colour" ILIKE '%gry%' OR "Colour" ILIKE '%gy%' OR "Colour" ILIKE '%GRY%' OR "Colour" ILIKE '%graey%' OR "Name" ILIKE '%grey%' OR "Name" ILIKE '%gray%' OR "Name" ILIKE '%gry%' OR "Name" ILIKE '%gy%' OR "Name" ILIKE '%GRY%' OR "Name" ILIKE '%graey%' LIMIT 10; # Example with alternative spelling, typo

User Question: "Parts with more than 10 cavities"
SQL Query: SELECT "Number", "Number Of Cavities" FROM "Leoni_attributes" WHERE "Number Of Cavities" > 10 LIMIT 10;

User Question: "What is a TPA?"
SQL Query: NO_SQL

User Question: "{user_query}"
SQL Query:
"""
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Text-to-SQL assistant generating PostgreSQL queries optimized for finding matches despite keyword variations and typos."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL_FOR_SQL,
            temperature=0.1,
            max_tokens=131072
        )
        if not response.choices or not response.choices[0].message:
            return None

        # ░░░ STRIP REASONING BLOCK FIRST ░░░
        generated_sql = strip_think_tags(response.choices[0].message.content)

        if generated_sql == "NO_SQL":
            return None

        # Check if valid SQL (starts with SELECT, ends with ;)
        if generated_sql.upper().startswith("SELECT") and generated_sql.rstrip().endswith(';'):
            forbidden = ["UPDATE", "DELETE", "INSERT", "DROP", "TRUNCATE",
                         "ALTER", "CREATE", "EXECUTE", "GRANT", "REVOKE"]
            pattern = re.compile(r'\b(?:' + '|'.join(forbidden) + r')\b', re.IGNORECASE)
            if pattern.search(generated_sql):
                return None

            # Check if the target table name appears after FROM
            table_name_pattern = r'FROM\s+(?:[\w]+\.)?("?' + ATTRIBUTE_TABLE_NAME + r'"?)'
            if not re.search(table_name_pattern, generated_sql, re.IGNORECASE):
                return None

            return generated_sql
        else:
            return None
    except Exception as e:
        return None

# ───────────────────────────────────────────────────────────────────────────
#  SQL EXECUTION FUNCTION
# ───────────────────────────────────────────────────────────────────────────
def _to_dict(maybe_json):
    """
    Ensure the value is a Python dict. Decode JSON/JSONB strings if needed.
    """
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        # Try fast JSON decode first
        try:
            return json.loads(maybe_json)
        except json.JSONDecodeError:
            pass
        # Fallback: literal_eval handles single quotes, etc.
        try:
            return ast.literal_eval(maybe_json)
        except Exception:
            pass
    # Give up – return an empty dict to avoid crashing format_context
    return {}

# ───────────────────────────────────────────────────────────────────────────
#  CONTEXT FORMATTING
# ───────────────────────────────────────────────────────────────────────────
def format_context(attribute_rows):
    context_str = ""
    attr_present = bool(attribute_rows)
    if attr_present:
        context_str += "Context from Leoni Attributes Table:\n\n"
        for i, row in enumerate(attribute_rows):
            context_str += f"--- Attribute Row {i+1} ---\n"
            row_str_parts = []
            for key, value in row.items():
                if value is not None:
                    row_str_parts.append(f"  {key}: {json.dumps(value)}")
            context_str += "\n".join(row_str_parts)
            context_str += "\n---\n\n"
    if not attr_present:
        return "No relevant information found in the knowledge base (attributes)."
    return context_str.strip()

# ───────────────────────────────────────────────────────────────────────────
#  CHAT COMPLETION FOR ANSWERS
# ───────────────────────────────────────────────────────────────────────────
def get_groq_chat_response(prompt, context_provided=True):
    if context_provided:
        system_message = ("You are a helpful assistant knowledgeable about LEOparts standards and attributes. "
                          "Answer the user's question based *only* on the provided context from the Standards Document and/or the Attributes Table. "
                          "The Attributes Table context shows rows retrieved based on the user's query; assume these rows accurately reflect the query's conditions as interpreted by the client-side filters. "
                          "Synthesize information from both sources if relevant and available. Be concise. If listing items, like part numbers, list them clearly.")
    else:
        system_message = ("You are a helpful assistant knowledgeable about LEOparts standards and attributes. "
                          "You were unable to find relevant information in the knowledge base (documents or attributes) to answer the user's question. "
                          "State clearly that the information is not available in the provided materials. Do not make up information or answer from general knowledge.")

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL_FOR_ANSWER,
            temperature=0.1,
            stream=False
        )
        raw_reply = response.choices[0].message.content
        return strip_think_tags(raw_reply)
    except Exception as e:
        st.error(f"    Error calling Groq API: {e}")
        return "Error contacting LLM."

# ───────────────────────────────────────────────────────────────────────────
#  MAIN CHAT LOOP
# ───────────────────────────────────────────────────────────────────────────
def run_chatbot():
    st.title("🤖 Chatbot")
    st.markdown("Ask questions about the extracted data.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_attribute_rows = []
                context_was_found = False
                generated_sql = None

                # 1. Attempt Text-to-SQL generation
                generated_sql = generate_sql_from_query(prompt, leoni_attributes_schema_for_main_loop)

                # 2. Execute SQL (using client-side filters)
                if generated_sql:
                    relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
                    if relevant_attribute_rows:
                        context_was_found = True

                # 4. Prepare Context
                context_str = format_context(relevant_attribute_rows)

                # 5. Generate Response
                prompt_for_llm = f"""Context:
{context_str}

User Question: {prompt}

Answer the user question based *only* on the provided context."""
                llm_response = get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)
                
                # Display the response
                st.markdown(llm_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": llm_response})

# The chatbot will be called from app.py
if __name__ == "__main__":
    run_chatbot() 