# LEOPARTS DEEP DIVE ANALYSIS

## 🎯 **COMPLETE SYSTEM ARCHITECTURE ANALYSIS**

### **1. APPLICATION STARTUP DEEP DIVE**

#### **app.py:main() - Entry Point Analysis**
```python
def main():
    # UI Configuration
    st.set_page_config(page_title="LEOPARTS", page_icon="🦁", layout="wide")
    
    # CSS Styling System
    # - Blue gradient theme (#1e3c72 to #4a90e2)
    # - Custom button styling with hover effects
    # - Responsive layout with sidebar navigation
    
    # Navigation System
    with st.sidebar:
        # Navigation buttons with page switching
        # - Home (app.py)
        # - Chatbot (pages/chatbot.py)
        # - Extraction (pages/extraction_attributs.py)
        # - Debug Interface (debug_interface.py)
    
    # Main Content Area
    # - Header band with LEONI branding
    # - Welcome section with action buttons
    # - Two-column layout for main actions
```

**Key Functions:**
- `st.set_page_config()` - Streamlit UI initialization
- `st.sidebar` - Navigation container
- `st.switch_page()` - Page navigation system
- CSS styling with gradient themes and animations

---

### **2. DOCUMENT PROCESSING PIPELINE DEEP DIVE**

#### **PDF Processing Flow - Complete Analysis**

**Step 1: File Upload & Validation**
```python
# pages/extraction_attributs.py
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)
```

**Step 2: PDF Processing Pipeline**
```python
# pdf_processor.py:process_uploaded_pdfs()
async def process_uploaded_pdfs(uploaded_files, temp_dir):
    # 1. Create temporary directory
    # 2. Save uploaded files
    # 3. Process each PDF asynchronously
    # 4. Extract text and images
    # 5. Create document chunks
    # 6. Tag chunks with attributes
    # 7. Return processed documents
```

**Step 3: Individual PDF Processing**
```python
# pdf_processor.py:process_single_pdf()
async def process_single_pdf(file_path, file_basename, client, model_name):
    # Text Extraction with PyMuPDF
    doc = fitz.open(file_path)
    text_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
    
    # Image Extraction with Mistral Vision
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            # Extract and process images
            # Use Mistral Vision API for image analysis
    
    # Chunk Creation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text_content)
    
    # Attribute Tagging
    tagged_chunks = []
    for chunk in chunks:
        tags = tag_chunk_with_dictionary(chunk, ATTRIBUTE_REGEXES)
        # Create Document objects with metadata
```

**Step 4: Vector Store Setup**
```python
# vector_store.py:setup_vector_store()
def setup_vector_store(documents, embedding_function):
    # 1. Initialize Chroma client
    # 2. Create or load collection
    # 3. Add documents with embeddings
    # 4. Create retriever with similarity threshold
    # 5. Return configured retriever
```

---

### **3. ENHANCED SEARCH QUERY SYSTEM DEEP DIVE**

#### **create_enhanced_search_queries() - Complete Analysis**

**Function Purpose:**
- Generate multiple search queries for better chunk retrieval
- Use attribute dictionary values and synonyms
- Create combined queries for comprehensive search

**Detailed Implementation:**
```python
def create_enhanced_search_queries(attribute_key: str, base_query: str) -> list:
    queries = [base_query]  # Always include original query
    
    # 1. Get dictionary values for this attribute
    dict_values = ATTRIBUTE_DICT.get(attribute_key, [])
    
    # 2. Add attribute-specific synonyms
    attribute_terms = {
        "Material Filling": ["filling", "additive", "filler", "glass fiber", "GF", "GB"],
        "Material Name": ["material", "polymer", "PA", "PBT", "PP", "PET", "PC"],
        "Contact Systems": ["contact system", "terminal system", "MQS", "MCP", "TAB"],
        # ... 25+ attribute categories
    }
    
    # 3. Add dictionary values as search terms
    for value in dict_values[:10]:
        if isinstance(value, str) and len(value) > 1:
            queries.append(value)
    
    # 4. Create combined queries
    for value in dict_values[:5]:
        combined_query = f"{base_query} {value}"
        queries.append(combined_query)
    
    # 5. Remove duplicates and limit
    unique_queries = list(dict.fromkeys(queries))[:20]
    return unique_queries
```

**Example Queries Generated:**
For "Contact Systems" attribute:
- Base: "Contact Systems"
- Dictionary values: ["MQS", "MCP", "TAB", "MLK"]
- Synonyms: ["contact system", "terminal system"]
- Combined: ["Contact Systems MQS", "Contact Systems MCP"]
- **Total: 20 unique queries**

#### **retrieve_and_log_chunks() - Complete Analysis**

**Function Purpose:**
- Execute enhanced search queries
- Collect unique chunks by content hash
- Log retrieval process for debugging

**Detailed Implementation:**
```python
def retrieve_and_log_chunks(retriever, query: str, attribute_key: str):
    # 1. Create enhanced search queries
    enhanced_queries = create_enhanced_search_queries(attribute_key, query)
    
    # 2. Execute each query
    all_chunks = []
    seen_chunks = set()  # Track unique chunks by content hash
    
    for search_query in enhanced_queries:
        try:
            chunks = retriever.invoke(search_query)
            
            # 3. Add unique chunks only
            for chunk in chunks:
                chunk_hash = _hash_chunk(chunk)
                if chunk_hash not in seen_chunks:
                    seen_chunks.add(chunk_hash)
                    all_chunks.append(chunk)
                    
        except Exception as e:
            logger.warning(f"Query '{search_query}' failed: {e}")
            continue
    
    # 4. Limit total chunks
    max_chunks = 10
    if len(all_chunks) > max_chunks:
        all_chunks = all_chunks[:max_chunks]
    
    # 5. Log retrieval process
    _log_retrieved_chunks(attribute_key, query, all_chunks)
    return all_chunks
```

---

### **4. THREE-STAGE EXTRACTION PROCESS DEEP DIVE**

#### **Stage 1: Web Data Extraction - Complete Analysis**

**Web Scraping Process:**
```python
# llm_interface.py:scrape_website_table_html()
async def scrape_website_table_html(part_number: str):
    # 1. Setup AsyncWebCrawler
    crawler = AsyncWebCrawler(
        browser_config=BrowserConfig(
            browser_type="chromium",
            headless=True
        )
    )
    
    # 2. Generate search URLs
    search_urls = [
        f"https://www.leoni.com/en/products/automotive/connectors/{part_number}",
        f"https://www.leoni.com/en/products/automotive/connectors/search?q={part_number}",
        # ... multiple supplier URLs
    ]
    
    # 3. Crawl websites
    for url in search_urls:
        try:
            result = await crawler.arun(
                url,
                CrawlerRunConfig(
                    cache_mode=CacheMode.ENABLED
                )
            )
            
            # 4. Extract table HTML
            if result.html_content:
                soup = BeautifulSoup(result.html_content, 'html.parser')
                tables = soup.find_all('table')
                
                # 5. Clean and return HTML
                cleaned_html = clean_scraped_html(result.html_content, url)
                if cleaned_html:
                    return cleaned_html
                    
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            continue
    
    return None
```

**Web Extraction Chain:**
```python
# llm_interface.py:create_web_extraction_chain()
def create_web_extraction_chain(llm):
    template = """
    You are an expert data extractor. Extract information from the provided website data.
    
    --- Cleaned Scraped Website Data ---
    {cleaned_web_data}
    --- End Cleaned Scraped Website Data ---
    
    Extraction Instructions:
    {extraction_instructions}
    
    For the attribute key "{attribute_key}", respond with ONLY a JSON object:
    {{"{attribute_key}": "extracted_value"}}
    """
    
    web_chain = (
        RunnableParallel(
            cleaned_web_data=lambda x: x['cleaned_web_data'],
            extraction_instructions=lambda x: x['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key']
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return web_chain
```

#### **Stage 2: NuMind Structured Extraction - Complete Analysis**

**NuMind Chain Creation:**
```python
# llm_interface.py:create_numind_extraction_chain()
def create_numind_extraction_chain():
    try:
        from numind import NuMind
        
        if not NUMIND_API_KEY or not NUMIND_PROJECT_ID:
            return None
            
        client = NuMind(api_key=NUMIND_API_KEY)
        return client
        
    except ImportError:
        logger.warning("NuMind not available")
        return None
```

**NuMind Extraction Process:**
```python
# llm_interface.py:extract_with_numind_using_schema()
async def extract_with_numind_using_schema(client, file_bytes, extraction_schema):
    # 1. Get custom schema from numind_schema_config.py
    schema = get_custom_schema()
    
    # 2. Call NuMind API
    result = await client.extract(
        file_bytes,
        schema=schema,
        project_id=NUMIND_PROJECT_ID
    )
    
    # 3. Process structured results
    if result and result.extractions:
        return result.extractions[0].data
    return None
```

**Schema Configuration:**
```python
# numind_schema_config.py
CUSTOM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "Material Filling": {
            "type": "string",
            "enum": ["none", "GF", "CF", "(GB+GF)"]
        },
        "Material Name": {
            "type": "string", 
            "enum": ["PA66", "PBT", "PA", "Silicone Rubber", "PA6"]
        },
        # ... 25+ attribute definitions
    }
}
```

#### **Stage 3: Enhanced PDF Fallback - Complete Analysis**

**PDF Chain with Context Lambda:**
```python
# llm_interface.py:create_pdf_extraction_chain()
def create_pdf_extraction_chain(retriever, llm):
    template = """
    You are an expert data extractor. Extract information from the provided context.
    
    Context:
    {context}
    
    Extraction Instructions:
    {extraction_instructions}
    
    Available values for {attribute_key}: {available_values}
    
    Respond with ONLY a JSON object:
    {{"{attribute_key}": "extracted_value"}}
    """
    
    pdf_chain = (
        RunnableParallel(
            # KEY: Context lambda with enhanced search
            context=lambda x: format_docs(
                retrieve_and_log_chunks(
                    retriever, 
                    x['extraction_instructions'], 
                    x['attribute_key']
                )
            ),
            extraction_instructions=lambda x: x['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key'],
            part_number=lambda x: x.get('part_number', "Not Provided"),
            available_values=lambda x: str(ATTRIBUTE_DICT.get(x['attribute_key'], []))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return pdf_chain
```

**Enhanced Prompt System:**
```python
# For Stage 3, enhanced prompts are used:
enhanced_instruction = f"""
{pdf_instruction}

CRITICAL: Previous extraction returned '{previous_value}'. 
This may be incorrect. Please be extremely thorough and look for ANY mention 
of this attribute, even if it's not explicitly labeled. Consider technical 
specifications, material properties, dimensions, or any related information 
that might indicate this attribute's value.
"""
```

---

### **5. LLM PROCESSING DEEP DIVE**

#### **_invoke_chain_and_process() - Complete Analysis**

**Function Purpose:**
- Execute LLM chains with error handling
- Process and validate responses
- Handle rate limits and retries

**Detailed Implementation:**
```python
# llm_interface.py:_invoke_chain_and_process()
async def _invoke_chain_and_process(chain, input_data, attribute_key):
    try:
        # 1. Execute chain
        result = await chain.ainvoke(input_data)
        
        # 2. Strip think tags
        cleaned_result = strip_think_tags(result)
        
        # 3. Validate JSON format
        if not cleaned_result.strip().startswith('{'):
            return f'{{"error": "Invalid JSON format: {cleaned_result[:100]}..."}}'
        
        # 4. Check for required attribute key
        if attribute_key not in cleaned_result:
            return f'{{"error": "Missing attribute key {attribute_key} in response"}}'
        
        return cleaned_result
        
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        return f'{{"error": "Exception during chain execution: {str(e)}"}}'
```

**Response Processing:**
```python
# llm_interface.py:strip_think_tags()
def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses"""
    if not text:
        return text
    return re.sub(
        r'<\s*think\s*>.*?<\s*/\s*think\s*>',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()
```

---

### **6. VECTOR STORE & EMBEDDING SYSTEM DEEP DIVE**

#### **HuggingFaceAPIEmbeddings - Complete Analysis**

**Class Implementation:**
```python
# vector_store.py:HuggingFaceAPIEmbeddings
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_url="https://hbaananou-embedder-model.hf.space/embed"):
        self.api_url = api_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 1. Pre-process texts (limit length)
        processed_texts = []
        for text in texts:
            if len(text) > config.EMBEDDING_MAX_TEXT_LENGTH:
                text = text[:config.EMBEDDING_MAX_TEXT_LENGTH]
            processed_texts.append(text)
        
        # 2. Batch processing
        all_embeddings = []
        for i in range(0, len(processed_texts), config.EMBEDDING_BATCH_SIZE):
            batch_texts = processed_texts[i:i + config.EMBEDDING_BATCH_SIZE]
            
            # 3. API call with retry logic
            for retry in range(3):
                try:
                    response = requests.post(
                        self.api_url,
                        headers={"Content-Type": "application/json"},
                        json={"texts": batch_texts},
                        timeout=config.EMBEDDING_TIMEOUT
                    )
                    response.raise_for_status()
                    
                    # 4. Extract embeddings from response
                    result = response.json()
                    if "embeddings" in result:
                        batch_embeddings = result["embeddings"]
                    elif "vectors" in result:
                        batch_embeddings = result["vectors"]
                    else:
                        batch_embeddings = result.get("data", result.get("result", result))
                    
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    if retry == 2:
                        raise e
                    time.sleep(1)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        # Single text embedding
        embeddings = self.embed_documents([text])
        return embeddings[0]
```

#### **Chroma Vector Store Setup - Complete Analysis**

**Store Initialization:**
```python
# vector_store.py:setup_vector_store()
def setup_vector_store(documents, embedding_function):
    # 1. Get Chroma client
    client = get_chroma_client()
    
    # 2. Create or load collection
    collection_name = config.COLLECTION_NAME
    try:
        collection = client.get_collection(collection_name)
        logger.info(f"Loaded existing collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        logger.info(f"Created new collection: {collection_name}")
    
    # 3. Add documents with embeddings
    if documents:
        # Prepare documents for Chroma
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = embedding_function.embed_documents(texts)
        
        # Add to collection
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Added {len(documents)} documents to vector store")
    
    # 4. Create retriever with similarity threshold
    retriever = ThresholdRetriever(
        vectorstore=Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        ),
        search_kwargs={"k": config.RETRIEVER_K},
        threshold=config.VECTOR_SIMILARITY_THRESHOLD
    )
    
    return retriever
```

#### **ThresholdRetriever - Complete Analysis**

**Custom Retriever Implementation:**
```python
# vector_store.py:ThresholdRetriever
class ThresholdRetriever:
    def __init__(self, vectorstore, search_kwargs, threshold):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
        self.threshold = threshold
    
    def invoke(self, query: str) -> List[Document]:
        # 1. Get embeddings for query
        query_embedding = self.vectorstore.embedding_function.embed_query(query)
        
        # 2. Search vector store
        results = self.vectorstore.similarity_search_with_score(
            query,
            **self.search_kwargs
        )
        
        # 3. Filter by similarity threshold
        filtered_results = []
        for doc, score in results:
            if score >= self.threshold:
                filtered_results.append(doc)
        
        return filtered_results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
    
    async def ainvoke(self, query: str) -> List[Document]:
        return self.invoke(query)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)
```

---

### **7. DEBUG & MONITORING SYSTEM DEEP DIVE**

#### **DebugLogger - Complete Analysis**

**Class Implementation:**
```python
# debug_logger.py:DebugLogger
class DebugLogger:
    def __init__(self, log_file="debug_log.txt", enable_console=True):
        self.log_file = log_file
        self.enable_console = enable_console
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_counter = 0
        
        # Clear previous log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== DEBUG LOG STARTED: {datetime.now().isoformat()} ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write("=" * 80 + "\n\n")
    
    def _log(self, level: str, message: str, data=None, context=None):
        self.step_counter += 1
        timestamp = datetime.now().isoformat()
        
        # Build log entry
        log_entry = f"\n{'='*60}\n"
        log_entry += f"STEP {self.step_counter} - {level.upper()} - {timestamp}\n"
        log_entry += f"{'='*60}\n"
        log_entry += f"MESSAGE: {message}\n"
        
        if context:
            log_entry += f"CONTEXT: {json.dumps(context, indent=2, default=str)}\n"
        
        if data is not None:
            if isinstance(data, (dict, list)):
                log_entry += f"DATA: {json.dumps(data, indent=2, default=str)}\n"
            else:
                log_entry += f"DATA: {str(data)}\n"
        
        log_entry += f"{'='*60}\n"
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Console output
        if self.enable_console:
            if level == "error":
                logger.error(f"DEBUG [{self.step_counter}]: {message}")
            elif level == "warning":
                logger.warning(f"DEBUG [{self.step_counter}]: {message}")
            elif level == "info":
                logger.info(f"DEBUG [{self.step_counter}]: {message}")
            else:
                logger.debug(f"DEBUG [{self.step_counter}]: {message}")
```

**Specialized Logging Methods:**
```python
def llm_request(self, prompt: str, model: str, temperature: float, max_tokens: int, context=None):
    self.info(
        f"LLM REQUEST: {model}",
        data={
            "prompt_length": len(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        context=context
    )

def llm_response(self, model: str, response: str, tokens_used: int, latency: float, context=None):
    self.info(
        f"LLM RESPONSE: {model}",
        data={
            "response_length": len(response),
            "tokens_used": tokens_used,
            "latency": latency
        },
        context=context
    )

def extraction_step(self, attribute: str, source: str, input_data, output_data, success: bool, context=None):
    self.info(
        f"EXTRACTION STEP: {attribute} from {source}",
        data={
            "success": success,
            "input_data": input_data,
            "output_data": output_data
        },
        context=context
    )
```

---

### **8. CONFIGURATION SYSTEM DEEP DIVE**

#### **config.py - Complete Analysis**

**Environment Variables:**
```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-32b")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "mistral-small-latest")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS = True

# API Embedding Configuration
USE_API_EMBEDDINGS = os.getenv("USE_API_EMBEDDINGS", "true").lower() == "true"
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://hbaananou-embedder-model.hf.space/embed")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 5))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", 120))
EMBEDDING_MAX_TEXT_LENGTH = int(os.getenv("EMBEDDING_MAX_TEXT_LENGTH", 4000))

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db_prod")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_qa_prod_collection")

# Retriever Configuration
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 8))
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.7))

# LLM Request Configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 8192))
```

**Configuration Validation:**
```python
# Validation
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables.")

# Chroma Settings
class SimpleChromaSettings:
    def __init__(self, persistent_flag):
        self.is_persistent = persistent_flag

is_persistent = bool(CHROMA_PERSIST_DIRECTORY)
CHROMA_SETTINGS = SimpleChromaSettings(is_persistent)
```

---

### **9. EXTRACTION PROMPTS SYSTEM DEEP DIVE**

#### **Prompt Structure Analysis**

**Material Properties Prompts:**
```python
# extraction_prompts.py
MATERIAL_PROMPT = """
Extract the material filling information from the provided context.

Context:
{context}

Instructions:
- Look for material filling additives like glass fiber (GF), carbon fiber (CF), etc.
- Check for abbreviations like GF, CF, GB, MF
- Consider combinations like (GB+GF)
- If no filling is mentioned, return "none"

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Material Filling": "extracted_value"}}
"""

MATERIAL_NAME_PROMPT = """
Extract the main material name from the provided context.

Context:
{context}

Instructions:
- Look for polymer names like PA66, PBT, PP, PET, PC
- Check for silicone rubber or other materials
- Consider material combinations
- Focus on the primary material used

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Material Name": "extracted_value"}}
"""
```

**Physical/Mechanical Attributes Prompts:**
```python
PULL_TO_SEAT_PROMPT = """
Extract pull-to-seat information from the provided context.

Context:
{context}

Instructions:
- Look for pull-to-seat, pull-back, or tug-lock mechanisms
- Check for terminal insertion and seating information
- Consider mechanical locking mechanisms
- Focus on insertion and removal forces

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Pull-To-Seat": "extracted_value"}}
"""

GENDER_PROMPT = """
Extract the connector gender information from the provided context.

Context:
{context}

Instructions:
- Look for male/female designations
- Check for plug/receptacle terminology
- Consider socket/header descriptions
- Focus on mating interface type

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Gender": "extracted_value"}}
"""
```

**Electrical Properties Prompts:**
```python
CONTACT_SYSTEMS_PROMPT = """
Extract the contact system information from the provided context.

Context:
{context}

Instructions:
- Look for contact system types like MQS, MCP, TAB, MLK
- Check for terminal system specifications
- Consider contact system codes and standards
- Focus on electrical connection method

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Contact Systems": "extracted_value"}}
"""

TERMINAL_POSITION_ASSURANCE_PROMPT = """
Extract terminal position assurance (TPA) information from the provided context.

Context:
{context}

Instructions:
- Look for TPA (Terminal Position Assurance) features
- Check for anti-backout mechanisms
- Consider secondary locking features
- Focus on terminal retention methods

Available values: {available_values}

Respond with ONLY a JSON object:
{{"Terminal Position Assurance": "extracted_value"}}
"""
```

---

### **10. ERROR HANDLING & FALLBACKS DEEP DIVE**

#### **Comprehensive Error Handling Strategy**

**API Failure Handling:**
```python
# Groq API Error Handling
try:
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS
    )
except Exception as e:
    logger.error(f"Groq API error: {e}")
    # Fallback to local model or retry logic
    return f'{{"error": "API failure: {str(e)}"}}'
```

**Web Scraping Error Handling:**
```python
# Web Scraping Error Handling
try:
    result = await crawler.arun(url, config)
    if result.html_content:
        return clean_scraped_html(result.html_content, url)
except Exception as e:
    logger.warning(f"Web scraping failed for {url}: {e}")
    # Continue to PDF-only extraction
    return None
```

**NuMind Error Handling:**
```python
# NuMind Error Handling
try:
    result = await client.extract(file_bytes, schema=schema)
    if result and result.extractions:
        return result.extractions[0].data
except Exception as e:
    logger.error(f"NuMind extraction failed: {e}")
    # Fallback to PDF extraction
    return None
```

**Vector Store Error Handling:**
```python
# Vector Store Error Handling
try:
    chunks = retriever.invoke(search_query)
    return chunks
except Exception as e:
    logger.error(f"Vector store query failed: {e}")
    # Fallback to basic retrieval
    return []
```

**Embedding Error Handling:**
```python
# Embedding Error Handling
try:
    embeddings = embedding_function.embed_documents(texts)
    return embeddings
except Exception as e:
    logger.error(f"Embedding generation failed: {e}")
    # Fallback to local model or error response
    return []
```

---

### **11. PERFORMANCE OPTIMIZATIONS DEEP DIVE**

#### **Caching Strategies**

**Streamlit Caching:**
```python
@st.cache_resource
def initialize_embeddings():
    """Cache embedding function initialization"""
    embeddings = get_embedding_function()
    return embeddings

@st.cache_resource
def initialize_llm_cached():
    """Cache LLM initialization"""
    llm_instance = initialize_llm()
    return llm_instance
```

**Batch Processing:**
```python
# Embedding Batch Processing
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    batch_size = config.EMBEDDING_BATCH_SIZE
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = self._process_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

**Async Operations:**
```python
# Async PDF Processing
async def process_uploaded_pdfs(uploaded_files, temp_dir):
    tasks = []
    for uploaded_file in uploaded_files:
        task = process_single_pdf(file_path, file_basename, client, model_name)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [doc for docs in results if docs for doc in docs]
```

**Thread Pool for PDF Processing:**
```python
# Global thread pool
pdf_thread_pool = ThreadPoolExecutor(max_workers=2)

# Use thread pool for CPU-intensive operations
def process_pdfs_in_background(uploaded_files, temp_dir):
    return asyncio.create_task(process_uploaded_pdfs(uploaded_files, temp_dir))
```

---

### **12. COMPLETE EXECUTION SEQUENCE DEEP DIVE**

#### **Step-by-Step Detailed Flow**

**1. Application Startup:**
```python
# app.py:main()
def main():
    # UI Setup
    st.set_page_config(page_title="LEOPARTS", page_icon="🦁", layout="wide")
    
    # CSS Styling
    st.markdown("""
        <style>
        .header-band {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #4a90e2 100%);
            color: white;
            padding: 0.1rem 0;
            margin: 0 0 0.2rem 0;
            text-align: center;
            box-shadow: 0 2px 6px rgba(30, 60, 114, 0.15);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
        if st.button("🏠 Home"):
            st.switch_page("app.py")
        if st.button("💬 Chat with Leoparts"):
            st.switch_page("pages/chatbot.py")
        if st.button("📄 Extract a new Part"):
            st.switch_page("pages/extraction_attributs.py")
        if st.button("🔍 Debug Interface"):
            st.switch_page("debug_interface.py")
    
    # Main Content
    st.markdown("""
        <div class="header-band">
            <h1>LEOPARTS</h1>
            <h2>LEONI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown("""
        <div class="welcome-section">
            <h2>Welcome!</h2>
            <p>Choose a Tool</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 Chat with Leoparts", key="main_chat_btn", use_container_width=True):
                st.switch_page("pages/chatbot.py")
        with c2:
            if st.button("📄 Extract a new Part", key="main_extract_btn", use_container_width=True):
                st.switch_page("pages/extraction_attributs.py")
```

**2. Document Processing Pipeline:**
```python
# pages/extraction_attributs.py
if process_button and uploaded_files:
    # Reset state
    st.session_state.retriever = None
    st.session_state.pdf_chain = None
    st.session_state.web_chain = None
    st.session_state.processed_files = []
    reset_evaluation_state()
    
    # Store uploaded file data
    st.session_state.uploaded_file_data = [(f.name, f.getvalue()) for f in uploaded_files]
    
    # PDF Processing
    with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
        processed_docs = loop.run_until_complete(
            process_uploaded_pdfs(uploaded_files, temp_dir)
        )
    
    # Vector Store Setup
    if processed_docs and len(processed_docs) > 0:
        with st.spinner("Indexing documents in vector store..."):
            st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
            
            if st.session_state.retriever:
                st.session_state.processed_files = filenames
                st.session_state.processed_documents = processed_docs
                
                # Create Extraction Chains
                with st.spinner("Preparing extraction engines..."):
                    st.session_state.pdf_chain = create_pdf_extraction_chain(st.session_state.retriever, llm)
                    st.session_state.web_chain = create_web_extraction_chain(llm)
                    st.session_state.numind_chain = create_numind_extraction_chain()
```

**3. Three-Stage Extraction Process:**
```python
# Stage 1: Web Data Extraction
if part_number:
    scraped_table_html = loop.run_until_complete(
        scrape_website_table_html(part_number)
    )
    
    if scraped_table_html:
        for prompt_name, instructions in prompts_to_run.items():
            web_input = {
                "cleaned_web_data": scraped_table_html,
                "attribute_key": prompt_name,
                "extraction_instructions": instructions["web"]
            }
            
            json_result_str = loop.run_until_complete(
                _invoke_chain_and_process(st.session_state.web_chain, web_input, f"{prompt_name} (Web)")
            )
            
            # Parse and store results
            parsed_json = extract_json_from_string(json_result_str)
            if parsed_json and prompt_name in parsed_json:
                final_answer_value = str(parsed_json[prompt_name])
                if "not found" in final_answer_value.lower():
                    pdf_fallback_needed.append(prompt_name)

# Stage 2: NuMind Structured Extraction
if pdf_fallback_needed:
    file_data = st.session_state.uploaded_file_data[0][1]
    extraction_schema = get_custom_schema()
    
    numind_result = loop.run_until_complete(
        extract_with_numind_using_schema(st.session_state.numind_chain, file_data, extraction_schema)
    )
    
    if numind_result:
        for prompt_name in pdf_fallback_needed:
            final_answer_value = extract_specific_attribute_from_numind_result(numind_result, prompt_name)
            if final_answer_value is None:
                final_answer_value = "NOT FOUND"

# Stage 3: Enhanced PDF Fallback
for prompt_name in final_fallback_needed:
    context_chunks = fetch_chunks(
        st.session_state.retriever,
        part_number,
        prompt_name,
        k=12
    )
    context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])
    
    enhanced_pdf_input = {
        "context": context_text,
        "extraction_instructions": enhanced_instruction,
        "attribute_key": prompt_name,
        "part_number": part_number if part_number else "Not Provided"
    }
    
    json_result_str = loop.run_until_complete(
        _invoke_chain_and_process(st.session_state.pdf_chain, enhanced_pdf_input, f"{prompt_name} (Final Fallback)")
    )
```

**4. Result Display and Evaluation:**
```python
# Display Results
if st.session_state.evaluation_results:
    results_df = pd.DataFrame(st.session_state.evaluation_results)
    
    # Card UI Display
    num_cols = 5
    cards = st.session_state.evaluation_results
    cols = st.columns(num_cols)
    
    for idx, result in enumerate(cards):
        with cols[idx % num_cols]:
            prompt_name = result.get('Prompt Name', f'Field {idx+1}')
            extracted_value = result.get('Extracted Value', '')
            source = result.get('Source', '')
            
            # Determine circle color based on source
            source_label = str(source).strip().lower()
            if source_label == 'web':
                circle_color = '#28a745'  # green
            elif source_label == 'numind':
                circle_color = '#007bff'  # blue
            elif source_label == 'pdf':
                circle_color = '#ffc107'  # yellow
            else:
                circle_color = '#6c757d'  # gray
            
            st.markdown(f"""
            <div class='result-item'>
                <div class='result-label'>🔍 {prompt_name}</div>
                <div class='result-value' title='{extracted_value}' style='position:relative; display:flex; align-items:center;'>
                    <span style='flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>
                        {extracted_value[:100] + ('...' if len(extracted_value) > 100 else '')}
                    </span>
                    <span style='position:absolute; right:8px; top:50%; transform:translateY(-50%); display:inline-block; width:14px; height:14px; border-radius:50%; background:{circle_color};'></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
```

---

This deep-dive analysis covers every aspect of your LEOPARTS system, from the initial application startup through the complete three-stage extraction process, including all the intricate details of function implementations, error handling, performance optimizations, and system integrations. 