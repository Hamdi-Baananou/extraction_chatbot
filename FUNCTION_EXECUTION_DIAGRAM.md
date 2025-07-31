# LEOPARTS Function Execution Diagram

## Project Overview
LEOPARTS is a document processing and attribute extraction system for LEONI automotive parts. It uses a multi-stage approach combining web scraping, PDF processing, and AI-powered extraction.

## Complete Function Execution Flow

### 1. APPLICATION STARTUP
```
app.py:main()
├── st.set_page_config() - Configure Streamlit UI
├── CSS styling setup
├── Navigation sidebar creation
└── Welcome page display
```

### 2. USER NAVIGATION PATHS

#### Path A: Chatbot Interface
```
pages/chatbot.py:run_chatbot()
├── Configuration & Initialization
│   ├── Load secrets (SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY)
│   ├── Initialize Supabase client
│   ├── Initialize embedding function (HuggingFace API or local)
│   └── Initialize Groq LLM client
├── Chat Interface
│   ├── Display chat history
│   ├── Process user input
│   └── Generate responses
└── Query Processing Pipeline
    ├── generate_sql_from_query() - Convert natural language to SQL
    ├── find_relevant_attributes_with_sql() - Execute SQL query
    ├── format_context() - Format results for LLM
    └── get_groq_chat_response() - Generate final answer
```

#### Path B: Document Extraction Interface
```
pages/extraction_attributs.py
├── UI Setup & Styling
├── Component Initialization
│   ├── initialize_embeddings() - Setup embedding function
│   ├── initialize_llm_cached() - Setup LLM
│   └── Install Playwright browsers
└── Main Processing Flow
```

### 3. DOCUMENT PROCESSING PIPELINE

#### 3.1 File Upload & Processing
```
User uploads PDF files
├── process_button clicked
├── Reset session state
└── PDF Processing
    ├── process_uploaded_pdfs() [pdf_processor.py]
    │   ├── Create temporary directory
    │   ├── Save uploaded files
    │   └── Process each PDF
    │       ├── process_single_pdf()
    │       │   ├── Extract text with PyMuPDF
    │       │   ├── Extract images with Mistral Vision
    │       │   ├── Tag chunks with attribute dictionary
    │       │   └── Create Document objects
    │       └── Merge all documents
    └── Vector Store Setup
        ├── setup_vector_store() [vector_store.py]
        │   ├── get_embedding_function() - Initialize embeddings
        │   ├── Create Chroma vector store
        │   ├── Add documents to store
        │   └── Create retriever
        └── Create extraction chains
            ├── create_pdf_extraction_chain() [llm_interface.py]
            │   ├── RunnableParallel with context lambda:
            │   │   └── context=lambda x: format_docs(retrieve_and_log_chunks(retriever, x['extraction_instructions'], x['attribute_key']))
            │   ├── Uses retrieve_and_log_chunks() with create_enhanced_search_queries()
            │   └── Enhanced search for better chunk retrieval
            ├── create_web_extraction_chain() [llm_interface.py]
            └── create_numind_extraction_chain() [llm_interface.py]
```

#### 3.2 Three-Stage Extraction Process

##### Stage 1: Web Data Extraction
```
if part_number provided:
├── scrape_website_table_html() [llm_interface.py]
│   ├── AsyncWebCrawler setup
│   ├── Browser configuration
│   ├── Crawl supplier websites
│   ├── Extract table HTML
│   └── Clean and return HTML
└── Web Extraction Loop
    ├── For each attribute in prompts_to_run:
    │   ├── _invoke_chain_and_process() [llm_interface.py]
    │   │   ├── Call web_chain with HTML data
    │   │   ├── Process LLM response
    │   │   └── Extract JSON result
    │   ├── extract_json_from_string() - Parse response
    │   ├── Check for "NOT FOUND" or errors
    │   └── Store intermediate results
    └── Queue failed extractions for Stage 2
```

##### Stage 2: NuMind Structured Extraction
```
if web extraction failed or no web data:
├── Check if NuMind chain available
├── extract_with_numind_using_schema() [llm_interface.py]
│   ├── Get custom schema from numind_schema_config.py
│   ├── Call NuMind API with file data
│   └── Extract structured results
└── Process NuMind Results
    ├── For each attribute needing fallback:
    │   ├── extract_specific_attribute_from_numind_result()
    │   └── Update intermediate results
    └── If NuMind fails, fallback to PDF extraction
```

##### Stage 3: Final PDF Fallback
```
For attributes still not found:
├── Enhanced PDF extraction
│   ├── retrieve_and_log_chunks() [llm_interface.py]
│   │   ├── create_enhanced_search_queries() - Generate multiple search queries
│   │   │   ├── Use attribute dictionary values
│   │   │   ├── Add attribute-specific synonyms
│   │   │   ├── Create combined queries
│   │   │   └── Limit to 20 unique queries
│   │   ├── Query vector store with each enhanced query
│   │   ├── Collect unique chunks
│   │   └── Limit to 10 chunks maximum
│   ├── _invoke_chain_and_process() with enhanced prompts
│   │   ├── Calls PDF chain with context lambda
│   │   └── context=lambda x: format_docs(retrieve_and_log_chunks(...))
│   └── Parse and validate results
└── Rollback Logic
    ├── Preserve original values if Stage 3 fails
    ├── Confirm "none" responses are correct
    └── Update final results
```

### 4. MANUAL RECHECK SYSTEM
```
User selects attributes for recheck
├── Manual recheck loop
│   ├── Enhanced prompts for thorough search
│   ├── retrieve_and_log_chunks() with create_enhanced_search_queries()
│   │   ├── Generate 20+ enhanced search queries
│   │   ├── Use attribute dictionary and synonyms
│   │   └── Retrieve up to 15 chunks maximum
│   ├── _invoke_chain_and_process()
│   │   ├── Calls PDF chain with context lambda
│   │   └── context=lambda x: format_docs(retrieve_and_log_chunks(...))
│   └── Update results with rollback logic
└── Display updated results
```

### 5. RESULT DISPLAY & EVALUATION
```
Display extraction results
├── Card-based UI showing:
│   ├── Attribute name
│   ├── Extracted value
│   ├── Source indicator (Web/NuMind/PDF)
│   └── Success/failure status
├── Metrics calculation
└── Export functionality
```

### 6. DEBUG & MONITORING SYSTEM
```
debug_logger.py:DebugLogger
├── Comprehensive logging
│   ├── Function calls and returns
│   ├── LLM requests and responses
│   ├── PDF processing steps
│   ├── Web scraping activities
│   ├── Extraction steps
│   └── Session state changes
├── Performance tracking
└── Error handling and reporting
```

### 7. CONFIGURATION & UTILITIES

#### Configuration Management
```
config.py
├── API keys (GROQ_API_KEY)
├── Model settings (LLM_MODEL_NAME, EMBEDDING_MODEL_NAME)
├── Vector store settings (CHROMA_PERSIST_DIRECTORY)
├── Embedding settings (EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE)
└── Retriever settings (RETRIEVER_K, VECTOR_SIMILARITY_THRESHOLD)
```

#### Utility Functions
```
utils/thinking_log_component.py
├── Real-time progress display
├── Animated thinking indicator
└── Log content display

debug_interface.py
├── Debug log parsing
├── Real-time monitoring
└── Session state logging
```

### 8. EXTRACTION PROMPTS SYSTEM
```
extraction_prompts.py & extraction_prompts_web.py
├── Material Properties prompts
├── Physical/Mechanical Attributes prompts
├── Sealing & Environmental prompts
├── Terminals & Connections prompts
├── Assembly & Type prompts
└── Specialized Attributes prompts
```

### 9. VECTOR STORE & EMBEDDING SYSTEM
```
vector_store.py
├── HuggingFaceAPIEmbeddings class
│   ├── Batch processing
│   ├── Text length limiting
│   ├── Retry logic
│   └── Fallback mechanisms
├── Chroma vector store setup
├── ThresholdRetriever class
└── Document retrieval functions
```

### 10. ENHANCED SEARCH QUERY SYSTEM
```
llm_interface.py
├── create_enhanced_search_queries() [llm_interface.py]
│   ├── Use attribute dictionary values
│   ├── Add attribute-specific synonyms
│   │   ├── Material properties (PA, PBT, GF, etc.)
│   │   ├── Physical attributes (height, length, width)
│   │   ├── Electrical properties (contact systems, TPA, CPA)
│   │   └── Environmental properties (temperature, sealing)
│   ├── Create combined queries
│   └── Limit to 20 unique queries
├── retrieve_and_log_chunks() [llm_interface.py]
│   ├── Call create_enhanced_search_queries()
│   ├── Query vector store with each enhanced query
│   ├── Collect unique chunks by content hash
│   ├── Limit to 10 chunks maximum
│   └── Log retrieval process for debugging
└── Integration with PDF extraction chain
    ├── Used in create_pdf_extraction_chain()
    │   └── context=lambda x: format_docs(retrieve_and_log_chunks(retriever, x['extraction_instructions'], x['attribute_key']))
    ├── Used in manual recheck system
    └── Provides better chunk retrieval for LLM processing
```

### 11. COMPLETE EXECUTION SEQUENCE

```
1. User starts application (app.py:main())
2. User navigates to extraction page
3. User uploads PDF files
4. User clicks "Process Uploaded Documents"
5. PDF processing begins:
   ├── Text extraction with PyMuPDF
   ├── Image processing with Mistral Vision
   ├── Chunk creation and tagging
   └── Vector store indexing
6. User enters part number (optional)
7. Three-stage extraction begins:
   ├── Stage 1: Web scraping and extraction
   ├── Stage 2: NuMind structured extraction
   └── Stage 3: Enhanced PDF fallback
8. Results displayed in card format
9. User can perform manual rechecks
10. User can export results
11. Debug logging throughout entire process
```

### 12. ERROR HANDLING & FALLBACKS

```
Error Handling Strategy:
├── API failures → Fallback to local models
├── Web scraping failures → PDF-only extraction
├── NuMind failures → PDF extraction
├── LLM errors → Retry with different prompts
├── Vector store errors → Rebuild index
└── Configuration errors → Use defaults
```

### 13. PERFORMANCE OPTIMIZATIONS

```
Performance Features:
├── Caching with @st.cache_resource
├── Batch processing for embeddings
├── Async operations for web scraping
├── Thread pool for PDF processing
├── Vector similarity thresholds
└── Configurable timeouts and retries
```

This diagram represents the complete function execution flow from application startup through document processing, extraction, and result display, including all error handling and optimization strategies. 