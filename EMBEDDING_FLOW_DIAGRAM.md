# EMBEDDING FLOW DIAGRAM

## Overview
This diagram shows WHEN embeddings happen in the LEOPARTS system, including all phases from initialization to retrieval.

```mermaid
graph TD
    %% INITIALIZATION PHASE
    A[System Startup] --> B[Initialize Embedding Function]
    B --> C{Use API Embeddings?}
    C -->|Yes| D[Initialize HuggingFaceAPIEmbeddings]
    C -->|No| E[Initialize Local SentenceTransformer]
    
    D --> F[Test API Connection]
    E --> F
    F --> G{Test Successful?}
    G -->|No| H[Return None - System Fails]
    G -->|Yes| I[Embedding Function Ready]
    
    %% DOCUMENT PROCESSING PHASE
    J[User Uploads PDFs] --> K[PDF Processing with Mistral Vision]
    K --> L[Extract Text from PDF Pages]
    L --> M[Create Document Objects]
    M --> N[Tag Documents with Attribute Dictionary]
    
    %% VECTOR STORE SETUP PHASE
    N --> O[Setup Vector Store]
    O --> P[Chroma.from_documents()]
    P --> Q[EMBEDDING PHASE 1: Document Indexing]
    Q --> R[embed_documents() - Batch Processing]
    R --> S[Process Documents in Batches]
    S --> T[Send to HuggingFace API]
    T --> U[Receive Embeddings]
    U --> V[Store in Chroma Vector Store]
    V --> W[Vector Store Ready]
    
    %% RETRIEVAL PHASE
    X[User Query] --> Y[EMBEDDING PHASE 2: Query Embedding]
    Y --> Z[embed_query() - Single Query]
    Z --> AA[Send Query to HuggingFace API]
    AA --> BB[Receive Query Embedding]
    BB --> CC[Similarity Search in Vector Store]
    CC --> DD[Return Relevant Documents]
    
    %% FALLBACK SCENARIOS
    R -->|Batch Fails| EE[Fallback: embed_documents_fallback()]
    EE --> FF[Process Documents Individually]
    FF --> GG[Send Individual Requests to API]
    GG --> U
    
    Z -->|Query Fails| HH[Return Zero Vector [0.0] * 768]
    HH --> CC
    
    %% STYLING
    classDef embeddingPhase fill:#ff9999,stroke:#333,stroke-width:2px
    classDef apiCall fill:#99ccff,stroke:#333,stroke-width:2px
    classDef vectorStore fill:#99ff99,stroke:#333,stroke-width:2px
    classDef error fill:#ffcc99,stroke:#333,stroke-width:2px
    
    class Q,Y embeddingPhase
    class T,AA apiCall
    class V,W vectorStore
    class H,HH error
```

## DETAILED EMBEDDING TRIGGERS

### 1. **INITIALIZATION EMBEDDING** (System Startup)
- **Trigger**: `get_embedding_function()` called during app initialization
- **Method**: `embed_query("test")`
- **Purpose**: Test API connection and validate embedding function
- **Location**: `vector_store.py:238`

### 2. **DOCUMENT INDEXING EMBEDDING** (PDF Processing)
- **Trigger**: `setup_vector_store()` called after PDF processing
- **Method**: `embed_documents()` - Batch processing
- **Purpose**: Convert all extracted text chunks to embeddings for vector storage
- **Location**: `vector_store.py:29-122`
- **Batch Size**: Configurable (default: 5 documents per batch)
- **Text Length Limit**: 4000 characters per text

### 3. **QUERY EMBEDDING** (User Search)
- **Trigger**: `SimpleRetriever.retrieve()` or `ThresholdRetriever.invoke()`
- **Method**: `embed_query()` - Single query
- **Purpose**: Convert user query to embedding for similarity search
- **Location**: `vector_store.py:174-222`

### 4. **CHATBOT QUERY EMBEDDING** (Alternative Path)
- **Trigger**: `get_query_embedding()` in chatbot interface
- **Method**: Direct API call or local model encoding
- **Purpose**: Embed user queries for chatbot functionality
- **Location**: `pages/chatbot.py:197-226`

## EMBEDDING CONFIGURATION

### API Configuration
```python
EMBEDDING_API_URL = "https://hbaananou-embedder-model.hf.space/embed"
EMBEDDING_BATCH_SIZE = 5
EMBEDDING_TIMEOUT = 120
EMBEDDING_MAX_TEXT_LENGTH = 4000
EMBEDDING_DIMENSIONS = 1024
```

### Fallback Behavior
- **Batch Processing Fails**: Falls back to individual document processing
- **API Timeout**: Retries up to 3 times with 1-second delays
- **API Failure**: Returns zero vector `[0.0] * 768` as fallback
- **Text Too Long**: Truncates to `EMBEDDING_MAX_TEXT_LENGTH`

## EMBEDDING TIMELINE

### Phase 1: System Initialization
```
1. App starts → Initialize embedding function
2. Test API connection → embed_query("test")
3. Validate response → Embedding function ready
```

### Phase 2: Document Processing
```
1. User uploads PDFs → Process with Mistral Vision
2. Extract text → Create Document objects
3. Tag with attributes → Setup vector store
4. EMBED ALL DOCUMENTS → embed_documents() in batches
5. Store in Chroma → Vector store ready
```

### Phase 3: User Queries
```
1. User submits query → embed_query()
2. Send to API → Receive embedding
3. Similarity search → Return relevant documents
```

## KEY EMBEDDING POINTS

| Phase | When | Method | Purpose | Location |
|-------|------|--------|---------|----------|
| **Init** | App startup | `embed_query("test")` | Test API | `vector_store.py:238` |
| **Indexing** | After PDF processing | `embed_documents()` | Index all docs | `vector_store.py:29` |
| **Retrieval** | User query | `embed_query()` | Search query | `vector_store.py:174` |
| **Chatbot** | Chat interface | `get_query_embedding()` | Chat queries | `pages/chatbot.py:197` |

## ERROR HANDLING

### Embedding Failures
- **API Timeout**: Retry logic with exponential backoff
- **Batch Failure**: Fallback to individual processing
- **Complete Failure**: Return zero vector fallback
- **Invalid Response**: Log error and return fallback

### Performance Optimizations
- **Batch Processing**: Process multiple documents together
- **Text Truncation**: Limit text length to prevent API overload
- **Caching**: Use `@st.cache_resource` for embedding function
- **Early Stopping**: Stop processing when enough results found

## EMBEDDING DIMENSIONS

- **Expected**: 1024 dimensions (BAAI/bge-m3 model)
- **Fallback**: 768 dimensions (zero vector)
- **Validation**: Check dimensions during initialization

This diagram shows that embeddings happen at **3 critical points**:
1. **System initialization** (testing)
2. **Document indexing** (bulk processing)
3. **Query processing** (user searches)

Each phase has its own error handling and fallback mechanisms to ensure system reliability. 