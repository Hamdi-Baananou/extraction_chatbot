# COMPREHENSIVE PROCESSING FLOW DIAGRAM

## Overview
This diagram shows the complete processing phase including chunking, from PDF upload through Mistral extraction, tagging, and vector store storage.

```mermaid
graph TD
    %% PDF UPLOAD PHASE
    A[User Uploads PDFs] --> B[Streamlit File Uploader]
    B --> C[Save to Temporary Directory]
    C --> D[temp_pdf/ directory]
    
    %% MISTRAL VISION EXTRACTION PHASE
    D --> E[Initialize Mistral Vision Client]
    E --> F[Process Each PDF File]
    F --> G[Convert PDF Page to Image]
    G --> H[High-Resolution Image (300 DPI)]
    H --> I[Encode Image to Base64]
    I --> J[Send to Mistral Vision API]
    J --> K[Extract Text from Image]
    K --> L[Get Markdown Formatted Text]
    
    %% CHUNKING PHASE (SPECIAL CASE)
    L --> M{CHUNKING STRATEGY}
    M -->|CURRENT APPROACH| N[Page-Level Chunking]
    M -->|ALTERNATIVE| O[Text Splitter Chunking]
    
    N --> P[Treat Each Page as One Document]
    O --> Q[RecursiveCharacterTextSplitter]
    Q --> R[CHUNK_SIZE = 1000]
    Q --> S[CHUNK_OVERLAP = 200]
    
    %% TAGGING PHASE
    P --> T[Tag with Attribute Dictionary]
    S --> T
    T --> U[Load Attribute Dictionary]
    U --> V[Build Regex Patterns]
    V --> W[Apply Regex to Text]
    W --> X[Extract Attribute Matches]
    X --> Y[Create Metadata Tags]
    
    %% DOCUMENT CREATION PHASE
    Y --> Z[Create Document Objects]
    Z --> AA[Document Structure]
    AA --> BB[page_content: Extracted Text]
    AA --> CC[metadata: Source + Page + Tags]
    
    %% VECTOR STORE PROCESSING PHASE
    BB --> DD[Vector Store Setup]
    CC --> DD
    DD --> EE[Initialize Embedding Function]
    EE --> FF[HuggingFace API Embeddings]
    FF --> GG[Embed All Documents]
    GG --> HH[Batch Processing]
    HH --> II[EMBEDDING_BATCH_SIZE = 5]
    HH --> JJ[EMBEDDING_MAX_TEXT_LENGTH = 4000]
    
    %% STORAGE PHASE
    II --> KK[Store in Chroma Vector Store]
    JJ --> KK
    KK --> LL[Collection: pdf_qa_prod_collection]
    LL --> MM[Persist to ./chroma_db_prod/]
    MM --> NN[Vector Store Ready]
    
    %% CLEANUP PHASE
    DD --> OO[Cleanup Temporary Files]
    OO --> PP[Remove temp_pdf/ files]
    
    %% RETRIEVAL PHASE
    NN --> QQ[User Query]
    QQ --> RR[Embed Query]
    RR --> SS[Similarity Search]
    SS --> TT[Return Relevant Chunks]
    
    %% STYLING
    classDef uploadPhase fill:#e1f5fe,stroke:#333,stroke-width:2px
    classDef extractionPhase fill:#f3e5f5,stroke:#333,stroke-width:2px
    classDef chunkingPhase fill:#fff3e0,stroke:#333,stroke-width:2px
    classDef taggingPhase fill:#e8f5e8,stroke:#333,stroke-width:2px
    classDef vectorPhase fill:#fce4ec,stroke:#333,stroke-width:2px
    classDef storagePhase fill:#f1f8e9,stroke:#333,stroke-width:2px
    classDef retrievalPhase fill:#e0f2f1,stroke:#333,stroke-width:2px
    
    class A,B,C,D uploadPhase
    class E,F,G,H,I,J,K,L extractionPhase
    class M,N,O,P,Q,R,S chunkingPhase
    class T,U,V,W,X,Y,Z,AA,BB,CC taggingPhase
    class DD,EE,FF,GG,HH,II,JJ vectorPhase
    class KK,LL,MM,NN storagePhase
    class QQ,RR,SS,TT retrievalPhase
```

## DETAILED PROCESSING PHASES

### 1. **PDF UPLOAD PHASE**
- **Trigger**: User uploads PDF files via Streamlit
- **Storage**: Temporary files in `temp_pdf/` directory
- **Processing**: Parallel processing with ThreadPoolExecutor
- **Code**: `pages/extraction_attributs.py:640-650`

### 2. **MISTRAL VISION EXTRACTION PHASE**
- **Image Conversion**: PDF pages → High-resolution images (300 DPI)
- **API Call**: Send images to Mistral Vision API
- **Text Extraction**: Extract structured Markdown text
- **Model**: `mistral-small-latest` (configurable)
- **Code**: `pdf_processor.py:110-240`

### 3. **CHUNKING PHASE** (Special Case)
- **Current Strategy**: **Page-Level Chunking**
  - Each PDF page becomes one document
  - No text splitting within pages
  - Preserves page context and structure
- **Alternative Strategy**: Text Splitter (commented out)
  - `RecursiveCharacterTextSplitter`
  - `CHUNK_SIZE = 1000`
  - `CHUNK_OVERLAP = 200`
- **Code**: `pdf_processor.py:201-210`

### 4. **TAGGING PHASE**
- **Attribute Dictionary**: Load from `attribute_dictionary.json`
- **Regex Building**: Create patterns for each attribute
- **Tagging Process**: Apply regex to extracted text
- **Metadata Creation**: Store matches as metadata tags
- **Code**: `pdf_processor.py:54-82`

### 5. **DOCUMENT CREATION PHASE**
- **Structure**: LangChain Document objects
- **Content**: Extracted text from Mistral Vision
- **Metadata**: Source file, page number, attribute tags
- **Code**: `pdf_processor.py:202-210`

### 6. **VECTOR STORE PROCESSING PHASE**
- **Embedding Function**: HuggingFace API embeddings
- **Batch Processing**: Process documents in batches of 5
- **Text Limits**: Maximum 4000 characters per text
- **Timeout**: 120 seconds for API calls
- **Code**: `vector_store.py:29-122`

### 7. **STORAGE PHASE**
- **Vector Store**: Chroma with persistence
- **Collection**: `pdf_qa_prod_collection`
- **Directory**: `./chroma_db_prod/`
- **Persistence**: Permanent storage across sessions
- **Code**: `vector_store.py:500-535`

## CHUNKING STRATEGY DETAILS

### **Current Approach: Page-Level Chunking**
```python
# Instead of splitting into chunks, treat the whole page as one document
chunk_doc = Document(
    page_content=page_content,  # Full page text
    metadata={
        'source': file_basename,
        'page': page_num + 1,
        **chunk_tags  # All attribute tags
    }
)
```

### **Alternative Approach: Text Splitter** (Commented Out)
```python
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=CHUNK_OVERLAP
# )
# chunks = text_splitter.split_text(page_content)
```

## PROCESSING CONFIGURATION

### **Mistral Vision Configuration**
```python
VISION_MODEL_NAME = "mistral-small-latest"
# High-resolution image conversion
pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
```

### **Embedding Configuration**
```python
EMBEDDING_BATCH_SIZE = 5
EMBEDDING_MAX_TEXT_LENGTH = 4000
EMBEDDING_TIMEOUT = 120
EMBEDDING_DIMENSIONS = 1024
```

### **Vector Store Configuration**
```python
CHROMA_PERSIST_DIRECTORY = "./chroma_db_prod"
COLLECTION_NAME = "pdf_qa_prod_collection"
```

## PROCESSING TIMELINE

### **Phase 1: Upload & Extraction**
```
1. User uploads PDFs → Save to temp_pdf/
2. Process each PDF → Convert pages to images
3. Send to Mistral Vision → Extract structured text
4. Create page-level documents → Tag with attributes
```

### **Phase 2: Vector Processing**
```
1. Initialize embedding function → Test API connection
2. Process documents in batches → Embed all text
3. Store in Chroma → Persist to ./chroma_db_prod/
4. Clean up temp files → Remove temp_pdf/ files
```

### **Phase 3: Retrieval Ready**
```
1. Vector store ready → Enable similarity search
2. User queries → Embed and search
3. Return relevant chunks → Based on similarity
```

## KEY PROCESSING FEATURES

### **Parallel Processing**
- **ThreadPoolExecutor**: Process multiple PDFs simultaneously
- **Max Workers**: `min(len(saved_file_paths), 4)`
- **Async Processing**: Non-blocking PDF processing

### **Error Handling**
- **API Timeouts**: Retry logic with delays
- **Failed Extractions**: Log warnings, continue processing
- **Cleanup**: Automatic temporary file removal

### **Performance Optimizations**
- **Batch Processing**: Embed multiple documents together
- **Text Truncation**: Limit text length to prevent API overload
- **Early Stopping**: Stop when enough results found
- **Caching**: Use `@st.cache_resource` for embedding function

## DATA FLOW SUMMARY

```
PDF Upload → Temp Storage → Mistral Vision → Page-Level Chunking → 
Tagging → Document Creation → Embedding → Vector Store → 
Permanent Storage → Retrieval Ready
```

## IMPORTANT NOTES

### **1. No Traditional Chunking**
- System uses **page-level chunking** instead of text splitting
- Each PDF page becomes one document
- Preserves page context and structure

### **2. Attribute Tagging**
- All extracted text is tagged with attribute dictionary
- Tags stored in document metadata
- Enables tag-aware retrieval

### **3. Permanent Storage**
- Vector store persists across sessions
- Embeddings enable semantic search
- No raw text files stored

### **4. Parallel Processing**
- Multiple PDFs processed simultaneously
- Non-blocking async operations
- Automatic cleanup of temporary files

This comprehensive flow shows how your system processes PDFs from upload through extraction, chunking, tagging, and permanent storage in the vector database. 