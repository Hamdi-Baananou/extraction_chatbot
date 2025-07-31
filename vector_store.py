# vector_store.py
from typing import List, Optional
from loguru import logger
import os
import time
import requests
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from chromadb import Client as ChromaClient
from langchain.embeddings.base import Embeddings

import config # Import configuration
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# --- Custom Hugging Face API Embeddings ---
class HuggingFaceAPIEmbeddings(Embeddings):
    """Custom embeddings class that uses Hugging Face API instead of local model."""
    
    def __init__(self, api_url: str = "https://hbaananou-embedder-model.hf.space/embed"):
        self.api_url = api_url
        logger.info(f"Initialized HuggingFace API embeddings with URL: {api_url}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Hugging Face API with batching and text length limiting."""
        if not texts:
            return []
        
        # Batch size - adjust based on your API's capacity
        batch_size = config.EMBEDDING_BATCH_SIZE
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        all_embeddings = []
        
        # Pre-process texts to limit length
        processed_texts = []
        for text in texts:
            if len(text) > max_text_length:
                logger.warning(f"Truncating text from {len(text)} to {max_text_length} characters")
                processed_text = text[:max_text_length]
            else:
                processed_text = text
            processed_texts.append(processed_text)
        
        # Process texts in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(processed_texts) + batch_size - 1)//batch_size
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} with {len(batch_texts)} texts")
            
            # Calculate total characters in this batch
            total_chars = sum(len(text) for text in batch_texts)
            logger.debug(f"Batch {batch_num} total characters: {total_chars}")
            
            # Retry logic for failed batches
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Prepare the request payload
                    payload = {"texts": batch_texts}
                    
                    # Make the API request with increased timeout for batches
                    response = requests.post(
                        self.api_url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=config.EMBEDDING_TIMEOUT  # Configurable timeout for batch processing
                    )
                    
                    # Check if the request was successful
                    response.raise_for_status()
                    
                    # Parse the response
                    result = response.json()
                    
                    # Extract embeddings from the response
                    # Handle different API response formats
                    if "embeddings" in result:
                        batch_embeddings = result["embeddings"]
                    elif "vectors" in result:
                        batch_embeddings = result["vectors"]
                    elif isinstance(result, list):
                        # If the API returns embeddings directly as a list
                        batch_embeddings = result
                    else:
                        # Try to find embeddings in the response structure
                        batch_embeddings = result.get("data", result.get("result", result))
                        if not isinstance(batch_embeddings, list):
                            raise ValueError(f"Unexpected API response format: {result}")
                    
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"Successfully embedded batch {batch_num} with {len(batch_texts)} documents")
                    
                    # Success, break out of retry loop
                    break
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Batch {batch_num} timed out (attempt {retry + 1}/{max_retries})")
                    if retry == max_retries - 1:
                        logger.error(f"Batch {batch_num} failed after {max_retries} timeout attempts")
                        raise
                    time.sleep(1)  # Wait before retry
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Batch {batch_num} failed (attempt {retry + 1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts")
                        raise
                    time.sleep(1)  # Wait before retry
                    
                except Exception as e:
                    logger.error(f"Unexpected error in batch {batch_num}: {e}")
                    raise
        
        logger.info(f"Successfully embedded {len(processed_texts)} documents in {len(all_embeddings)} batches")
        return all_embeddings

    def embed_documents_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method for individual document processing."""
        if not texts:
            return []
        
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        all_embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Limit text length
                if len(text) > max_text_length:
                    logger.warning(f"Truncating text {i+1} from {len(text)} to {max_text_length} characters")
                    processed_text = text[:max_text_length]
                else:
                    processed_text = text
                
                # Process single document
                payload = {"texts": [processed_text]}
                
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=config.EMBEDDING_TIMEOUT
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract single embedding
                if "embeddings" in result:
                    embedding = result["embeddings"][0]
                elif "vectors" in result:
                    embedding = result["vectors"][0]
                elif isinstance(result, list):
                    embedding = result[0]
                else:
                    embedding = result.get("data", result.get("result", result))[0]
                
                all_embeddings.append(embedding)
                logger.debug(f"Successfully embedded document {i+1}/{len(texts)}")
                
            except Exception as e:
                logger.error(f"Failed to embed document {i+1}: {e}")
                # Return zero vector as fallback
                all_embeddings.append([0.0] * 768)  # Assuming 768-dimensional embeddings
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not text:
            return []
        
        max_text_length = config.EMBEDDING_MAX_TEXT_LENGTH
        
        # Limit text length
        if len(text) > max_text_length:
            logger.warning(f"Truncating query from {len(text)} to {max_text_length} characters")
            processed_text = text[:max_text_length]
        else:
            processed_text = text
        
        try:
            # Prepare the request payload
            payload = {"texts": [processed_text]}
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=config.EMBEDDING_TIMEOUT
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract embedding from the response
            if "embeddings" in result:
                embedding = result["embeddings"][0]
            elif "vectors" in result:
                embedding = result["vectors"][0]
            elif isinstance(result, list):
                embedding = result[0]
            else:
                embedding = result.get("data", result.get("result", result))[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            # Return zero vector as fallback
            return [0.0] * 768  # Assuming 768-dimensional embeddings

# --- Unified Simple Retriever (NEW) ---
class SimpleRetriever:
    """
    Centralized retrieval system that replaces all confusing retrieval methods.
    
    This ONE method handles all retrieval cases:
    - Simple semantic search
    - Attribute-specific search with enhanced queries
    - Part number filtering
    - Threshold filtering
    - Early stopping for performance
    """
    
    def __init__(self, vectorstore, config):
        self.vectorstore = vectorstore
        self.config = config
        self.attribute_dict = self._load_attribute_dictionary()
    
    def _load_attribute_dictionary(self):
        """Load the attribute dictionary from JSON file."""
        try:
            import json
            import os
            dict_path = os.path.join(os.path.dirname(__file__), 'attribute_dictionary.json')
            with open(dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load attribute dictionary: {e}")
            return {}
    
    def _hash_chunk(self, chunk):
        """Hash chunk content and metadata for deduplication."""
        import hashlib
        import json
        m = hashlib.sha256()
        m.update(chunk.page_content.encode('utf-8'))
        m.update(json.dumps(chunk.metadata, sort_keys=True).encode('utf-8'))
        return m.hexdigest()
    
    def retrieve(self, query: str, attribute_key: str = None, 
                part_number: str = None, max_queries: int = 3) -> List[Document]:
        """
        ONE method that replaces ALL your current retrieval methods.
        
        Args:
            query: The search query
            attribute_key: Optional attribute for enhanced search
            part_number: Optional part number for filtering
            max_queries: Maximum queries to try (default 3 instead of 20)
        
        Returns:
            List of relevant documents (max 5)
        
        Examples:
            # Simple retrieval (like ThresholdRetriever.invoke)
            chunks = retriever.retrieve("material name")
            
            # PDF retrieval with part number (like fetch_chunks)
            chunks = retriever.retrieve("material name", part_number="ABC123")
            
            # Enhanced retrieval (like retrieve_and_log_chunks)
            chunks = retriever.retrieve("material name", attribute_key="Material Name", max_queries=5)
        """
        logger.info(f"🔍 SIMPLE RETRIEVAL: query='{query}', attribute='{attribute_key}', part_number='{part_number}'")
        
        # 1. Create optimized queries (max 3 instead of 20)
        queries = self._create_queries(query, attribute_key, max_queries)
        logger.info(f"📋 Using {len(queries)} queries: {queries}")
        
        # 2. Retrieve with threshold filtering
        all_chunks = []
        seen_chunks = set()
        
        for i, search_query in enumerate(queries):
            logger.debug(f"🔍 Search {i+1}/{len(queries)}: '{search_query}'")
            
            try:
                chunks = self._get_chunks_with_threshold(search_query)
                
                if chunks:
                    # Add unique chunks only
                    for chunk in chunks:
                        chunk_hash = self._hash_chunk(chunk)
                        if chunk_hash not in seen_chunks:
                            seen_chunks.add(chunk_hash)
                            all_chunks.append(chunk)
                            logger.debug(f"  ✅ Added unique chunk from query '{search_query}'")
                
                # Early stopping if we have enough chunks
                if len(all_chunks) >= 5:
                    logger.info(f"📊 Early stopping: Found {len(all_chunks)} chunks after {i+1} queries")
                    break
                    
            except Exception as e:
                logger.warning(f"❌ Query '{search_query}' failed: {e}")
                continue
        
        # 3. Apply part number filtering if needed
        if part_number:
            all_chunks = self._filter_by_part_number(all_chunks, part_number)
            logger.info(f"🔍 Part number filtering: {len(all_chunks)} chunks after filtering")
        
        # 4. Limit total chunks to avoid overwhelming the LLM
        max_chunks = 5
        if len(all_chunks) > max_chunks:
            logger.info(f"📊 Limiting chunks from {len(all_chunks)} to {max_chunks}")
            all_chunks = all_chunks[:max_chunks]
        
        logger.info(f"✅ Retrieved {len(all_chunks)} chunks for query '{query}'")
        return all_chunks
    
    def _create_queries(self, query: str, attribute_key: str, max_queries: int) -> List[str]:
        """Create optimized queries (max 3 instead of 20)."""
        queries = [query]  # Base query
        
        if attribute_key and attribute_key in self.attribute_dict:
            # Add only 1-2 most relevant dictionary values
            dict_values = self.attribute_dict[attribute_key]
            for value in dict_values[:2]:  # Only top 2 values
                if isinstance(value, str) and len(value) > 1:
                    queries.append(value)
        
        # Remove duplicates and limit to max_queries
        unique_queries = list(dict.fromkeys(queries))[:max_queries]
        return unique_queries
    
    def _get_chunks_with_threshold(self, query: str) -> List[Document]:
        """Get chunks with similarity threshold filtering."""
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.config.RETRIEVER_K
        )
        
        filtered_docs = []
        for doc, score in docs_and_scores:
            if score >= self.config.VECTOR_SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                logger.debug(f"Chunk passed threshold (score: {score:.3f}): {doc.page_content[:100]}")
            else:
                logger.debug(f"Chunk below threshold (score: {score:.3f}): {doc.page_content[:100]}")
        
        logger.info(f"Retrieved {len(docs_and_scores)} chunks, {len(filtered_docs)} passed threshold {self.config.VECTOR_SIMILARITY_THRESHOLD}")
        return filtered_docs
    
    def _filter_by_part_number(self, chunks: List[Document], part_number: str) -> List[Document]:
        """Filter chunks by part number if available."""
        filtered = []
        for chunk in chunks:
            chunk_part_number = chunk.metadata.get("part_number", "")
            
            # Check part number match (if part_number is provided AND stored in metadata)
            part_number_match = True
            if part_number and chunk_part_number:
                part_number_match = str(chunk_part_number).strip() == str(part_number).strip()
                logger.debug(f"Part number check: chunk='{chunk_part_number}' vs query='{part_number}' -> {part_number_match}")
            elif part_number and not chunk_part_number:
                # If user provided part number but chunk doesn't have it, skip the check
                logger.debug(f"Part number provided '{part_number}' but chunk has no part_number field, skipping part number check")
                part_number_match = True  # Allow through since we can't verify
            
            if part_number_match:
                filtered.append(chunk)
                logger.debug(f"Chunk accepted: part_number={chunk_part_number}")
            else:
                logger.debug(f"Chunk rejected: part_number_match={part_number_match}")
        
        return filtered

# --- Vector Store Setup Functions ---
@logger.catch(reraise=True)
def setup_vector_store(
    documents: List[Document],
    embedding_function,
) -> Optional[SimpleRetriever]:
    """
    Sets up a Chroma vector store with the provided documents and embedding function.
    Args:
        documents: List of documents to add to the vector store.
        embedding_function: The embedding function to use.
    Returns:
        A SimpleRetriever object if successful, otherwise None.
    """
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    if not persist_directory:
        logger.warning("Persistence directory not configured. Cannot setup vector store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for setup_vector_store.")
        return None

    logger.info(f"Setting up vector store '{collection_name}' with {len(documents)} documents...")

    try:
        # Create the vector store with batch processing
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        except Exception as e:
            logger.warning(f"Batch processing failed, trying fallback method: {e}")
            # If batch processing fails, try individual processing
            if hasattr(embedding_function, 'embed_documents_fallback'):
                # Create a temporary embedding function that uses fallback
                class FallbackEmbeddingFunction:
                    def __init__(self, original_function):
                        self.original_function = original_function
                    
                    def embed_documents(self, texts):
                        return self.original_function.embed_documents_fallback(texts)
                    
                    def embed_query(self, text):
                        return self.original_function.embed_query(text)
                
                fallback_embedding = FallbackEmbeddingFunction(embedding_function)
                
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=fallback_embedding,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
            else:
                raise e

        # Ensure persistence after creation/update
        if persist_directory:
            logger.info(f"Persisting vector store to directory: {persist_directory}")
            vector_store.persist() # Explicitly call persist just in case

        logger.success(f"Vector store '{collection_name}' created/updated and persisted successfully.")
        # Return the new SimpleRetriever
        return SimpleRetriever(
            vectorstore=vector_store,
            config=config
        )

    except Exception as e:
        logger.error(f"Failed to setup vector store: {e}", exc_info=True)
        return None

@logger.catch(reraise=True)
def load_existing_vector_store(embedding_function) -> Optional[SimpleRetriever]:
    """
    Loads an existing Chroma vector store from the persistent directory.
    Args:
        embedding_function: The embedding function to use.
    Returns:
        A SimpleRetriever object if the store exists and loads, otherwise None.
    """
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    if not persist_directory:
        logger.warning("Persistence directory not configured. Cannot load existing store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for load_existing_vector_store.")
        return None

    if not os.path.exists(persist_directory):
         logger.warning(f"Persistence directory '{persist_directory}' does not exist. Cannot load.")
         return None

    logger.info(f"Attempting to load existing vector store from: '{persist_directory}', Collection: '{collection_name}'")

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )

        logger.success(f"Successfully loaded vector store '{collection_name}'.")
        return SimpleRetriever(
            vectorstore=vector_store,
            config=config
        )

    except Exception as e:
        logger.warning("Failed to load existing vector store '{}' from '{}': {}".format(collection_name, persist_directory, e), exc_info=False)
        if "does not exist" in str(e).lower():
             logger.warning("Persistent collection '{}' not found in directory '{}'. Cannot load.".format(collection_name, persist_directory))

        return None

# --- Legacy ThresholdRetriever (for backward compatibility) ---
class ThresholdRetriever:
    """Custom retriever that applies similarity threshold filtering."""
    
    def __init__(self, vectorstore, search_kwargs, threshold):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
        self.threshold = threshold
    
    def invoke(self, query: str) -> List[Document]:
        """Get documents with similarity threshold filtering."""
        # Get documents with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.search_kwargs.get("k", 8)
        )
        
        # Filter by threshold
        filtered_docs = []
        for doc, score in docs_and_scores:
            if score >= self.threshold:
                filtered_docs.append(doc)
                logger.debug("Chunk passed threshold (score: {:.3f}): {}".format(score, doc.page_content[:100]))
            else:
                logger.debug("Chunk below threshold (score: {:.3f}): {}".format(score, doc.page_content[:100]))
        
        logger.info("Retrieved {} chunks, {} passed threshold {}".format(len(docs_and_scores), len(filtered_docs), self.threshold))
        return filtered_docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain compatibility method - same as invoke."""
        return self.invoke(query)
    
    async def ainvoke(self, query: str) -> List[Document]:
        """Async version of document retrieval."""
        return self.invoke(query)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async LangChain compatibility method."""
        return self.invoke(query)