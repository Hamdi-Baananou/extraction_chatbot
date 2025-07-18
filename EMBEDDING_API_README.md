# Hugging Face API Embeddings Integration

This project now supports using Hugging Face API embeddings instead of loading the heavy embedding model locally. This is especially useful for Streamlit deployments where memory and loading time are concerns.

## Configuration

### Environment Variables

Set these environment variables to configure the API embeddings:

```bash
# Enable API embeddings (default: true)
USE_API_EMBEDDINGS=true

# Hugging Face API URL (default: https://hbaananou-embedder-model.hf.space/embed)
EMBEDDING_API_URL=https://hbaananou-embedder-model.hf.space/embed

# Embedding dimensions (default: 1024 for BAAI/bge-m3)
EMBEDDING_DIMENSIONS=1024

# Batch size for processing multiple texts (default: 5 for large files)
EMBEDDING_BATCH_SIZE=5

# Timeout for API requests in seconds (default: 120 for large files)
EMBEDDING_TIMEOUT=120

# Maximum text length in characters (default: 4000)
EMBEDDING_MAX_TEXT_LENGTH=4000
```

### Fallback to Local Embeddings

If you want to use local embeddings instead of the API, set:

```bash
USE_API_EMBEDDINGS=false
```

## API Response Format

The API expects to receive requests in this format:

```json
{
  "texts": ["Hello world", "Test sentence"]
}
```

And should return responses in one of these formats:

### Format 1: Direct embeddings array
```json
[
  [0.1, 0.2, 0.3, ...],
  [0.4, 0.5, 0.6, ...]
]
```

### Format 2: Wrapped in embeddings key
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

### Format 3: Wrapped in vectors key
```json
{
  "vectors": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

### Format 4: Wrapped in data/result key
```json
{
  "data": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ]
}
```

## Testing

Run the test script to verify the API integration:

```bash
python test_embedding_api.py
```

## Implementation Details

### Files Modified

1. **`vector_store.py`**: Added `HuggingFaceAPIEmbeddings` class and updated `get_embedding_function()`
2. **`config.py`**: Added API configuration variables
3. **`pages/chatbot.py`**: Updated to use API embeddings instead of local SentenceTransformer
4. **`requirements.txt`**: Updated requests dependency

### Key Classes

#### HuggingFaceAPIEmbeddings

A custom LangChain embeddings class that:
- Makes HTTP requests to the Hugging Face API
- Handles different response formats
- Provides proper error handling and logging
- Implements the required LangChain embeddings interface

#### Usage in Code

```python
from vector_store import get_embedding_function

# Get the embedding function (API or local based on config)
embedding_function = get_embedding_function()

# Use with vector stores
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    collection_name="my_collection"
)
```

## Benefits

1. **Reduced Memory Usage**: No need to load the embedding model in memory
2. **Faster Startup**: No model loading time
3. **Scalability**: Can handle multiple concurrent requests
4. **Flexibility**: Easy to switch between API and local embeddings
5. **Streamlit Compatibility**: Works well with Streamlit's memory constraints
6. **Dimension Validation**: Automatically validates that API returns correct 1024-dimensional embeddings
7. **Batch Processing**: Processes large document sets in configurable batches to avoid timeouts
8. **Configurable Timeouts**: Adjustable timeout settings for different API response times
9. **Text Length Limiting**: Automatically truncates very long texts to prevent API overload
10. **Retry Logic**: Automatic retries with exponential backoff for failed requests
11. **Fallback Processing**: Individual text processing if batch processing fails

## Troubleshooting

### Common Issues

1. **API Timeout**: Increase `EMBEDDING_TIMEOUT` if needed (default: 120 seconds for large files)
2. **Batch Size**: Reduce `EMBEDDING_BATCH_SIZE` if processing large documents (default: 5 for large files)
3. **Text Length**: Reduce `EMBEDDING_MAX_TEXT_LENGTH` if texts are too long (default: 4000 characters)
4. **Response Format**: Ensure your API returns embeddings in one of the supported formats
5. **Network Issues**: Check connectivity to the Hugging Face API endpoint

### Debug Mode

Enable debug logging to see detailed API requests and responses:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Migration from Local Embeddings

If you're migrating from local embeddings:

1. Set `USE_API_EMBEDDINGS=true` in your environment
2. Ensure your API endpoint is accessible
3. Test with the provided test script
4. Update any custom embedding code to use the new `get_embedding_function()`

The integration is designed to be backward compatible, so existing code should work without changes. 