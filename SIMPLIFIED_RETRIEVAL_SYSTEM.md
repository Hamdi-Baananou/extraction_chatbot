# Simplified Retrieval System

## Overview
The retrieval system has been simplified to use **only similarity search with tagging**, removing the complex query search mechanism.

## How It Works

### 1. Tagging (PDF Processing)
- Each chunk is tagged with keywords from the attribute dictionary during PDF processing
- Tags are stored in document metadata
- **This remains completely untouched**

### 2. Retrieval (Simplified)
- **Single similarity search** using the provided query
- **Tag filtering**: Only chunks with the specific attribute tag are returned
- **Part number filtering**: Optional filtering by part number
- **Fallback**: If no tagged chunks found, falls back to semantic similarity only

## Usage Examples

```python
# Basic similarity search
chunks = retriever.retrieve("material name")

# Similarity search + tag filtering
chunks = retriever.retrieve("material name", attribute_key="Material Name")

# Similarity search + part number filtering
chunks = retriever.retrieve("material name", part_number="ABC123")

# Full retrieval with all filters
chunks = retriever.retrieve("material name", attribute_key="Material Name", part_number="ABC123")
```

## Key Changes Made

### ✅ Removed Complexity
- ❌ Multiple query generation (`_create_queries`)
- ❌ Query search with enhanced queries
- ❌ Deduplication logic (`_hash_chunk`)
- ❌ Attribute dictionary loading (`_load_attribute_dictionary`)

### ✅ Kept Simple
- ✅ Single similarity search
- ✅ Tag filtering
- ✅ Part number filtering
- ✅ Fallback mechanisms (untouched)
- ✅ Threshold filtering

## Fallback Mechanisms (Untouched)
All fallback mechanisms between stages remain completely untouched:
- Stage 1 → Stage 2 fallback
- Stage 2 → NuMind fallback
- Retrieval fallback (tagged → semantic similarity)

## Benefits
1. **Simpler**: No complex query generation
2. **Faster**: Single similarity search instead of multiple queries
3. **More predictable**: Direct similarity matching
4. **Maintainable**: Less code complexity
5. **Reliable**: Tag-based filtering ensures relevance 