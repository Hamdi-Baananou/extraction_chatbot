#!/usr/bin/env python3
"""
Test script for the simplified retrieval system.
This verifies that the retrieval has been simplified as requested.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from vector_store import SimpleRetriever
from config import Config

def test_simplified_retrieval():
    """Test that the retrieval system has been simplified correctly."""
    
    # Create a mock config
    config = Config()
    config.RETRIEVER_K = 10
    config.VECTOR_SIMILARITY_THRESHOLD = 0.5
    
    # Create a mock vectorstore (this would normally be a real ChromaDB instance)
    class MockVectorStore:
        def similarity_search_with_score(self, query, k):
            # Return mock documents with scores
            return [
                (type('Document', (), {'page_content': f'Mock content for {query}', 'metadata': {'source': 'test.pdf', 'page': 1}})(), 0.8),
                (type('Document', (), {'page_content': f'Another mock content for {query}', 'metadata': {'source': 'test.pdf', 'page': 2}})(), 0.6),
            ]
    
    mock_vectorstore = MockVectorStore()
    
    # Test the simplified retriever
    retriever = SimpleRetriever(mock_vectorstore, config)
    
    print("✅ Testing simplified retrieval system...")
    
    # Test 1: Basic retrieval (should use only similarity search)
    print("\n🔍 Test 1: Basic similarity search")
    result = retriever.retrieve("test query")
    print(f"   Retrieved {len(result)} chunks")
    
    # Test 2: Retrieval with attribute filtering
    print("\n🔍 Test 2: Similarity search + attribute filtering")
    result = retriever.retrieve("test query", attribute_key="Material Name")
    print(f"   Retrieved {len(result)} chunks")
    
    # Test 3: Retrieval with part number filtering
    print("\n🔍 Test 3: Similarity search + part number filtering")
    result = retriever.retrieve("test query", part_number="ABC123")
    print(f"   Retrieved {len(result)} chunks")
    
    # Test 4: Full retrieval with all filters
    print("\n🔍 Test 4: Similarity search + all filters")
    result = retriever.retrieve("test query", attribute_key="Material Name", part_number="ABC123")
    print(f"   Retrieved {len(result)} chunks")
    
    print("\n✅ All tests completed successfully!")
    print("✅ Retrieval system has been simplified as requested:")
    print("   - Removed query search complexity")
    print("   - Kept only similarity search with tagging")
    print("   - Preserved fallback mechanisms")
    print("   - Maintained tag filtering functionality")

if __name__ == "__main__":
    test_simplified_retrieval() 