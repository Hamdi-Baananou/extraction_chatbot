#!/usr/bin/env python3
"""
Test script for Hugging Face API embeddings integration.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embedding_api():
    """Test the Hugging Face embedding API."""
    
    # API configuration
    api_url = os.getenv("EMBEDDING_API_URL", "https://hbaananou-embedder-model.hf.space/embed")
    expected_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))
    
    # Test texts
    test_texts = [
        "Hello world",
        "Test sentence",
        "This is a longer test sentence to verify the embedding API works correctly."
    ]
    
    print(f"Testing embedding API at: {api_url}")
    print(f"Test texts: {test_texts}")
    print("-" * 50)
    
    try:
        # Make the API request
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json={"texts": test_texts},
            timeout=30
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        print(f"API Response: {json.dumps(result, indent=2)}")
        
        # Extract embeddings from the response
        if "embeddings" in result:
            embeddings = result["embeddings"]
        elif isinstance(result, list):
            embeddings = result
        else:
            embeddings = result.get("data", result.get("result", result))
            if not isinstance(embeddings, list):
                raise ValueError(f"Unexpected API response format: {result}")
        
        print(f"\nSuccessfully received {len(embeddings)} embeddings")
        
        # Check embedding dimensions
        for i, embedding in enumerate(embeddings):
            print(f"Text {i+1} embedding dimension: {len(embedding)}")
            if len(embedding) != expected_dimensions:
                print(f"⚠️  Warning: Expected {expected_dimensions} dimensions, got {len(embedding)}")
            print(f"Text {i+1} embedding preview: {embedding[:5]}...")
        
        print("\n✅ API test successful!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"❌ Failed to parse API response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_single_embedding():
    """Test single text embedding."""
    
    api_url = os.getenv("EMBEDDING_API_URL", "https://hbaananou-embedder-model.hf.space/embed")
    expected_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))
    test_text = "Single test sentence"
    
    print(f"\nTesting single embedding for: '{test_text}'")
    print("-" * 30)
    
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json={"texts": [test_text]},
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract single embedding
        if "embeddings" in result:
            embedding = result["embeddings"][0]
        elif isinstance(result, list):
            embedding = result[0]
        else:
            embedding = result.get("data", result.get("result", result))[0]
        
        print(f"Single embedding dimension: {len(embedding)}")
        if len(embedding) != expected_dimensions:
            print(f"⚠️  Warning: Expected {expected_dimensions} dimensions, got {len(embedding)}")
        print(f"Single embedding preview: {embedding[:5]}...")
        print("✅ Single embedding test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Single embedding test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Hugging Face Embedding API Integration")
    print("=" * 60)
    
    # Test multiple embeddings
    success1 = test_embedding_api()
    
    # Test single embedding
    success2 = test_single_embedding()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! API integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the API configuration.") 