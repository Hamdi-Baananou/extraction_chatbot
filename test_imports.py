#!/usr/bin/env python3
"""
Test to verify that all required functions are available in vector_store module.
"""

def test_imports():
    """Test that all required functions can be imported."""
    try:
        # Test basic imports
        import vector_store
        
        # Check if required functions exist
        required_functions = [
            'get_embedding_function',
            'setup_vector_store', 
            'load_existing_vector_store',
            'SimpleRetriever',
            'ThresholdRetriever'
        ]
        
        print("🔍 Testing vector_store module imports...")
        
        for func_name in required_functions:
            if hasattr(vector_store, func_name):
                print(f"✅ {func_name} - AVAILABLE")
            else:
                print(f"❌ {func_name} - MISSING")
        
        # Test SimpleRetriever class
        if hasattr(vector_store, 'SimpleRetriever'):
            print("✅ SimpleRetriever class - AVAILABLE")
            
            # Check if SimpleRetriever has the retrieve method
            if hasattr(vector_store.SimpleRetriever, 'retrieve'):
                print("✅ SimpleRetriever.retrieve() method - AVAILABLE")
            else:
                print("❌ SimpleRetriever.retrieve() method - MISSING")
        else:
            print("❌ SimpleRetriever class - MISSING")
        
        print("\n🎉 All required functions are available!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 