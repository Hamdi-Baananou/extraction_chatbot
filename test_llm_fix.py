#!/usr/bin/env python3
"""
Test script to verify LLM initialization fix.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_llm_initialization():
    """Test that the LLM initialization works correctly."""
    
    try:
        from llm_interface import initialize_llm
        print("✅ LLM interface imports successfully")
        
        # Test LLM initialization (will fail without API key, but should not crash)
        llm = initialize_llm()
        if llm is None:
            print("⚠️ LLM initialization returned None (expected without API key)")
        else:
            print("✅ LLM initialization successful")
            
        print("✅ LLM fix verification completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during LLM initialization test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_llm_initialization() 