#!/usr/bin/env python3
"""
Simple test script for NuMind integration without extraction_schema parameter
"""

import os
import sys
import asyncio
from loguru import logger

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_interface import (
    create_numind_extraction_chain,
    extract_with_numind_from_bytes,
    extract_with_numind_using_schema,
    extract_specific_attribute_from_numind_result
)

async def test_numind_simple():
    """Test the NuMind integration functions without extraction_schema parameter."""
    
    print("🧪 Testing NuMind Integration (Simple)...")
    
    # Test 1: Create NuMind client
    print("\n1. Testing NuMind client creation...")
    try:
        client = create_numind_extraction_chain()
        if client:
            print("✅ NuMind client created successfully")
        else:
            print("❌ Failed to create NuMind client")
            return False
    except Exception as e:
        print(f"❌ Error creating NuMind client: {e}")
        return False
    
    # Test 2: Test with a sample PDF without extraction_schema parameter
    print("\n2. Testing NuMind extraction without extraction_schema parameter...")
    
    # Create a simple test PDF content
    test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF Content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
    
    try:
        # Test without extraction_schema parameter
        result = await extract_with_numind_using_schema(client, test_pdf_content, {})
        if result:
            print("✅ NuMind extraction completed successfully")
            print(f"   Result keys: {list(result.keys())}")
            
            # Test attribute extraction
            print("\n3. Testing attribute extraction from result...")
            test_attributes = ["Material Name", "Gender", "Contact Systems"]
            
            for attr in test_attributes:
                value = extract_specific_attribute_from_numind_result(result, attr)
                if value:
                    print(f"✅ Found '{attr}': {value}")
                else:
                    print(f"❌ Not found: {attr}")
        else:
            print("❌ NuMind extraction returned no result")
            return False
    except Exception as e:
        print(f"❌ Error during NuMind extraction: {e}")
        return False
    
    print("\n🎉 NuMind integration test completed successfully!")
    return True

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_numind_simple())
    
    if success:
        print("\n✅ NuMind integration is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ NuMind integration has issues!")
        sys.exit(1) 