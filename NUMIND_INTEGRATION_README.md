# NuMind Integration for LEOPARTS

This document describes the integration of NuMind structured extraction as a fallback method in the LEOPARTS extraction system.

## Overview

The NuMind integration replaces the original PDF-based RAG fallback (Stage 2) with a more deterministic structured extraction approach using the NuMind API. This provides better accuracy and consistency for attribute extraction.

## Architecture

### Current Extraction Flow

1. **Stage 1**: Web Data Extraction (unchanged)
   - Scrapes supplier websites for product data
   - Uses LLM chains to extract attributes from web content
   - Marks attributes as "NOT FOUND" if extraction fails

2. **Stage 2**: NuMind Fallback (NEW)
   - Uses NuMind API for structured extraction from PDF files
   - Extracts all attributes in a single API call
   - Provides deterministic results based on trained extraction models

### Fallback Logic

- If NuMind is available and configured: Use NuMind extraction
- If NuMind is not available: Fall back to original PDF RAG extraction
- If both fail: Mark attributes as "NOT FOUND"

## Configuration

### Environment Variables

The following environment variables are used for NuMind configuration:

```bash
NUMIND_API_KEY=your_numind_api_key_here
NUMIND_PROJECT_ID=your_numind_project_id_here
```

### Default Values

The system includes default values for testing:
- API Key: Pre-configured test key (replace with your own)
- Project ID: `dab6080e-5409-43b0-8f02-7a844ba933d5`

### Extraction Schema Configuration

**IMPORTANT**: To get the same results as your NuMind playground, you need to configure the extraction schema in `numind_schema_config.py`.

1. **Get your schema from NuMind playground:**
   - Go to your NuMind project
   - Look at the extraction schema configuration
   - Copy the exact schema structure

2. **Update the configuration file:**
   - Open `numind_schema_config.py`
   - Replace `CUSTOM_EXTRACTION_SCHEMA` with your actual schema
   - Update `CUSTOM_EXTRACTION_INSTRUCTIONS` if needed

3. **Example schema structure:**
   ```python
   CUSTOM_EXTRACTION_SCHEMA = {
       "type": "object",
       "properties": {
           "Material Name": {
               "type": "string",
               "description": "The main material name of the connector"
           },
           # ... add all your attributes here
       },
       "required": []
   }
   ```

## Implementation Details

### New Functions in `llm_interface.py`

1. **`create_numind_extraction_chain()`**
   - Initializes NuMind client
   - Returns client instance or None if configuration is missing

2. **`extract_with_numind_from_bytes(client, file_bytes, attribute_key)`**
   - Performs NuMind extraction using file bytes
   - Returns structured extraction results

3. **`extract_specific_attribute_from_numind_result(numind_result, attribute_key)`**
   - Extracts specific attribute values from NuMind results
   - Handles nested result structures

### Modified Files

1. **`requirements.txt`**
   - Added `numind` dependency

2. **`pages/extraction_attributs.py`**
   - Added NuMind chain initialization
   - Modified Stage 2 logic to use NuMind
   - Added file data storage for NuMind extraction

3. **`llm_interface.py`**
   - Added NuMind integration functions
   - Added configuration management

## Usage

### Running the System

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (optional, defaults are provided):
   ```bash
   export NUMIND_API_KEY="your_api_key"
   export NUMIND_PROJECT_ID="your_project_id"
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

### Testing the Integration

Run the test script to verify NuMind integration:

```bash
python test_numind_integration.py
```

## Benefits

1. **Deterministic Results**: NuMind provides consistent extraction results
2. **Better Accuracy**: Trained models outperform generic LLM prompts
3. **Faster Processing**: Single API call extracts all attributes
4. **Fallback Safety**: Original PDF extraction remains as backup

## Limitations

1. **API Dependency**: Requires active NuMind API access
2. **Training Required**: NuMind models need to be trained on your data
3. **Cost**: NuMind API calls may incur costs
4. **File Size**: Large PDFs may have processing limits

## Troubleshooting

### Common Issues

1. **NuMind Client Creation Fails**
   - Check API key and project ID configuration
   - Verify internet connectivity
   - Ensure `numind` package is installed

2. **Extraction Returns No Results**
   - Verify PDF file format and content
   - Check NuMind project configuration
   - Review API response for errors

3. **Attribute Not Found in Results**
   - Check NuMind project schema
   - Verify attribute names match exactly
   - Review extraction model training

### Debug Information

The system provides detailed logging for debugging:
- NuMind client initialization
- API call details
- Result processing
- Error handling

## Future Enhancements

1. **Batch Processing**: Support for multiple files
2. **Custom Schemas**: Dynamic schema configuration
3. **Result Caching**: Cache NuMind results for performance
4. **Model Training**: Integration with NuMind training workflows

## Support

For issues with NuMind integration:
1. Check the logs for detailed error messages
2. Verify API credentials and project configuration
3. Test with the provided test script
4. Review NuMind documentation for API-specific issues 