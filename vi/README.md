# Vietnamese Translation Module

This module provides Vietnamese translation functionality for the MedAI Processing application using the Helsinki-NLP/opus-mt-en-vi model.

## Features

- **English to Vietnamese Translation**: Translates English text to Vietnamese using the Helsinki-NLP/opus-mt-en-vi model
- **Batch Processing**: Efficiently translates multiple texts at once
- **Dictionary Translation**: Translates specific fields in data dictionaries
- **Integration**: Seamlessly integrates with both SFT and RAG processing workflows
- **Error Handling**: Graceful fallback to original text if translation fails
- **Logging**: Comprehensive logging for debugging and monitoring

## Configuration

Add the following environment variable to your `.env` file:

```bash
EN_VI=Helsinki-NLP/opus-mt-en-vi
```

## Usage

### Basic Translation

```python
from vi.translator import VietnameseTranslator

# Initialize translator
translator = VietnameseTranslator()

# Load the model
translator.load_model()

# Translate single text
translated = translator.translate_text("Hello, how are you?")

# Translate batch of texts
texts = ["Text 1", "Text 2", "Text 3"]
translated_batch = translator.translate_batch(texts)
```

### Dictionary Translation

```python
# Translate specific fields in a dictionary
data = {
    "instruction": "Answer the question",
    "input": "What is diabetes?",
    "output": "Diabetes is a metabolic disorder..."
}

translated_data = translator.translate_dict(data, ["instruction", "input", "output"])
```

## Integration

The translation functionality is automatically integrated into the processing workflows:

1. **UI Toggle**: Users can enable Vietnamese translation via the checkbox in the web interface
2. **SFT Processing**: All text fields in SFT format are translated when enabled
3. **RAG Processing**: All text fields in RAG format are translated when enabled
4. **Metadata**: Translated rows are marked with `vietnamese_translated: true` in metadata

## Model Information

- **Model**: Helsinki-NLP/opus-mt-en-vi
- **Source Language**: English
- **Target Language**: Vietnamese
- **BLEU Score**: 37.2
- **chrF Score**: 0.542
- **License**: Apache 2.0

## Testing

Run the test script to verify translation functionality:

```bash
python test_translation.py
```

## Files

- `translator.py`: Main translation class
- `download.py`: Model download script for Docker
- `processing_utils.py`: Utility functions for processing integration
- `__init__.py`: Module initialization
- `README.md`: This documentation

## Notes

- The model is automatically downloaded during Docker build
- Translation is performed on the CPU by default, but can use GPU if available
- The model requires the target language token `>>vie<<` for proper translation
- All translation operations include comprehensive error handling and logging
