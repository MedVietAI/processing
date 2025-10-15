# Local Mode Documentation

## Overview

The MedAI Processing system now supports two modes of operation:

- **Cloud Mode** (default): Uses NVIDIA and Gemini APIs for processing
- **Local Mode**: Uses MedAlpaca-13b model running locally for processing

## Local Mode Features

### Local Mode Benefits
- **No API costs**: Process data without external API calls
- **Privacy**: All processing happens locally
- **Offline capability**: Works without internet connection (after model download)
- **Medical specialization**: Uses MedAlpaca-13b, a model specifically fine-tuned for medical tasks

### Technical Details
- **Model**: [MedAlpaca-13b](https://huggingface.co/medalpaca/medalpaca-13b)
- **Quantization**: 4-bit quantization for memory efficiency
- **CUDA Support**: Automatic GPU acceleration when available
- **Memory Management**: Automatic model unloading to free memory

## Building and Running

### Build Script
Use the provided build script for easy building:

```bash
# Build for local mode
./build.sh local

# Build for cloud mode  
./build.sh cloud
```

### Manual Docker Build

#### Local Mode
```bash
docker build --build-arg IS_LOCAL=true -t medai-processing:local .
```

#### Cloud Mode
```bash
docker build --build-arg IS_LOCAL=false -t medai-processing:cloud .
```

## Environment Variables

### Local Mode Required
- `IS_LOCAL=true`: Enables local mode
- `HF_TOKEN`: Hugging Face token for model download (default: provided token)

### Local Mode Optional
- `HF_HOME`: Hugging Face cache directory (default: ~/.cache/huggingface)

### Cloud Mode Required
- `IS_LOCAL=false`: Enables cloud mode (default)
- `NVIDIA_API_1`: NVIDIA API key
- `GEMINI_API_1`: Gemini API key

## Output Differences

### Local Mode
- **Output Location**: `data/` folder (local filesystem)
- **No Google Drive**: Files are saved locally only
- **No OAuth**: Google Drive authentication is disabled

### Cloud Mode
- **Output Location**: `cache/outputs/` folder
- **Google Drive**: Files are uploaded to Google Drive
- **OAuth**: Google Drive authentication is available

## Model Information

### MedAlpaca-13b
- **Size**: 13 billion parameters
- **Specialization**: Medical domain tasks
- **Training Data**: 
  - ChatDoctor (200k Q&A pairs)
  - WikiDoc (67k items)
  - StackExchange (academia, biology, fitness, health)
  - Anki flashcards (33k items)

### Performance Considerations
- **Memory**: Requires ~8GB RAM (with 4-bit quantization)
- **GPU**: CUDA acceleration recommended for faster inference
- **Storage**: Model download requires ~7GB disk space

## Usage Examples

### Processing with Local Mode
1. Set `IS_LOCAL=true` in environment
2. Provide `HF_TOKEN` for model access
3. Run processing jobs - they will use MedAlpaca locally
4. Output files will be saved to `data/` folder

### Processing with Cloud Mode
1. Set `IS_LOCAL=false` (or omit)
2. Provide NVIDIA and Gemini API keys
3. Run processing jobs - they will use external APIs
4. Output files will be uploaded to Google Drive

## Troubleshooting

### Local Mode Issues
- **Model download fails**: Check HF_TOKEN and internet connection
- **Out of memory**: Ensure sufficient RAM (8GB+ recommended)
- **Slow inference**: Enable CUDA if available

### Cloud Mode Issues
- **API errors**: Check API keys and quotas
- **Upload failures**: Verify Google Drive authentication

## Migration Guide

### From Cloud to Local
1. Update environment: `IS_LOCAL=true`
2. Add HF_TOKEN
3. Rebuild container with local mode
4. Output will switch from Google Drive to local `data/` folder

### From Local to Cloud
1. Update environment: `IS_LOCAL=false`
2. Add NVIDIA and Gemini API keys
3. Rebuild container with cloud mode
4. Output will switch from local to Google Drive
