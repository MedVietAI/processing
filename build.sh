#!/bin/bash

# Build script for MedAI Processing with dynamic local/cloud mode support

echo "🏗️  MedAI Processing Build Script"
echo "=================================="

# Check if mode is specified
if [ "$1" = "local" ]; then
    echo "🏠 Building in LOCAL mode (MedAlpaca-13b)"
    docker build --build-arg IS_LOCAL=true -t medai-processing:local .
elif [ "$1" = "cloud" ]; then
    echo "☁️  Building in CLOUD mode (NVIDIA/Gemini APIs)"
    docker build --build-arg IS_LOCAL=false -t medai-processing:cloud .
else
    echo "Usage: $0 [local|cloud]"
    echo ""
    echo "  local  - Build with MedAlpaca-13b model for local inference"
    echo "  cloud  - Build with NVIDIA/Gemini API integration"
    echo ""
    echo "Examples:"
    echo "  $0 local   # Build for local mode"
    echo "  $0 cloud   # Build for cloud mode"
    exit 1
fi

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "To run the container:"
if [ "$1" = "local" ]; then
    echo "  docker run -p 7860:7860 -e HF_TOKEN=your_token_here medai-processing:local"
else
    echo "  docker run -p 7860:7860 -e NVIDIA_API_1=your_key -e GEMINI_API_1=your_key medai-processing:cloud"
fi
