#!/bin/bash

# Build script for MedAI Processing with dynamic local/cloud mode support

echo "🏗️  MedAI Processing Build Script"
echo "=================================="

# Function to prompt user for yes/no
prompt_yes_no() {
    local prompt="$1"
    local response
    while true; do
        read -p "$prompt (y/n): " response
        case $response in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}


# Check if mode is specified as argument
if [ "$1" = "local" ]; then
    MODE="local"
elif [ "$1" = "cloud" ]; then
    MODE="cloud"
else
    # Interactive mode selection
    echo "Please select the runtime mode:"
    echo "1) Local mode (MedAlpaca-13b) - No API costs, complete privacy, heavy dependencies"
    echo "2) Cloud mode (NVIDIA/Gemini APIs) - Faster processing, requires API keys, lightweight"
    echo ""
    
    while true; do
        read -p "Enter your choice (1 or 2): " choice
        case $choice in
            1) MODE="local"; break;;
            2) MODE="cloud"; break;;
            *) echo "Please enter 1 for local mode or 2 for cloud mode.";;
        esac
    done
fi

echo ""
echo "Selected mode: $MODE"
echo ""

# Ask if user wants to build/rebuild Docker image
if prompt_yes_no "Would you like to build/rebuild the Docker image?"; then
    echo ""
    if [ "$MODE" = "local" ]; then
        echo "🏠 Building in LOCAL mode (MedAlpaca-13b)..."
        echo "   - Installing local LLM dependencies (torch-cuda, accelerate, bitsandbytes, etc.)"
        echo "   - Excluding cloud API dependencies (google-genai, google-auth, etc.)"
        docker build --build-arg IS_LOCAL=true -t medai-processing:local .
    else
        echo "☁️  Building in CLOUD mode (NVIDIA/Gemini APIs)..."
        echo "   - Installing cloud API dependencies (google-genai, google-auth, etc.)"
        echo "   - Excluding heavy local LLM dependencies (torch-cuda, accelerate, bitsandbytes, etc.)"
        docker build --build-arg IS_LOCAL=false -t medai-processing:cloud .
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Build completed successfully!"
    else
        echo ""
        echo "❌ Build failed! Please check the error messages above."
        exit 1
    fi
else
    echo "⏭️  Skipping Docker build..."
fi

echo ""

# Ask if user wants to run the container
if prompt_yes_no "Would you like to run the Docker container now?"; then
    echo ""
    echo "🚀 Starting Docker container..."
    
    # Set the image name based on mode
    if [ "$MODE" = "local" ]; then
        IMAGE_NAME="medai-processing:local"
        IS_LOCAL_FLAG="-e IS_LOCAL=true"
    else
        IMAGE_NAME="medai-processing:cloud"
        IS_LOCAL_FLAG="-e IS_LOCAL=false"
    fi
    
    # Build docker run command
    DOCKER_CMD="docker run -p 7860:7860 $IS_LOCAL_FLAG"
    
    # Add environment variables
    if [ "$MODE" = "local" ]; then
        echo "🔑 Please provide your Hugging Face token:"
        read -p "HF_TOKEN: " hf_token
        if [ -n "$hf_token" ]; then
            # Validate HF token format (should start with hf_)
            if [[ "$hf_token" =~ ^hf_ ]]; then
                DOCKER_CMD="$DOCKER_CMD -e HF_TOKEN=\"$hf_token\""
            else
                echo "⚠️  Warning: HF token should start with 'hf_'. Adding anyway..."
                DOCKER_CMD="$DOCKER_CMD -e HF_TOKEN=\"$hf_token\""
            fi
        else
            echo "⚠️  Warning: No HF_TOKEN provided. Model download may fail."
        fi
    else
        echo "🔑 Please provide your API keys:"
        read -p "NVIDIA_API_1: " nvidia_key
        read -p "GEMINI_API_1: " gemini_key
        
        if [ -n "$nvidia_key" ]; then
            DOCKER_CMD="$DOCKER_CMD -e NVIDIA_API_1=\"$nvidia_key\""
        fi
        if [ -n "$gemini_key" ]; then
            DOCKER_CMD="$DOCKER_CMD -e GEMINI_API_1=\"$gemini_key\""
        fi
        
        if [ -z "$nvidia_key" ] && [ -z "$gemini_key" ]; then
            echo "⚠️  Warning: No API keys provided. Processing may fail."
        fi
    fi
    
    # Add image name
    DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "❌ Error: Docker image '$IMAGE_NAME' not found!"
        echo "   Please build the image first by running this script and choosing 'yes' for building."
        exit 1
    fi
    
    # Run the container
    echo "Running: $DOCKER_CMD"
    echo ""
    eval "$DOCKER_CMD"
else
    echo ""
    echo "📋 Manual run command:"
    if [ "$MODE" = "local" ]; then
        echo "  docker run -p 7860:7860 -e IS_LOCAL=true -e HF_TOKEN=your_token_here medai-processing:local"
    else
        echo "  docker run -p 7860:7860 -e IS_LOCAL=false -e NVIDIA_API_1=your_key -e GEMINI_API_1=your_key medai-processing:cloud"
    fi
fi
