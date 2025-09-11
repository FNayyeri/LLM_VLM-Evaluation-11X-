# CrossModal Intelligence Evaluation (CMIE) - LLMVLM-11X

Here we implement four approaches for multimodal LLM-VLM inference:
    1) OpenAI API approach - Using OpenAI's GPT-4V for multimodal inference for image, video and Document processing
    2) Gemini approach
    3) Claude approach
    4) LLaVA-based approach - Using a pre-trained multimodal model like LLaVA for image and video processing
     
## Model Options:
    - GPT-4V (OpenAI): 1-3s (API) | Score: 92.5/100 (highest quality)
    - GPT-4 Turbo (OpenAI): 1-3s (API) | Score: 92.5/100 (highest quality)
    - Gemini 1.5 Pro (Google): Text, Image, Video, Document | 1-3s | Score: 89.3/100
    - Gemini 2.0 Flash (Google): Text, Image, Video, Document | 0.5-1.5s | Score: 93.8/100
    - Claude 3.5 Sonnet (Anthropic): Text, Image, Document | 1-3s | Score: 94.2/100 (highest)
    - Claude 3 Haiku (Anthropic): Text, Image, Document | 1-2s | Score: 88.5/100
    - LLaVA 1.5 7B: Image, Video | 2-5s (GPU) / 30-60s (CPU) | Score: 78.5/100
    - LLaVA 1.5 13B: Image, Video | 4-8s (GPU) / 60-120s (CPU) | Score: 82.1/100
    - LLaVA 1.6 Next Image, Video | Mistral 7B: 3-6s (GPU) / 45-90s (CPU) | Score: 85.3/100
    - LLaVA 1.6 Next Image, Video | Vicuna 7B: 3-6s (GPU) / 45-90s (CPU) | Score: 83.7/100
    - LLaVA 1.6 Next Image, Video | Vicuna 13B: 5-10s (GPU) / 90-180s (CPU) | Score: 87.2/100

# OpenAI API approach:
The OpenAI API approach leverages OpenAI's powerful GPT-4V model for multimodal inference. This method is straightforward and provides high-quality results with minimal setup.

##  File Types Supported:
    - Image - JPG, PNG, etc.
    - Video - MP4, AVI, MOV
    - Document - DOCX files

## Key Features:
    - OpenAI GPT-4V integration - Uses OpenAI's multimodal API
    - API key input - Secure password field for OpenAI API key
    - Base64 image encoding - Handles image/video uploads for API
    - DOCX file support - Upload and process Word documents
    - Document summarisation - Extract and summarise text content
    - Fast inference - 1-3s response time via API
    - Text preview - Shows first 500 characters of document

# LLaVA approach:
the LLaVA  (Large Language and Vision Assistant) approach is optimised for multimodal understanding. This model is specifically designed for multimodal inference and provides good performance.

## Model Options Include:
    

## Key Features:  
    - Image processing: Direct image analysis using PIL
    - Video processing: Extracts key frames from videos using OpenCV
    - Text processing: Handles text-only queries
    - Unified interface: Single infer() method for all modalities

## Files Created:
    multimodal_inference.py - Main inference class: Creating the main multimodal LLM inference script that handles image, video, and text inputs
    requirements.txt - Dependencies: Creating requirements file with necessary dependencies for the multimodal LLM inference
    example_usage.py - Usage examples: Creating an example usage script to demonstrate how to use the multimodal LLM inference

## Quick Start:
`$ pip install -r requirements.txt`
`$ python app.py`

This code automatically handles GPU acceleration if available and processes videos by extracting representative frames for analysis.

# Performance Metrics:
1. Runtime Estimates: for typical inference on modern GPU and CPU hardware
2. Score: based on multimodal benchmarks (MME, SEED-Bench, LLaVA-Bench) 

# Evaluation Metrics:
1. Inference Time: Actual time taken for the model to process and generate response
2. Quality Score: Simple evaluation based on response length and coherence (60-95/100)

The metrics appear as Streamlit metric widgets below the description, showing:
- Real-time inference performance
- Basic quality assessment of the generated response


# Run the application: 
## from HPC:
    cd /path/to/prject
    git clone https://github.com/FNayyeri/LLM_VLM-Evaluation-11X-.git
    python -m venv venv
    pip install -r requirements.txt
    source /path/to/prject/venv/bin/activate
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

## From local machine, 
    ssh -L 8501:localhost:8501 user@<hpc login>
## on browser:
    localhost:8501
