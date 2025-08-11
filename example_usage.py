from multimodal_inference import MultimodalLLM

def main():
    # Initialize the multimodal LLM
    print("Loading multimodal LLM...")
    llm = MultimodalLLM()
    
    # Example 1: Image analysis
    print("\n=== Image Analysis ===")
    try:
        result = llm.infer(
            text_prompt="Describe what you see in detail",
            image_path="sample_image.jpg"  # Replace with actual image path
        )
        print(f"Response: {result}")
    except Exception as e:
        print(f"Image analysis failed: {e}")
    
    # Example 2: Video analysis
    print("\n=== Video Analysis ===")
    try:
        result = llm.infer(
            text_prompt="What activities are happening in this video?",
            video_path="sample_video.mp4"  # Replace with actual video path
        )
        print(f"Response: {result}")
    except Exception as e:
        print(f"Video analysis failed: {e}")
    
    # Example 3: Text-only conversation
    print("\n=== Text Conversation ===")
    result = llm.infer(text_prompt="What are the key differences between supervised and unsupervised learning?")
    print(f"Response: {result}")

if __name__ == "__main__":
    main()