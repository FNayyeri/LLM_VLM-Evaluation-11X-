import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import cv2
import numpy as np
from typing import Union, List
import os

class LlavaLLM:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Set cache to project directory
        cache_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Model cache directory: {cache_dir}")
        
        # Use different processors for different model versions
        if "v1.6" in model_name or "next" in model_name.lower():
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            self.processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
            # use_fast=False parameter to use slow tokenizer and avoid sentencepiece issues
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            self.processor = LlavaProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
    
    def extract_video_frames(self, video_path: str, max_frames: int = 8) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames
    
    def infer(self, text_prompt: str, image_path: str = None, video_path: str = None) -> str:
        if not image_path and not video_path:
            return "Text-only inference not supported with this model"
        
        image = Image.open(image_path) if image_path else self.extract_video_frames(video_path)[0]
        
        prompt = f"USER: <image>\n{text_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.split("ASSISTANT:")[-1].strip()

# Usage example
if __name__ == "__main__":
    llm = LlavaLLM()
    
    # Text + Image
    result = llm.infer(
        text_prompt="What do you see in this image?",
        image_path="data/images/01.jpg"
    )
    print("Image analysis:", result)
    
    # Text + Video
    result = llm.infer(
        text_prompt="Describe what happens in this video",
        video_path="data/videos/01.mp4"
    )
    print("Video analysis:", result)
    
    # Text only
    result = llm.infer(text_prompt="Explain quantum computing in simple terms")
    print("Text response:", result)