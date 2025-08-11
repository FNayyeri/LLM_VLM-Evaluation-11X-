import base64
import cv2
from PIL import Image
from openai import OpenAI
from typing import List
import os

class OpenAIMultimodalLLM:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def encode_image(self, image_path: str) -> str:
        # Convert image to supported format if needed
        img = Image.open(image_path)
        supported_formats = ['png', 'jpeg', 'jpg', 'gif', 'webp']
        if img.format.lower() not in supported_formats:
            # Convert to JPEG
            temp_path = image_path.rsplit('.', 1)[0] + '_converted.jpg'
            img.convert('RGB').save(temp_path, 'JPEG')
            image_path = temp_path
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up temporary file if created
        if 'converted' in image_path:
            os.remove(image_path)
        
        return encoded
    
    def extract_video_frames(self, video_path: str, max_frames: int = 4) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                temp_path = f"temp_frame_{idx}.jpg"
                cv2.imwrite(temp_path, frame)
                frames.append(self.encode_image(temp_path))
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        cap.release()
        return frames
    
    def infer(self, text_prompt: str, image_path: str = None, video_path: str = None) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
        
        if image_path:
            base64_image = self.encode_image(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        if video_path:
            frames = self.extract_video_frames(video_path)
            for frame in frames:
                messages[0]["content"].append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })
        
        if not image_path and not video_path:
            messages = [{"role": "user", "content": text_prompt}]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200
        )
        
        return response.choices[0].message.content