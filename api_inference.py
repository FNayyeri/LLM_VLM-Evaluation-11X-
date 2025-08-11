import base64
import cv2
from PIL import Image
from openai import OpenAI
import anthropic
import google.generativeai as genai
from typing import List
import os
from docx import Document

class APIMultimodalLLM:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        
        if "gpt" in model_name.lower():
            self.client = OpenAI(api_key=api_key)
            self.provider = "openai"
        elif "claude" in model_name.lower():
            self.client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
        elif "gemini" in model_name.lower():
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
            self.provider = "google"
    
    def encode_image(self, image_path: str) -> str:
        img = Image.open(image_path)
        if img.format.lower() not in ['png', 'jpeg', 'jpg', 'gif', 'webp']:
            temp_path = image_path.rsplit('.', 1)[0] + '_converted.jpg'
            img.convert('RGB').save(temp_path, 'JPEG')
            image_path = temp_path
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
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
    
    def extract_docx_text(self, docx_path: str) -> str:
        doc = Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        return '\n'.join(text)
    
    def infer(self, text_prompt: str, image_path: str = None, video_path: str = None, docx_path: str = None) -> str:
        if self.provider == "openai":
            return self._openai_infer(text_prompt, image_path, video_path, docx_path)
        elif self.provider == "anthropic":
            return self._anthropic_infer(text_prompt, image_path, video_path, docx_path)
        elif self.provider == "google":
            return self._google_infer(text_prompt, image_path, video_path, docx_path)
    
    def _openai_infer(self, text_prompt: str, image_path: str = None, video_path: str = None, docx_path: str = None) -> str:
        if docx_path:
            text_content = self.extract_docx_text(docx_path)
            if len(text_content) > 8000:
                text_content = text_content[:8000] + "..."
            prompt = f"{text_prompt}\n\nDocument content:\n{text_content}"
            messages = [{"role": "user", "content": prompt}]
        else:
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
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def _anthropic_infer(self, text_prompt: str, image_path: str = None, video_path: str = None, docx_path: str = None) -> str:
        content = [{"type": "text", "text": text_prompt}]
        
        if docx_path:
            text_content = self.extract_docx_text(docx_path)[:8000]
            content[0]["text"] = f"{text_prompt}\n\nDocument content:\n{text_content}"
        
        if image_path:
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
        
        if video_path:
            frames = self.extract_video_frames(video_path, 2)  # Claude has stricter limits
            for frame in frames:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame
                    }
                })
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text
    
    def _google_infer(self, text_prompt: str, image_path: str = None, video_path: str = None, docx_path: str = None) -> str:
        content = [text_prompt]
        
        if docx_path:
            text_content = self.extract_docx_text(docx_path)[:8000]
            content[0] = f"{text_prompt}\n\nDocument content:\n{text_content}"
        
        if image_path:
            img = Image.open(image_path)
            content.append(img)
        
        if video_path:
            frames = self.extract_video_frames(video_path, 3)
            for frame_b64 in frames:
                frame_data = base64.b64decode(frame_b64)
                temp_path = f"temp_gemini_{len(content)}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(frame_data)
                img = Image.open(temp_path)
                content.append(img)
                os.remove(temp_path)
        
        response = self.client.generate_content(content)
        return response.text