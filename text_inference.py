from docx import Document
from openai import OpenAI
import os

class TextLLM:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def extract_docx_text(self, docx_path: str) -> str:
        doc = Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        return '\n'.join(text)
    
    def infer(self, text_prompt: str, docx_path: str = None, text_content: str = None) -> str:
        if docx_path:
            text_content = self.extract_docx_text(docx_path)
        
        if not text_content:
            return "No text content provided"
        
        # Truncate if too long (OpenAI token limit)
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "..."
        
        prompt = f"{text_prompt}\n\nDocument content:\n{text_content}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        return response.choices[0].message.content