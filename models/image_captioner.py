"""
Fixed image captioning module that produces proper descriptions instead of garbled text.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import re

class ImageCaptioner:
    def __init__(self, model_config):
        self.model_id = model_config["model_id"]
        self.max_new_tokens = model_config["max_new_tokens"]
        self.temperature = model_config["temperature"]
        self.repetition_penalty = model_config["repetition_penalty"]
        self.do_sample = model_config["do_sample"]
        
        # Load processor and model
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def caption_image(self, image_path_or_pil, prompt="Describe this image in detail:"):
        """
        Generate a proper image caption without repetitive garbled text.
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object
            prompt: Optional prompt to guide captioning
            
        Returns:
            str: Clean, coherent image description
        """
        try:
            # Load image if path provided
            if isinstance(image_path_or_pil, str):
                image = Image.open(image_path_or_pil).convert('RGB')
            else:
                image = image_path_or_pil.convert('RGB')
            
            # Process inputs
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)
            
            # Generate caption with proper parameters to avoid repetition
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Prevent 3-gram repetitions
                )
            
            # Decode and clean the output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Remove the input prompt from output if present
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            # Clean up repetitive patterns and artifacts
            cleaned_text = self._clean_caption(generated_text)
            
            return cleaned_text
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def _clean_caption(self, text):
        """
        Clean up the generated caption to remove repetitive patterns and artifacts.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove repetitive phrases (same phrase repeated 3+ times)
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            # Check for immediate repetitions
            repeat_count = 1
            j = i + 1
            while j < len(words) and words[j] == current_word:
                repeat_count += 1
                j += 1
            
            # Only add the word once if it's repeated excessively
            if repeat_count >= 3:
                cleaned_words.append(current_word)
                i = j
            else:
                cleaned_words.append(current_word)
                i += 1
        
        cleaned_text = ' '.join(cleaned_words)
        
        # Remove common artifacts
        artifacts = [
            "Question:", "Answer:", "Caption:", "Description:",
            "Image shows", "The image shows", "This image shows"
        ]
        for artifact in artifacts:
            cleaned_text = cleaned_text.replace(artifact, "").strip()
        
        # Ensure proper sentence structure
        if cleaned_text and not cleaned_text[0].isupper():
            cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
        
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'
        
        return cleaned_text