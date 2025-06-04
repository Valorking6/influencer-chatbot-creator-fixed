"""
GPU-optimized image captioning module with CUDA acceleration.
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import re
import logging
from utils.gpu import get_device, get_optimal_dtype, configure_model_for_gpu, optimize_memory
from utils.perf import track_model_operation

logger = logging.getLogger(__name__)

class ImageCaptioner:
    def __init__(self, model_config):
        self.model_id = model_config["model_id"]
        self.max_new_tokens = model_config["max_new_tokens"]
        self.temperature = model_config["temperature"]
        self.repetition_penalty = model_config["repetition_penalty"]
        self.do_sample = model_config["do_sample"]
        
        # Get optimal device and dtype
        self.device = get_device()
        self.dtype = get_optimal_dtype()
        
        logger.info(f"Loading image captioning model on {self.device} with dtype {self.dtype}")
        
        # Load processor and model with GPU optimization
        with track_model_operation("load_image_captioner"):
            self.processor = Blip2Processor.from_pretrained(self.model_id)
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Configure model for optimal GPU performance
            if self.device.type == "cuda":
                self.model = configure_model_for_gpu(self.model)
                # Optimize memory after loading
                optimize_memory()
        
        logger.info("Image captioning model loaded successfully")
        
    def caption_image(self, image_path_or_pil, prompt="Describe this image in detail:"):
        """
        Generate a proper image caption without repetitive garbled text.
        Optimized for GPU acceleration.
        """
        try:
            with track_model_operation("image_captioning"):
                # Load image if path provided
                if isinstance(image_path_or_pil, str):
                    image = Image.open(image_path_or_pil).convert('RGB')
                else:
                    image = image_path_or_pil.convert('RGB')
                
                # Process inputs with proper device placement
                inputs = self.processor(image, prompt, return_tensors="pt")
                
                # Move inputs to device
                if self.device.type == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate caption with GPU optimization
                with torch.no_grad():
                    # Use autocast for mixed precision on GPU
                    if self.device.type == "cuda" and self.dtype == torch.float16:
                        with torch.cuda.amp.autocast():
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=self.max_new_tokens,
                                temperature=self.temperature,
                                do_sample=self.do_sample,
                                repetition_penalty=self.repetition_penalty,
                                pad_token_id=self.processor.tokenizer.eos_token_id,
                                eos_token_id=self.processor.tokenizer.eos_token_id,
                                early_stopping=True,
                                no_repeat_ngram_size=3,
                                use_cache=True
                            )
                    else:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            do_sample=self.do_sample,
                            repetition_penalty=self.repetition_penalty,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                            use_cache=True
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
                
                # Clear GPU cache after generation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                return cleaned_text
                
        except Exception as e:
            logger.error(f"Error in image captioning: {e}")
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
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
