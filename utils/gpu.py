"""
GPU detection, configuration, and optimization utilities.
Provides centralized CUDA management for all AI models.
"""

import torch
import gc
import os
import logging
from typing import Dict, Optional, Tuple
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUManager:
    """Centralized GPU management and optimization."""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.device_info = self._get_device_info()
        self._configure_cuda_settings()
        
    def _get_optimal_device(self) -> torch.device:
        """Get the optimal device for computation."""
        if torch.cuda.is_available():
            # Use the GPU with most free memory
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                max_memory = 0
                best_gpu = 0
                for i in range(gpu_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i
                device = torch.device(f"cuda:{best_gpu}")
            else:
                device = torch.device("cuda:0")
            
            logger.info(f"Using GPU: {device}")
            return device
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    
    def _get_device_info(self) -> Dict:
        """Get detailed device information."""
        info = {
            "device_type": "cuda" if self.device.type == "cuda" else "cpu",
            "device_name": "CPU" if self.device.type == "cpu" else torch.cuda.get_device_name(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            info.update({
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
                "gpu_count": torch.cuda.device_count()
            })
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            info.update({
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_free_gb": round(info["total_memory_gb"] - memory_reserved, 2)
            })
        
        return info
    
    def _configure_cuda_settings(self):
        """Configure optimal CUDA settings for performance."""
        if torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Enable TensorFloat-32 for better performance on Ampere GPUs
            if torch.cuda.get_device_capability(self.device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TensorFloat-32 for Ampere GPU")
            
            logger.info("CUDA optimizations configured")
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type for the current device."""
        if self.device.type == "cuda":
            # Use float16 for GPU to save memory and increase speed
            return torch.float16
        else:
            # Use float32 for CPU for better compatibility
            return torch.float32
    
    def get_device_map(self, model_size: str = "medium") -> str:
        """Get optimal device mapping strategy."""
        if not torch.cuda.is_available():
            return "cpu"
        
        gpu_count = torch.cuda.device_count()
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                          for i in range(gpu_count)) / 1024**3
        
        # Determine strategy based on available memory and model size
        if total_memory > 24 and gpu_count > 1:
            return "auto"  # Let accelerate handle multi-GPU
        elif total_memory > 12:
            return "auto"  # Single GPU with auto placement
        else:
            return "sequential"  # Sequential loading for limited memory
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            logger.info("GPU memory optimized")
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory_info": "N/A"}
        
        stats = {
            "device": str(self.device),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(self.device) / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(self.device) / 1024**3, 2),
            "memory_cached_gb": round(torch.cuda.memory_cached(self.device) / 1024**3, 2),
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(self.device) / 1024**3, 2),
            "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved(self.device) / 1024**3, 2)
        }
        
        return stats
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.info("Peak memory statistics reset")
    
    def print_device_info(self):
        """Print comprehensive device information."""
        print("\n" + "="*50)
        print("GPU DEVICE INFORMATION")
        print("="*50)
        
        for key, value in self.device_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        if torch.cuda.is_available():
            print("\nMemory Statistics:")
            memory_stats = self.get_memory_stats()
            for key, value in memory_stats.items():
                if key != "device":
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("="*50)

# Global GPU manager instance
gpu_manager = GPUManager()

# Convenience functions
def get_device() -> torch.device:
    """Get the optimal device."""
    return gpu_manager.device

def get_optimal_dtype() -> torch.dtype:
    """Get optimal data type."""
    return gpu_manager.get_optimal_dtype()

def get_device_map(model_size: str = "medium") -> str:
    """Get optimal device mapping."""
    return gpu_manager.get_device_map(model_size)

def optimize_memory():
    """Optimize GPU memory."""
    gpu_manager.optimize_memory()

def device_info() -> Dict:
    """Get device information."""
    return gpu_manager.device_info

def memory_stats() -> Dict:
    """Get memory statistics."""
    return gpu_manager.get_memory_stats()

def print_gpu_info():
    """Print GPU information."""
    gpu_manager.print_device_info()

def configure_model_for_gpu(model, enable_gradient_checkpointing: bool = True):
    """Configure a model for optimal GPU performance."""
    if torch.cuda.is_available():
        # Move model to GPU
        model = model.to(gpu_manager.device)
        
        # Enable gradient checkpointing to save memory
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Enable mixed precision if supported
        if hasattr(model, 'half') and gpu_manager.get_optimal_dtype() == torch.float16:
            model = model.half()
        
        logger.info(f"Model configured for GPU: {gpu_manager.device}")
    
    return model

def get_generation_config() -> Dict:
    """Get optimal generation configuration for current device."""
    config = {
        "torch_dtype": gpu_manager.get_optimal_dtype(),
        "device_map": gpu_manager.get_device_map(),
        "low_cpu_mem_usage": True,
    }
    
    if torch.cuda.is_available():
        config.update({
            "use_cache": True,
            "pad_token_id": None,  # Will be set by tokenizer
        })
    
    return config