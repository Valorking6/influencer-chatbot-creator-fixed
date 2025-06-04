"""
Performance monitoring and GPU status utilities.
Provides real-time monitoring of GPU usage and performance metrics.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring will be limited.")

class PerformanceMonitor:
    """Monitor system and GPU performance in real-time."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.performance_data = []
        self.start_time = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.performance_data = []
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> List[Dict]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Performance monitoring stopped")
        return self.performance_data.copy()
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                data = self._collect_metrics()
                self.performance_data.append(data)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
    
    def _collect_metrics(self) -> Dict:
        """Collect current performance metrics."""
        timestamp = time.time() - (self.start_time or time.time())
        
        # System metrics
        metrics = {
            "timestamp": timestamp,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        }
        
        # GPU metrics
        if NVML_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                gpu_metrics = []
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None
                    
                    # Power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = None
                    
                    gpu_data = {
                        "gpu_id": i,
                        "gpu_utilization": util.gpu,
                        "memory_utilization": util.memory,
                        "memory_used_gb": mem_info.used / 1024**3,
                        "memory_total_gb": mem_info.total / 1024**3,
                        "memory_free_gb": mem_info.free / 1024**3,
                        "temperature_c": temp,
                        "power_usage_w": power
                    }
                    gpu_metrics.append(gpu_data)
                
                metrics["gpus"] = gpu_metrics
                
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
                metrics["gpus"] = []
        else:
            metrics["gpus"] = []
        
        return metrics
    
    def get_current_status(self) -> Dict:
        """Get current system and GPU status."""
        return self._collect_metrics()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from monitoring data."""
        if not self.performance_data:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [d["cpu_percent"] for d in self.performance_data]
        memory_values = [d["memory_percent"] for d in self.performance_data]
        
        summary = {
            "monitoring_duration_s": self.performance_data[-1]["timestamp"] if self.performance_data else 0,
            "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "cpu_max": max(cpu_values) if cpu_values else 0,
            "memory_avg": sum(memory_values) / len(memory_values) if memory_values else 0,
            "memory_max": max(memory_values) if memory_values else 0,
        }
        
        # GPU summary
        if self.performance_data and self.performance_data[0].get("gpus"):
            gpu_count = len(self.performance_data[0]["gpus"])
            gpu_summaries = []
            
            for gpu_id in range(gpu_count):
                gpu_util_values = []
                gpu_mem_values = []
                gpu_temp_values = []
                gpu_power_values = []
                
                for data in self.performance_data:
                    if data.get("gpus") and len(data["gpus"]) > gpu_id:
                        gpu_data = data["gpus"][gpu_id]
                        gpu_util_values.append(gpu_data["gpu_utilization"])
                        gpu_mem_values.append(gpu_data["memory_utilization"])
                        if gpu_data["temperature_c"] is not None:
                            gpu_temp_values.append(gpu_data["temperature_c"])
                        if gpu_data["power_usage_w"] is not None:
                            gpu_power_values.append(gpu_data["power_usage_w"])
                
                gpu_summary = {
                    "gpu_id": gpu_id,
                    "utilization_avg": sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0,
                    "utilization_max": max(gpu_util_values) if gpu_util_values else 0,
                    "memory_utilization_avg": sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else 0,
                    "memory_utilization_max": max(gpu_mem_values) if gpu_mem_values else 0,
                    "temperature_avg": sum(gpu_temp_values) / len(gpu_temp_values) if gpu_temp_values else None,
                    "temperature_max": max(gpu_temp_values) if gpu_temp_values else None,
                    "power_avg": sum(gpu_power_values) / len(gpu_power_values) if gpu_power_values else None,
                    "power_max": max(gpu_power_values) if gpu_power_values else None,
                }
                gpu_summaries.append(gpu_summary)
            
            summary["gpus"] = gpu_summaries
        
        return summary

class ModelPerformanceTracker:
    """Track performance metrics for individual model operations."""
    
    def __init__(self):
        self.operation_times = {}
        self.memory_usage = {}
    
    def start_operation(self, operation_name: str):
        """Start tracking an operation."""
        self.operation_times[operation_name] = {
            "start_time": time.time(),
            "start_memory": self._get_gpu_memory() if NVML_AVAILABLE else None
        }
    
    def end_operation(self, operation_name: str) -> Dict:
        """End tracking an operation and return metrics."""
        if operation_name not in self.operation_times:
            return {}
        
        start_data = self.operation_times[operation_name]
        end_time = time.time()
        end_memory = self._get_gpu_memory() if NVML_AVAILABLE else None
        
        metrics = {
            "operation": operation_name,
            "duration_s": end_time - start_data["start_time"],
            "start_memory_gb": start_data["start_memory"],
            "end_memory_gb": end_memory,
            "memory_delta_gb": (end_memory - start_data["start_memory"]) if (end_memory and start_data["start_memory"]) else None
        }
        
        # Clean up
        del self.operation_times[operation_name]
        
        return metrics
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get current GPU memory usage."""
        if not NVML_AVAILABLE:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / 1024**3
        except:
            return None

# Global instances
performance_monitor = PerformanceMonitor()
model_tracker = ModelPerformanceTracker()

# Convenience functions
def start_monitoring(interval: float = 1.0):
    """Start performance monitoring."""
    performance_monitor.start_monitoring(interval)

def stop_monitoring() -> List[Dict]:
    """Stop monitoring and get data."""
    return performance_monitor.stop_monitoring()

def get_current_status() -> Dict:
    """Get current system status."""
    return performance_monitor.get_current_status()

def get_summary_stats() -> Dict:
    """Get monitoring summary."""
    return performance_monitor.get_summary_stats()

def track_model_operation(operation_name: str):
    """Context manager for tracking model operations."""
    class OperationTracker:
        def __enter__(self):
            model_tracker.start_operation(operation_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            metrics = model_tracker.end_operation(operation_name)
            if metrics:
                logger.info(f"Operation '{operation_name}' completed in {metrics['duration_s']:.2f}s")
                if metrics.get('memory_delta_gb'):
                    logger.info(f"Memory usage changed by {metrics['memory_delta_gb']:.2f}GB")
    
    return OperationTracker()

def print_current_status():
    """Print current system and GPU status."""
    status = get_current_status()
    
    print("\n" + "="*50)
    print("CURRENT SYSTEM STATUS")
    print("="*50)
    print(f"CPU Usage: {status['cpu_percent']:.1f}%")
    print(f"Memory Usage: {status['memory_percent']:.1f}% ({status['memory_used_gb']:.1f}GB used)")
    
    if status.get("gpus"):
        print("\nGPU Status:")
        for gpu in status["gpus"]:
            print(f"  GPU {gpu['gpu_id']}:")
            print(f"    Utilization: {gpu['gpu_utilization']}%")
            print(f"    Memory: {gpu['memory_utilization']}% ({gpu['memory_used_gb']:.1f}GB/{gpu['memory_total_gb']:.1f}GB)")
            if gpu['temperature_c']:
                print(f"    Temperature: {gpu['temperature_c']}°C")
            if gpu['power_usage_w']:
                print(f"    Power: {gpu['power_usage_w']:.1f}W")
    else:
        print("\nGPU Status: No GPU monitoring available")
    
    print("="*50)
