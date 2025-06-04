"""
Utilities package for GPU optimization and performance monitoring.
"""

from .gpu import (
    get_device,
    get_optimal_dtype,
    get_device_map,
    optimize_memory,
    device_info,
    memory_stats,
    print_gpu_info,
    configure_model_for_gpu,
    get_generation_config
)

from .perf import (
    start_monitoring,
    stop_monitoring,
    get_current_status,
    get_summary_stats,
    track_model_operation,
    print_current_status
)

__all__ = [
    'get_device',
    'get_optimal_dtype', 
    'get_device_map',
    'optimize_memory',
    'device_info',
    'memory_stats',
    'print_gpu_info',
    'configure_model_for_gpu',
    'get_generation_config',
    'start_monitoring',
    'stop_monitoring',
    'get_current_status',
    'get_summary_stats',
    'track_model_operation',
    'print_current_status'
]