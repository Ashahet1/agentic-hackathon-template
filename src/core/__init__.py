"""
Ocean Plastic Sentinel - Core Module

This module provides the main components for the multi-agent ocean plastic
detection and cleanup optimization system.
"""

from .agent import OceanPlasticSentinel, MissionRequest, MissionResult
from .planner import TaskPlanner, Task, ExecutionPlan, TaskType, TaskPriority
from .executor import TaskExecutor
from .memory import MemorySystem

__all__ = [
    'OceanPlasticSentinel',
    'MissionRequest', 
    'MissionResult',
    'MissionStatus',
    'MissionType',
    'TaskPlanner',
    'Task',
    'ExecutionPlan', 
    'TaskType',
    'TaskPriority',
    'TaskExecutor',
    'MemorySystem'
]

__version__ = "1.0.0"