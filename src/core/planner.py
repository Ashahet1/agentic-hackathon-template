"""
Task Planning Module for Ocean Plastic Sentinel.

This module handles strategic planning, task decomposition, and workflow
orchestration for marine debris detection and cleanup operations.
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.config import config
from src.integrations.gemini_client import GeminiAPIClient


logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of available task types."""
    SATELLITE_DETECTION = "satellite_detection"
    DRIFT_PREDICTION = "drift_prediction"
    ROUTE_OPTIMIZATION = "route_optimization"
    VALIDATION = "validation"
    DATA_COLLECTION = "data_collection"


class TaskPriority(Enum):
    """Task priority levels for execution scheduling."""
    CRITICAL = 1    # Immediate execution required
    HIGH = 2        # Execute within 1 hour
    NORMAL = 3      # Execute within 4 hours
    LOW = 4         # Execute when resources available


@dataclass
class Task:
    """Individual task representation."""
    
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_seconds: int = 300  # Default 5 minutes
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_for: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.scheduled_for is None:
            # Schedule based on priority
            delay_minutes = {
                TaskPriority.CRITICAL: 0,
                TaskPriority.HIGH: 5,
                TaskPriority.NORMAL: 15,
                TaskPriority.LOW: 60
            }
            self.scheduled_for = self.created_at + timedelta(
                minutes=delay_minutes[self.priority]
            )


@dataclass 
class ExecutionPlan:
    """Complete execution plan for a mission."""
    
    plan_id: str
    mission_id: str
    tasks: List[Task]
    execution_order: List[str]  # Task IDs in execution order
    total_estimated_duration: int
    resource_allocation: Dict[str, Any]
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class TaskPlanner:
    """
    Strategic task planner for Ocean Plastic Sentinel operations.
    
    This class analyzes mission requirements, decomposes them into executable
    tasks, and creates optimized execution plans considering resource constraints
    and dependencies.
    """
    
    def __init__(self, gemini_client: GeminiAPIClient):
        """
        Initialize the task planner.
        
        Args:
            gemini_client: Initialized Gemini API client for strategic analysis
        """
        self.gemini_client = gemini_client
        self.is_initialized = False
        
        # Planning state
        self.active_plans = {}
        self.resource_usage = {
            'api_calls_per_hour': 0,
            'memory_usage_mb': 0,
            'concurrent_tasks': 0
        }
        
        # Strategy templates for common mission types
        self.strategy_templates = {
            'standard_detection': self._create_standard_detection_strategy,
            'urgent_cleanup': self._create_urgent_cleanup_strategy,
            'validation_sweep': self._create_validation_strategy,
            'regional_survey': self._create_regional_survey_strategy
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the task planner.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing Task Planner...")
            
            # Initialize planning knowledge base
            await self._load_planning_knowledge()
            
            # Set up resource monitoring
            await self._initialize_resource_tracking()
            
            self.is_initialized = True
            logger.info("Task Planner initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task Planner initialization failed: {str(e)}")
            return False
    
    async def create_execution_plan(self, 
                                   mission_request: Any,
                                   context: Dict[str, Any]) -> ExecutionPlan:
        """
        Create a comprehensive execution plan for a mission.
        
        Args:
            mission_request: Mission parameters and requirements
            context: Historical data and current conditions
            
        Returns:
            ExecutionPlan with optimized task sequence and resource allocation
        """
        if not self.is_initialized:
            raise RuntimeError("TaskPlanner not initialized")
        
        logger.info(f"Creating execution plan for mission {mission_request.mission_id}")
        
        try:
            # Analyze mission requirements and select strategy
            strategy_type = await self._analyze_mission_requirements(mission_request, context)
            
            # Generate task list using selected strategy
            tasks = await self._generate_tasks(strategy_type, mission_request, context)
            
            # Optimize task dependencies and execution order
            execution_order = await self._optimize_task_order(tasks)
            
            # Allocate resources and create contingency plans
            resource_allocation = await self._allocate_resources(tasks)
            contingency_plans = await self._create_contingency_plans(tasks, context)
            
            # Calculate total estimated duration
            total_duration = sum(task.estimated_duration_seconds for task in tasks)
            
            # Create execution plan
            plan = ExecutionPlan(
                plan_id=f"plan_{mission_request.mission_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                mission_id=mission_request.mission_id,
                tasks=tasks,
                execution_order=execution_order,
                total_estimated_duration=total_duration,
                resource_allocation=resource_allocation,
                contingency_plans=contingency_plans
            )
            
            # Store plan for tracking
            self.active_plans[plan.plan_id] = plan
            
            logger.info(f"Execution plan created: {len(tasks)} tasks, {total_duration}s estimated duration")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {str(e)}")
            raise
    
    async def _analyze_mission_requirements(self, 
                                          mission_request: Any, 
                                          context: Dict[str, Any]) -> str:
        """Analyze mission requirements to select optimal strategy."""
        
        # Determine mission characteristics
        region_size = self._calculate_region_size(mission_request.target_region)
        historical_detection_rate = context.get('historical_data', {}).get('detection_rate', 0.0)
        urgency_level = mission_request.priority_level
        
        # Select strategy based on characteristics
        if urgency_level == 'urgent':
            return 'urgent_cleanup'
        elif region_size > config.system.max_search_area_km2:
            return 'regional_survey'
        elif historical_detection_rate < 0.3:
            return 'validation_sweep'
        else:
            return 'standard_detection'
    
    async def _generate_tasks(self, 
                             strategy_type: str,
                             mission_request: Any,
                             context: Dict[str, Any]) -> List[Task]:
        """Generate task list using selected strategy template."""
        
        strategy_func = self.strategy_templates.get(strategy_type)
        if not strategy_func:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return await strategy_func(mission_request, context)
    
    async def _create_standard_detection_strategy(self, 
                                                 mission_request: Any, 
                                                 context: Dict[str, Any]) -> List[Task]:
        """Create task list for standard detection mission."""
        
        tasks = []
        base_id = mission_request.mission_id
        
        # Task 1: Data Collection
        data_task = Task(
            task_id=f"{base_id}_data_collection",
            task_type=TaskType.DATA_COLLECTION,
            priority=TaskPriority.HIGH,
            parameters={
                'region': mission_request.target_region,
                'data_sources': ['sentinel2', 'noaa_currents', 'noaa_winds'],
                'time_range_hours': 48
            },
            estimated_duration_seconds=180
        )
        tasks.append(data_task)
        
        # Task 2: Satellite Detection
        detection_task = Task(
            task_id=f"{base_id}_satellite_detection",
            task_type=TaskType.SATELLITE_DETECTION,
            priority=TaskPriority.HIGH,
            parameters={
                'region': mission_request.target_region,
                'analysis_type': 'debris_detection',
                'confidence_threshold': config.system.min_confidence_threshold
            },
            dependencies=[data_task.task_id],
            estimated_duration_seconds=300
        )
        tasks.append(detection_task)
        
        # Task 3: Drift Prediction
        prediction_task = Task(
            task_id=f"{base_id}_drift_prediction",
            task_type=TaskType.DRIFT_PREDICTION,
            priority=TaskPriority.NORMAL,
            parameters={
                'time_horizon_hours': mission_request.time_window_hours,
                'physics_model': 'adaptive_coefficients',
                'update_learning': True
            },
            dependencies=[detection_task.task_id],
            estimated_duration_seconds=240
        )
        tasks.append(prediction_task)
        
        # Task 4: Route Optimization
        route_task = Task(
            task_id=f"{base_id}_route_optimization",
            task_type=TaskType.ROUTE_OPTIMIZATION,
            priority=TaskPriority.NORMAL,
            parameters={
                'vessel_constraints': mission_request.vessel_constraints,
                'optimization_target': 'fuel_efficiency',
                'weather_consideration': True
            },
            dependencies=[prediction_task.task_id],
            estimated_duration_seconds=180
        )
        tasks.append(route_task)
        
        return tasks
    
    async def _create_urgent_cleanup_strategy(self, 
                                            mission_request: Any, 
                                            context: Dict[str, Any]) -> List[Task]:
        """Create task list for urgent cleanup mission with parallel execution."""
        
        tasks = []
        base_id = mission_request.mission_id
        
        # Parallel data collection with reduced scope for speed
        data_task = Task(
            task_id=f"{base_id}_urgent_data",
            task_type=TaskType.DATA_COLLECTION,
            priority=TaskPriority.CRITICAL,
            parameters={
                'region': mission_request.target_region,
                'data_sources': ['sentinel2', 'noaa_currents'],  # Reduced scope
                'time_range_hours': 24,  # Shorter range
                'quality': 'rapid'
            },
            estimated_duration_seconds=120
        )
        tasks.append(data_task)
        
        # Rapid detection with lower confidence threshold
        detection_task = Task(
            task_id=f"{base_id}_urgent_detection",
            task_type=TaskType.SATELLITE_DETECTION,
            priority=TaskPriority.CRITICAL,
            parameters={
                'region': mission_request.target_region,
                'analysis_type': 'rapid_detection',
                'confidence_threshold': max(0.4, config.system.min_confidence_threshold - 0.2)
            },
            dependencies=[data_task.task_id],
            estimated_duration_seconds=180
        )
        tasks.append(detection_task)
        
        # Immediate route calculation (parallel with prediction)
        quick_route_task = Task(
            task_id=f"{base_id}_immediate_route",
            task_type=TaskType.ROUTE_OPTIMIZATION,
            priority=TaskPriority.CRITICAL,
            parameters={
                'vessel_constraints': mission_request.vessel_constraints,
                'optimization_target': 'time_efficiency',
                'use_current_positions': True
            },
            dependencies=[detection_task.task_id],
            estimated_duration_seconds=90
        )
        tasks.append(quick_route_task)
        
        return tasks
    
    async def _create_validation_strategy(self, 
                                        mission_request: Any, 
                                        context: Dict[str, Any]) -> List[Task]:
        """Create task list for validation sweep mission."""
        
        tasks = []
        base_id = mission_request.mission_id
        
        # Retrieve historical predictions for validation
        validation_task = Task(
            task_id=f"{base_id}_validation",
            task_type=TaskType.VALIDATION,
            priority=TaskPriority.NORMAL,
            parameters={
                'region': mission_request.target_region,
                'validation_period_days': 7,
                'accuracy_metrics': ['spatial_error', 'temporal_error', 'detection_rate']
            },
            estimated_duration_seconds=360
        )
        tasks.append(validation_task)
        
        return tasks
    
    async def _create_regional_survey_strategy(self, 
                                             mission_request: Any, 
                                             context: Dict[str, Any]) -> List[Task]:
        """Create task list for large regional survey mission."""
        
        tasks = []
        base_id = mission_request.mission_id
        
        # Break large region into grid cells
        grid_cells = self._create_region_grid(mission_request.target_region)
        
        for i, cell in enumerate(grid_cells):
            # Individual detection task for each cell
            cell_task = Task(
                task_id=f"{base_id}_cell_{i}",
                task_type=TaskType.SATELLITE_DETECTION,
                priority=TaskPriority.LOW,
                parameters={
                    'region': cell,
                    'analysis_type': 'survey_detection',
                    'confidence_threshold': config.system.min_confidence_threshold
                },
                estimated_duration_seconds=200
            )
            tasks.append(cell_task)
        
        return tasks
    
    async def _optimize_task_order(self, tasks: List[Task]) -> List[str]:
        """Optimize task execution order considering dependencies and priorities."""
        
        # Build dependency graph
        dependency_graph = {}
        for task in tasks:
            dependency_graph[task.task_id] = task.dependencies
        
        # Topological sort with priority consideration
        execution_order = []
        available_tasks = [t for t in tasks if not t.dependencies]
        completed_tasks = set()
        
        while available_tasks:
            # Sort by priority and scheduled time
            available_tasks.sort(key=lambda t: (t.priority.value, t.scheduled_for))
            
            # Execute highest priority task
            current_task = available_tasks.pop(0)
            execution_order.append(current_task.task_id)
            completed_tasks.add(current_task.task_id)
            
            # Add newly available tasks
            for task in tasks:
                if (task.task_id not in execution_order and 
                    task not in available_tasks and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    available_tasks.append(task)
        
        return execution_order
    
    async def _allocate_resources(self, tasks: List[Task]) -> Dict[str, Any]:
        """Allocate system resources to tasks."""
        
        total_api_calls = sum(1 for t in tasks if t.task_type in [TaskType.SATELLITE_DETECTION, TaskType.DATA_COLLECTION])
        concurrent_limit = config.system.max_concurrent_requests
        
        return {
            'api_calls_allocated': total_api_calls,
            'concurrent_tasks_limit': min(len(tasks), concurrent_limit),
            'memory_allocation_mb': len(tasks) * 50,  # 50MB per task estimate
            'timeout_seconds': config.system.request_timeout_seconds
        }
    
    async def _create_contingency_plans(self, 
                                       tasks: List[Task], 
                                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create contingency plans for potential failure scenarios."""
        
        contingencies = []
        
        # API rate limit contingency
        contingencies.append({
            'trigger': 'api_rate_limit_exceeded',
            'action': 'reduce_concurrent_requests',
            'parameters': {'new_limit': 2}
        })
        
        # Low confidence detection contingency
        contingencies.append({
            'trigger': 'low_confidence_detections',
            'action': 'expand_search_area',
            'parameters': {'expansion_factor': 1.5}
        })
        
        # Satellite data unavailable contingency
        contingencies.append({
            'trigger': 'satellite_data_unavailable',
            'action': 'use_historical_patterns',
            'parameters': {'fallback_days': 7}
        })
        
        return contingencies
    
    def _calculate_region_size(self, region: Dict[str, float]) -> float:
        """Calculate approximate region size in km²."""
        # Simplified calculation - would use proper geodetic formulas in production
        lat_diff = abs(region['north'] - region['south'])
        lon_diff = abs(region['east'] - region['west'])
        return lat_diff * lon_diff * 111 * 111  # Rough km² conversion
    
    def _create_region_grid(self, region: Dict[str, float], cell_size_degrees: float = 1.0) -> List[Dict[str, float]]:
        """Break large region into manageable grid cells."""
        cells = []
        
        lat_start = region['south']
        while lat_start < region['north']:
            lon_start = region['west']
            while lon_start < region['east']:
                cell = {
                    'north': min(lat_start + cell_size_degrees, region['north']),
                    'south': lat_start,
                    'east': min(lon_start + cell_size_degrees, region['east']),
                    'west': lon_start
                }
                cells.append(cell)
                lon_start += cell_size_degrees
            lat_start += cell_size_degrees
        
        return cells
    
    async def _load_planning_knowledge(self) -> None:
        """Load planning knowledge base and historical performance data."""
        # Implementation would load from database/files
        logger.debug("Loading planning knowledge base...")
    
    async def _initialize_resource_tracking(self) -> None:
        """Initialize resource usage tracking."""
        # Reset resource counters
        self.resource_usage = {
            'api_calls_per_hour': 0,
            'memory_usage_mb': 0,
            'concurrent_tasks': 0
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current planner status and metrics."""
        return {
            'initialized': self.is_initialized,
            'active_plans': len(self.active_plans),
            'resource_usage': self.resource_usage.copy()
        }
    
    async def shutdown(self) -> None:
        """Shutdown the task planner and cleanup resources."""
        logger.info("Shutting down Task Planner...")
        self.active_plans.clear()
        self.is_initialized = False