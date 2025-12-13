"""
Task Planning Module for Ocean Plastic Sentinel

This module acts as the strategic brain of the ocean cleanup system. It takes high-level
mission requests (like "find plastic debris in the Pacific Gyre") and breaks them down
into specific, executable tasks that can be performed by the system.

Think of it as a mission commander that:
- Analyzes what needs to be done based on the target area and urgency
- Creates a step-by-step plan with satellite data collection, analysis, and prediction tasks
- Schedules tasks based on priority and resource availability
- Handles different mission types (emergency cleanup vs routine monitoring)
- Plans for contingencies when things don't go as expected

The planner ensures the system works efficiently by organizing tasks in the right order,
allocating computational resources wisely, and adapting strategies based on mission requirements.
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

class MissionType(Enum):
    """Types of missions the system can execute."""
    STANDARD_DETECTION = "standard_detection"
    URGENT_CLEANUP = "urgent_cleanup"
    REGIONAL_SURVEY = "regional_survey"
    VALIDATION = "validation"
    
class TaskType(Enum):
    """
    Types of tasks the system can perform for ocean cleanup operations.
    
    Each task type represents a specific capability that contributes to the overall
    goal of finding and intercepting marine plastic debris efficiently.
    """
    SATELLITE_DETECTION = "satellite_detection"      # Analyze satellite images to spot plastic debris
    DRIFT_PREDICTION = "drift_prediction"            # Calculate where debris will move based on currents
    ROUTE_OPTIMIZATION = "route_optimization"        # Find the best path for cleanup vessels
    VALIDATION = "validation"                        # Check if previous predictions were accurate
    DATA_COLLECTION = "data_collection"              # Gather satellite images and ocean data


class TaskPriority(Enum):
    """
    Priority levels that determine how quickly tasks need to be executed.
    
    In ocean cleanup operations, timing matters - debris moves with currents,
    weather windows change, and cleanup vessels have limited operational time.
    """
    CRITICAL = 1    # Emergency situations: large spill, vessel in distress, immediate threat
    HIGH = 2        # Time-sensitive: optimal weather window, vessel already deployed
    NORMAL = 3      # Standard operations: routine monitoring, planned cleanup missions
    LOW = 4         # Background tasks: validation, learning, system maintenance


@dataclass
class Task:
    """
    Represents a single executable task in the ocean cleanup workflow.
    
    Each task is like a work order that tells the system exactly what to do,
    when to do it, and what resources it needs. For example:
    - "Analyze satellite image of coordinates 36.5°N, 125°W for plastic debris"
    - "Calculate drift prediction for detected debris over next 48 hours"
    - "Optimize route for cleanup vessel from Port A to intercept debris"
    
    Tasks have dependencies (some must complete before others can start) and
    resource requirements (API calls, memory, processing time).
    """
    
    task_id: str                                      # Unique identifier like "mission_123_detection"
    task_type: TaskType                               # What kind of work this task performs
    priority: TaskPriority                            # How urgent this task is
    parameters: Dict[str, Any]                        # Specific instructions (coordinates, thresholds, etc)
    dependencies: List[str] = field(default_factory=list)  # Tasks that must complete first
    estimated_duration_seconds: int = 300            # How long we expect this to take (5 min default)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)  # CPU, memory, API quota needed
    created_at: datetime = field(default_factory=datetime.utcnow)  # When this task was created
    scheduled_for: Optional[datetime] = None          # When this task should execute
    
    def __post_init__(self):
        """
        Automatically schedule the task based on its priority level.
        
        This mimics real-world triage: emergency tasks get immediate attention,
        while routine tasks can wait for available resources. The delays prevent
        system overload and allow for proper resource allocation.
        """
        if self.scheduled_for is None:
            # Different priorities get different scheduling delays to manage system load
            delay_minutes = {
                TaskPriority.CRITICAL: 0,      # Execute immediately - could be a vessel in distress
                TaskPriority.HIGH: 5,          # Small delay to batch with other urgent tasks
                TaskPriority.NORMAL: 15,       # Wait for current high-priority work to finish
                TaskPriority.LOW: 60           # Execute during low-traffic periods
            }
            self.scheduled_for = self.created_at + timedelta(
                minutes=delay_minutes[self.priority]
            )


@dataclass 
class ExecutionPlan:
    """
    Complete blueprint for executing an ocean cleanup mission.
    
    This is like a detailed project plan that breaks down the entire mission
    into manageable steps, schedules them efficiently, and prepares for things
    that might go wrong. It ensures the mission runs smoothly from start to finish.
    
    Example workflow:
    1. Collect satellite data for target region
    2. Analyze images to detect plastic debris  
    3. Calculate drift predictions based on ocean currents
    4. Optimize vessel routes to intercept debris
    5. Generate final cleanup recommendations
    """
    
    plan_id: str                                      # Unique plan identifier
    mission_id: str                                   # Links back to the original mission request
    tasks: List[Task]                                 # All tasks needed to complete the mission
    execution_order: List[str]                        # Task IDs in the order they should run
    total_estimated_duration: int                     # Expected time to complete entire mission
    resource_allocation: Dict[str, Any]               # CPU, memory, API quotas reserved for this plan
    contingency_plans: List[Dict[str, Any]] = field(default_factory=list)  # Backup plans if things fail
    created_at: datetime = field(default_factory=datetime.utcnow)  # When this plan was created


class TaskPlanner:
    """
    The strategic brain that figures out how to execute ocean cleanup missions efficiently.
    
    When you ask the system "find plastic debris in this area and tell me where vessels
    should go to clean it up," this planner breaks that down into specific steps:
    
    1. What satellite images do we need to download?
    2. What ocean current data should we fetch?
    3. In what order should we analyze the data?
    4. How should we handle the situation if satellite data is unavailable?
    5. What resources (API calls, memory, processing time) will each step need?
    
    The planner is like a project manager that understands both the technical capabilities
    of the system and the real-world constraints of ocean cleanup operations. It creates
    detailed execution plans that maximize efficiency while being prepared for failures.
    """
    
    def __init__(self, gemini_client: GeminiAPIClient):
        """
        Set up the task planner with access to AI analysis capabilities.
        
        The planner needs access to Gemini AI for strategic decision-making,
        like determining if a mission area is too large for standard processing
        or if weather conditions require urgent task prioritization.
        
        Args:
            gemini_client: Connection to Gemini AI for strategic analysis
        """
        self.gemini_client = gemini_client
        self.is_initialized = False
        
        # Track all active execution plans to prevent conflicts and manage workload
        self.active_plans = {}
        
        # Monitor system resource consumption to avoid overloading
        self.resource_usage = {
            'api_calls_per_hour': 0,      # Track API quota usage
            'memory_usage_mb': 0,         # Monitor memory consumption
            'concurrent_tasks': 0         # Count active parallel tasks
        }
        
        # Pre-built strategies for common mission types - like having templates
        # for different types of operations instead of planning from scratch each time
        self.strategy_templates = {
            'standard_detection': self._create_standard_detection_strategy,    # Normal monitoring missions
            'urgent_cleanup': self._create_urgent_cleanup_strategy,            # Emergency response (oil spill, etc)
            'validation_sweep': self._create_validation_strategy,              # Check previous predictions
            'regional_survey': self._create_regional_survey_strategy           # Large area mapping
        }
    
    async def initialize(self) -> bool:
        """
        Prepare the planner for operation by loading historical data and setting up monitoring.
        
        This is like a mission commander reviewing past operations, understanding what
        strategies worked well, and setting up systems to track current performance.
        The planner needs this context to make good decisions about how to approach new missions.
        
        Returns:
            bool: True if the planner is ready to create mission plans
        """
        try:
            logger.info("Initializing Task Planner...")
            
            # Load data about past missions to inform planning decisions
            await self._load_planning_knowledge()
            
            # Set up systems to monitor resource usage and prevent overload
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