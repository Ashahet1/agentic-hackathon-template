"""
Ocean Plastic Sentinel - Main Agent Coordinator

This module orchestrates the multi-agent system for marine debris detection,
drift prediction, and cleanup optimization. It manages the coordination between
planner, executor, and memory components while providing a clean interface for
mission execution and system monitoring.

The system follows a three-phase workflow:
1. Planning: Task decomposition and strategy selection based on mission requirements
2. Execution: Data processing and analysis using satellite imagery and ocean data
3. Learning: Context preservation and model updates based on prediction accuracy

Key Design Patterns:
- Command Pattern: MissionRequest encapsulates all operation parameters
- Strategy Pattern: Different mission types use different execution strategies  
- Observer Pattern: System metrics and status monitoring throughout execution
- Factory Pattern: Mission result construction based on execution outcomes

The coordinator maintains system state, tracks active missions, and provides
performance metrics for operational monitoring and system optimization.
"""
from enum import Enum
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from src.config import config
from src.core.planner import TaskPlanner
from src.core.executor import TaskExecutor  
from src.core.memory import MemorySystem
from src.integrations.gemini_client import GeminiAPIClient


logger = logging.getLogger(__name__)

class MissionStatus(Enum):
    """Status of a mission execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
@dataclass
class MissionRequest:
    """
    Container for cleanup mission parameters and constraints.
    
    This class encapsulates all inputs required for a marine debris cleanup
    mission, following the Command Pattern to ensure complete parameter
    specification and immutable request handling.
    
    Attributes:
        mission_id: Unique identifier for tracking and correlation
        target_region: Geographic boundaries in decimal degrees
        priority_level: Execution urgency affecting resource allocation
        vessel_constraints: Physical limitations of cleanup vessels
        time_window_hours: Maximum prediction horizon for drift calculations
        requested_by: Originator identification for audit trails
        timestamp: Request creation time for scheduling and timeout handling
    """
    
    mission_id: str
    target_region: Dict[str, float]  # {'north': lat, 'south': lat, 'east': lon, 'west': lon}
    priority_level: str = 'standard'  # 'urgent', 'standard', 'background'
    vessel_constraints: Optional[Dict[str, Any]] = None  # Speed, capacity, draft limitations
    time_window_hours: int = 72  # Prediction horizon for drift modeling
    requested_by: str = 'system'  # Source system or user identification
    timestamp: datetime = None  # Auto-populated request creation timestamp
    
    def __post_init__(self):
        """Post-initialization validation and timestamp assignment."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        # Validate geographic region bounds
        if not self._validate_region():
            raise ValueError("Invalid target region coordinates")
    
    def _validate_region(self) -> bool:
        """Validate geographic coordinates are within valid ranges."""
        region = self.target_region
        return (
            -90 <= region.get('south', 0) <= 90 and
            -90 <= region.get('north', 0) <= 90 and
            -180 <= region.get('west', 0) <= 180 and
            -180 <= region.get('east', 0) <= 180 and
            region.get('south') < region.get('north') and
            region.get('west') < region.get('east')
        )


@dataclass
class MissionResult:
    """
    Container for mission execution results and recommendations.
    
    This class provides structured output from the multi-agent system,
    containing all detection results, predictions, route optimizations,
    and performance metrics required for cleanup operations.
    
    Attributes:
        mission_id: Correlation ID linking to original MissionRequest
        detections: List of identified debris locations with confidence scores
        predictions: Drift trajectory forecasts with uncertainty bounds
        recommended_routes: Optimized vessel paths for fuel-efficient cleanup
        confidence_metrics: Statistical measures of prediction reliability
        execution_time_seconds: Total processing duration for performance monitoring
        status: Mission outcome classification for operational dashboards
        timestamp: Completion time for result validity assessment
    """
    
    mission_id: str
    detections: List[Dict[str, Any]]  # Debris locations with GPS coordinates and confidence
    predictions: List[Dict[str, Any]]  # Future positions with drift trajectories
    recommended_routes: List[Dict[str, Any]]  # Optimized vessel navigation paths
    confidence_metrics: Dict[str, float]  # Accuracy measures and uncertainty bounds
    execution_time_seconds: float  # Performance metric for system optimization
    status: str  # 'completed', 'partial', 'failed' - operational outcome
    timestamp: datetime = None  # Result generation timestamp
    
    def __post_init__(self):
        """Post-initialization timestamp assignment."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class OceanPlasticSentinel:
    """
    Main agent coordinator for the Ocean Plastic Sentinel system.

    This class orchestrates the multi-agent workflow ensuring proper coordination
    between planning, execution, and memory components while maintaining system
    resilience and performance monitoring.
    """
    
    def __init__(self):
        """
        Initialize the Ocean Plastic Sentinel coordinator.
        
        Sets up component references and initializes system state tracking
        for mission management and performance monitoring.
        """
        # Core component initialization - dependency injection pattern
        self.gemini_client = GeminiAPIClient()
        self.planner = None  # Initialized during system startup
        self.executor = None  # Initialized during system startup
        self.memory = None   # Initialized during system startup
        self.is_initialized = False
        
        # Mission state tracking for concurrent operation management
        self.active_missions = {}  # mission_id -> MissionRequest mapping
        
        # System-wide performance metrics for operational dashboards
        self.system_metrics = {
            'total_missions': 0,              # Cumulative mission counter
            'successful_predictions': 0,       # Validated accuracy tracker
            'average_accuracy': 0.0,          # Rolling accuracy average
            'fuel_savings_calculated': 0.0    # Economic impact measurement
        }
    
    async def initialize(self) -> bool:
        """
        Initialize all system components with comprehensive validation.
        
        Performs sequential initialization of all system components with proper
        error handling and rollback capabilities. This method ensures the system
        reaches a fully operational state before accepting mission requests.
        
        Initialization Sequence:
        1. Configuration validation and environment checks
        2. Gemini API connection establishment and testing
        3. Core component instantiation and cross-linking
        4. Component-specific initialization and health checks
        5. System metrics initialization and monitoring setup
        
        Returns:
            bool: True if all components initialized successfully, False otherwise
        """
        try:
            logger.info("Initializing Ocean Plastic Sentinel system...")
            
            # Step 1: Validate configuration completeness and correctness
            if not config.validate():
                logger.error("Invalid system configuration - missing required parameters")
                return False
            
            # Step 2: Establish Gemini API connection with health check
            logger.info("Connecting to Gemini 1.5 Pro API...")
            if not await self.gemini_client.initialize():
                logger.error("Failed to initialize Gemini API client - check API key and network")
                return False
            
            # Step 3: Instantiate core components with dependency injection
            logger.info("Initializing core system components...")
            self.planner = TaskPlanner(self.gemini_client)
            self.executor = TaskExecutor(self.gemini_client)
            self.memory = MemorySystem()
            
            # Step 4: Initialize each component with error isolation
            initialization_tasks = [
                ("TaskPlanner", self.planner.initialize()),
                ("TaskExecutor", self.executor.initialize()),
                ("MemorySystem", self.memory.initialize())
            ]
            
            for component_name, init_task in initialization_tasks:
                logger.info(f"Initializing {component_name}...")
                if not await init_task:
                    logger.error(f"{component_name} initialization failed")
                    return False
            
            # Step 5: System is ready for operation
            self.is_initialized = True
            logger.info("Ocean Plastic Sentinel system initialized successfully")
            logger.info("System ready to accept mission requests")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed with exception: {str(e)}")
            # Cleanup partial initialization state
            await self._cleanup_partial_initialization()
            return False
    
    async def execute_mission(self, request: MissionRequest) -> MissionResult:
        """
        Execute a complete cleanup mission workflow with full orchestration.
        
        This method implements the core three-phase workflow orchestrating all
        system components to transform a mission request into actionable cleanup
        recommendations with validated predictions and optimized routing.
        
        Args:
            request: Comprehensive mission parameters and operational constraints
            
        Returns:
            MissionResult: Complete mission outcome with detections, predictions,
                          routes, confidence metrics, and execution performance
            
        Raises:
            RuntimeError: If system not initialized or critical component failure
            ValueError: If mission request contains invalid parameters
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        start_time = datetime.utcnow()
        mission_id = request.mission_id
        
        try:
            logger.info(f"Starting mission execution: {mission_id}")
            logger.info(f"Target region: {request.target_region}")
            logger.info(f"Priority level: {request.priority_level}")
            
            # Register mission for concurrent tracking and monitoring
            self.active_missions[mission_id] = request
            
            # PHASE 1: STRATEGIC PLANNING
            logger.info(f"Phase 1: Strategic planning for mission {mission_id}")
            planning_context = await self._retrieve_mission_context(request)
            execution_plan = await self.planner.create_execution_plan(request, planning_context)
            
            logger.info(f"Generated execution plan with {len(execution_plan.tasks)} tasks")
            
            # PHASE 2: COORDINATED EXECUTION
            logger.info(f"Phase 2: Executing mission plan for {mission_id}")
            execution_results = await self.executor.execute_plan(execution_plan)
            
            logger.info(f"Execution completed: {len(execution_results.get('detections', []))} detections found")
            
            # PHASE 3: LEARNING INTEGRATION
            logger.info(f"Phase 3: Learning integration for mission {mission_id}")
            await self.memory.store_mission_results(mission_id, execution_results)
            learning_updates = await self.memory.update_predictive_models(execution_results)
            
            # Compile comprehensive mission results
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result = MissionResult(
                mission_id=mission_id,
                detections=execution_results.get('detections', []),
                predictions=execution_results.get('predictions', []),
                recommended_routes=execution_results.get('routes', []),
                confidence_metrics=execution_results.get('confidence_metrics', {}),
                execution_time_seconds=execution_time,
                status='completed'
            )
            
            # Update system-wide performance metrics
            self._update_system_metrics(result)
            
            # Clean up mission tracking resources
            self.active_missions.pop(mission_id, None)
            
            logger.info(f"Mission {mission_id} completed successfully in {execution_time:.2f}s")
            logger.info(f"Generated {len(result.recommended_routes)} optimized routes")
            
            return result
            
        except Exception as e:
            logger.error(f"Mission {mission_id} failed with exception: {str(e)}")
            
            # Generate failure result with diagnostic information
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            return MissionResult(
                mission_id=mission_id,
                detections=[],
                predictions=[],
                recommended_routes=[],
                confidence_metrics={'error': str(e), 'execution_time': execution_time},
                execution_time_seconds=execution_time,
                status='failed'
            )
    
    async def validate_previous_predictions(self, mission_ids: List[str]) -> Dict[str, Any]:
        """
        Validate accuracy of previous mission predictions.
        
        Args:
            mission_ids: List of mission IDs to validate
            
        Returns:
            Dictionary containing validation results and accuracy metrics
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        validation_results = {}
        
        for mission_id in mission_ids:
            try:
                # Retrieve previous mission data
                mission_data = await self.memory.retrieve_mission_data(mission_id)
                
                if not mission_data:
                    validation_results[mission_id] = {'error': 'Mission data not found'}
                    continue
                
                # Execute validation workflow
                validation_result = await self.executor.validate_predictions(mission_data)
                validation_results[mission_id] = validation_result
                
                # Update learning based on validation
                await self.memory.update_from_validation(mission_id, validation_result)
                
            except Exception as e:
                logger.error(f"Validation failed for mission {mission_id}: {str(e)}")
                validation_results[mission_id] = {'error': str(e)}
        
        return validation_results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and performance metrics.
        
        Returns:
            Dictionary containing system status and metrics
        """
        status = {
            'initialized': self.is_initialized,
            'active_missions': len(self.active_missions),
            'system_metrics': self.system_metrics.copy(),
            'component_status': {}
        }
        
        if self.is_initialized:
            status['component_status'] = {
                'planner': await self.planner.get_status(),
                'executor': await self.executor.get_status(), 
                'memory': await self.memory.get_status(),
                'gemini_client': 'operational'
            }
        
        return status
    
    async def _retrieve_mission_context(self, request: MissionRequest) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for strategic mission planning.
        
        Gathers historical performance data, current environmental conditions,
        and learned model parameters to inform planning decisions and optimize
        execution strategies based on regional patterns and system experience.
        
        Context Components:
        - Historical detection patterns and success rates for the region
        - Current oceanographic conditions affecting debris movement
        - Learned drift model coefficients and accuracy metrics
        - Recent system performance and resource utilization patterns
        
        Args:
            request: Mission request containing target region and parameters
            
        Returns:
            Dict containing all contextual information for planning optimization
        """
        logger.debug(f"Retrieving context for region: {request.target_region}")
        
        context = {
            # Historical performance data for regional optimization
            'historical_data': await self.memory.get_regional_history(request.target_region),
            
            # Real-time environmental conditions affecting operations
            'current_conditions': await self.memory.get_current_conditions(request.target_region),
            
            # Learned model parameters and accuracy metrics
            'learning_parameters': await self.memory.get_learning_parameters(),
            
            # System resource utilization for capacity planning
            'system_load': {
                'active_missions': len(self.active_missions),
                'average_execution_time': self._calculate_average_execution_time(),
                'resource_availability': await self._assess_resource_availability()
            }
        }
        
        logger.debug("Mission context retrieval completed")
        return context
    
    def _update_system_metrics(self, result: MissionResult) -> None:
        """
        Update system-wide performance metrics for operational monitoring.
        
        Maintains running statistics on system performance, prediction accuracy,
        and operational efficiency to support system optimization and capacity
        planning decisions.
        
        Metrics Updated:
        - Total mission counter for workload tracking
        - Rolling average accuracy for quality assessment  
        - Fuel savings calculations for economic impact measurement
        - Execution time trends for performance optimization
        
        Args:
            result: Completed mission result with performance data
        """
        self.system_metrics['total_missions'] += 1
        
        if result.status == 'completed':
            # Calculate accuracy metrics from confidence scores
            if result.confidence_metrics:
                avg_confidence = sum(result.confidence_metrics.values()) / len(result.confidence_metrics)
                
                # Update rolling average accuracy using exponential moving average
                total = self.system_metrics['total_missions']
                current_avg = self.system_metrics['average_accuracy']
                self.system_metrics['average_accuracy'] = (
                    (current_avg * (total - 1) + avg_confidence) / total
                )
                
                # Estimate fuel savings based on detection efficiency
                if 'fuel_efficiency_gain' in result.confidence_metrics:
                    self.system_metrics['fuel_savings_calculated'] += result.confidence_metrics['fuel_efficiency_gain']
        
        logger.debug(f"System metrics updated: {self.system_metrics}")
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate rolling average execution time for capacity planning."""
        # Implementation would maintain execution time history
        return 300.0  # Placeholder: 5 minutes average
    
    async def _assess_resource_availability(self) -> Dict[str, Any]:
        """Assess current system resource availability for load balancing."""
        return {
            'api_quota_remaining': 0.8,  # 80% quota remaining
            'memory_utilization': 0.3,   # 30% memory used
            'concurrent_capacity': config.system.max_concurrent_requests - len(self.active_missions)
        }
    
    async def _cleanup_partial_initialization(self) -> None:
        """Clean up resources from partial initialization failure."""
        logger.info("Cleaning up partial initialization state...")
        
        # Cleanup components that may have been partially initialized
        cleanup_tasks = []
        
        if self.memory and hasattr(self.memory, 'shutdown'):
            cleanup_tasks.append(self.memory.shutdown())
        if self.executor and hasattr(self.executor, 'shutdown'):
            cleanup_tasks.append(self.executor.shutdown())
        if self.planner and hasattr(self.planner, 'shutdown'):
            cleanup_tasks.append(self.planner.shutdown())
        
        # Execute cleanup tasks with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=30.0  # 30 second cleanup timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Cleanup timeout - some resources may not have been properly released")
        
        # Reset initialization state
        self.is_initialized = False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system and cleanup resources."""
        logger.info("Shutting down Ocean Plastic Sentinel system...")
        
        # Complete any active missions
        if self.active_missions:
            logger.info(f"Waiting for {len(self.active_missions)} active missions to complete...")
            # Implementation would wait for missions to complete
        
        # Cleanup components
        if self.memory:
            await self.memory.shutdown()
        if self.executor:
            await self.executor.shutdown()
        if self.planner:
            await self.planner.shutdown()
        
        self.is_initialized = False
        logger.info("System shutdown complete")