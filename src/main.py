"""
Ocean Plastic Sentinel - Mission Launcher

This is the main entry point for running ocean cleanup missions.
It provides examples of how to use the system and validates that everything works.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.core import OceanPlasticSentinel, MissionRequest


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ocean_sentinel.log')
    ]
)

logger = logging.getLogger(__name__)


async def run_demo_mission():
    """
    Run a demonstration mission in the Pacific Ocean.
    
    This example shows how to:
    1. Create a mission request for a specific ocean region
    2. Execute the mission using the Ocean Plastic Sentinel system
    3. Review the results and recommendations
    """
    print("Ocean Plastic Sentinel - Demo Mission")
    print("="*50)
    
    # Create the Ocean Plastic Sentinel system
    sentinel = OceanPlasticSentinel()
    
    try:
        # Initialize the system
        print("Initializing system...")
        if not await sentinel.initialize():
            print("[ERROR] System initialization failed")
            print("Check logs/ocean_sentinel.log for details")
            return
        
        print("[OK] System initialized successfully!")
        
        # Create a demo mission for the Pacific Ocean
        # This area is known to have plastic debris accumulation
        demo_mission = MissionRequest(
            mission_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            target_region={
                'north': 40.0,    # Northern Pacific
                'south': 35.0,
                'east': -140.0,   # West of California
                'west': -150.0
            },
            priority_level='standard',
            vessel_constraints={
                'max_speed_knots': 12,
                'fuel_capacity_liters': 50000,
                'cargo_capacity_m3': 1000
            },
            time_window_hours=48,
            requested_by='demo_user'
        )
        
        print(f"[TARGET] Starting demo mission: {demo_mission.mission_id}")
        print(f"[REGION] Target region: {demo_mission.target_region}")
        
        # Execute the mission
        print("[EXECUTE] Executing mission...")
        result = await sentinel.execute_mission(demo_mission)
        
        # Display results
        print("\nMISSION RESULTS")
        print("-" * 30)
        print(f"Mission ID: {result.mission_id}")
        print(f"Status: {result.status}")
        print(f"Execution Time: {result.execution_time_seconds:.2f} seconds")
        print(f"Detections Found: {len(result.detections)}")
        print(f"Predictions Generated: {len(result.predictions)}")
        print(f"Routes Recommended: {len(result.recommended_routes)}")
        
        if result.confidence_metrics:
            print(f"Confidence Metrics: {result.confidence_metrics}")
        
        # Save results to file
        output_dir = Path('outputs/missions')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f"{result.mission_id}_results.json"
        print(f"[SAVED] Results saved to: {result_file}")
        
        # Display system status
        status = await sentinel.get_system_status()
        print(f"\n[STATUS] System Status: {status}")
        
        if result.status == 'completed':
            print("\n[SUCCESS] Demo mission completed successfully!")
            print("[NEXT] You can now run your own missions by modifying the target_region coordinates")
        else:
            print(f"\n[WARNING] Mission completed with status: {result.status}")
            print("[INFO] Check the logs for detailed information")
        
    except Exception as e:
        logger.error(f"Demo mission failed: {e}")
        print(f"[ERROR] Demo mission failed: {e}")
        print("[INFO] Check logs/ocean_sentinel.log for detailed error information")
    
    finally:
        # Clean shutdown
        print("\n[SHUTDOWN] Shutting down system...")
        await sentinel.shutdown()
        print("[OK] System shutdown complete")


def display_mission_examples():
    """Display examples of different mission types users can run."""
    print("\n" + "="*60)
    print("MISSION EXAMPLES")
    print("="*60)
    
    examples = [
        {
            'name': 'Pacific Gyre Monitoring',
            'region': {'north': 42, 'south': 32, 'east': -140, 'west': -160},
            'priority': 'standard',
            'description': 'Monitor the Great Pacific Garbage Patch'
        },
        {
            'name': 'Mediterranean Cleanup',
            'region': {'north': 45, 'south': 35, 'east': 20, 'west': 0},
            'priority': 'high',
            'description': 'Detect debris in Mediterranean shipping lanes'
        },
        {
            'name': 'Emergency Response',
            'region': {'north': 30, 'south': 25, 'east': -80, 'west': -90},
            'priority': 'urgent',
            'description': 'Rapid response for reported plastic spill'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']} ({example['priority']} priority)")
        print(f"   Region: {example['region']}")
        print(f"   Use case: {example['description']}")
    
    print(f"\n[INFO] To run custom missions, modify the MissionRequest in this file")
    print(f"[DOCS] See ARCHITECTURE.md for detailed system documentation")


async def main():
    """Main entry point for the Ocean Plastic Sentinel application."""
    try:
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Run the demo mission
        await run_demo_mission()
        
        # Show additional examples
        display_mission_examples()
        
    except KeyboardInterrupt:
        print("\n[CANCELLED] Mission cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"[ERROR] Application error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
