"""
End-to-end workflow test for Ocean Plastic Sentinel.
Tests the complete multi-agent system workflow from initialization to completion.
"""
import unittest
import asyncio
import os
import tempfile
import sqlite3
from unittest.mock import Mock, patch, AsyncMock
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent import OceanPlasticSentinel
from src.core.memory import MemorySystem
from src.integrations.gemini_client import GeminiAPIClient


class TestE2EWorkflow(unittest.TestCase):
    """End-to-end workflow tests for Ocean Plastic Sentinel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test_api_key_12345',
            'DATABASE_URL': f'sqlite:///{self.temp_db.name}',
            'NOAA_API_KEY': 'test_noaa_key'
        })
        self.env_patcher.start()
        
        # Initialize memory system with temp database
        self.memory = MemorySystem(self.temp_db.name)
        
        # Mock Gemini client responses
        self.mock_gemini_responses = {
            'analyze_satellite_image': {
                'debris_detected': True,
                'debris_locations': [
                    {'lat': 25.7617, 'lon': -80.1918, 'confidence': 0.85, 'type': 'plastic_bottles'},
                    {'lat': 25.7620, 'lon': -80.1920, 'confidence': 0.78, 'type': 'plastic_bags'}
                ],
                'analysis': 'Detected plastic debris concentrated in convergence zone'
            },
            'validate_prediction': {
                'validation_result': True,
                'accuracy_score': 0.82,
                'analysis': 'Prediction accuracy within acceptable range'
            }
        }
        
        # Mock NOAA API responses
        self.mock_noaa_response = {
            'current_speed': 0.5,
            'current_direction': 45,
            'wind_speed': 10,
            'wind_direction': 90,
            'wave_height': 2.0
        }
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        os.unlink(self.temp_db.name)
    
    @patch('aiohttp.ClientSession.get')
    @patch('src.integrations.gemini_client.GeminiAPIClient.analyze_satellite_image')
    @patch('src.integrations.gemini_client.GeminiAPIClient.validate_prediction')
    async def test_complete_mission_workflow(self, mock_validate, mock_analyze, mock_http_get):
        """Test complete mission execution workflow."""
        print("\n=== Starting End-to-End Workflow Test ===")
        
        # Setup mocks
        mock_analyze.return_value = self.mock_gemini_responses['analyze_satellite_image']
        mock_validate.return_value = self.mock_gemini_responses['validate_prediction']
        
        # Mock NOAA API response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=self.mock_noaa_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_http_get.return_value = mock_response
        
        # Initialize agent
        agent = OceanPlasticSentinel()
        
        # Test Phase 1: Initialization
        print("Phase 1: Testing system initialization...")
        await agent.initialize()
        
        # Verify components are initialized
        self.assertIsNotNone(agent.planner)
        self.assertIsNotNone(agent.executor)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.gemini_client)
        print("✓ System initialization successful")
        
        # Test Phase 2: Mission Execution
        print("\nPhase 2: Testing mission execution...")
        
        # Define test mission parameters
        mission_params = {
            'region': 'north_atlantic_gyre',
            'coordinates': {'lat': 25.7617, 'lon': -80.1918},
            'search_radius_km': 50,
            'mission_type': 'debris_detection'
        }
        
        # Execute mission
        result = await agent.execute_mission(mission_params)
        
        # Verify mission execution
        self.assertIsNotNone(result)
        self.assertIn('mission_id', result)
        self.assertIn('tasks_completed', result)
        self.assertIn('detections', result)
        self.assertIn('drift_predictions', result)
        
        # Verify detections were found
        self.assertGreater(len(result['detections']), 0)
        detection = result['detections'][0]
        self.assertIn('lat', detection)
        self.assertIn('lon', detection)
        self.assertIn('confidence', detection)
        
        print(f"✓ Mission executed successfully with {len(result['detections'])} detections")
        
        # Test Phase 3: Data Persistence
        print("\nPhase 3: Testing data persistence...")
        
        # Verify mission was stored in database
        missions = await agent.memory.get_recent_missions(limit=1)
        self.assertEqual(len(missions), 1)
        
        stored_mission = missions[0]
        self.assertEqual(stored_mission['region'], mission_params['region'])
        self.assertEqual(stored_mission['status'], 'completed')
        
        # Verify detections were stored
        detections = await agent.memory.get_detections_for_mission(stored_mission['id'])
        self.assertGreater(len(detections), 0)
        
        print("✓ Data persistence verified")
        
        # Test Phase 4: Learning System
        print("\nPhase 4: Testing adaptive learning...")
        
        # Simulate validation of previous predictions
        await agent.validate_previous_predictions()
        
        # Verify learning state was updated
        learning_state = await agent.memory.get_learning_state()
        self.assertIsNotNone(learning_state)
        self.assertIn('drift_coefficients', learning_state)
        
        print("✓ Adaptive learning system verified")
        
        # Test Phase 5: Multi-Mission Workflow
        print("\nPhase 5: Testing multi-mission workflow...")
        
        # Execute second mission in different region
        mission_params_2 = {
            'region': 'pacific_gyre',
            'coordinates': {'lat': 35.0, 'lon': -140.0},
            'search_radius_km': 75,
            'mission_type': 'route_optimization'
        }
        
        result_2 = await agent.execute_mission(mission_params_2)
        
        # Verify second mission
        self.assertIsNotNone(result_2)
        self.assertNotEqual(result['mission_id'], result_2['mission_id'])
        
        # Verify we now have 2 missions in database
        all_missions = await agent.memory.get_recent_missions(limit=10)
        self.assertEqual(len(all_missions), 2)
        
        print("✓ Multi-mission workflow verified")
        
        # Test Phase 6: Performance Metrics
        print("\nPhase 6: Testing performance metrics...")
        
        # Calculate system performance
        total_detections = 0
        total_predictions = 0
        
        for mission in all_missions:
            mission_detections = await agent.memory.get_detections_for_mission(mission['id'])
            mission_predictions = await agent.memory.get_predictions_for_mission(mission['id'])
            total_detections += len(mission_detections)
            total_predictions += len(mission_predictions)
        
        print(f"✓ System processed {len(all_missions)} missions")
        print(f"✓ Total detections: {total_detections}")
        print(f"✓ Total predictions: {total_predictions}")
        
        # Verify system efficiency
        self.assertGreater(total_detections, 0)
        self.assertGreater(total_predictions, 0)
        
        print("\n=== End-to-End Workflow Test PASSED ===")
        
        return {
            'missions_executed': len(all_missions),
            'total_detections': total_detections,
            'total_predictions': total_predictions,
            'learning_active': learning_state is not None,
            'test_status': 'PASSED'
        }
    
    @patch('aiohttp.ClientSession.get')
    @patch('src.integrations.gemini_client.GeminiAPIClient.analyze_satellite_image')
    def test_error_handling_workflow(self, mock_analyze, mock_http_get):
        """Test error handling in workflow."""
        print("\n=== Testing Error Handling Workflow ===")
        
        # Setup error scenarios
        mock_analyze.side_effect = Exception("Simulated API error")
        mock_http_get.side_effect = Exception("Simulated network error")
        
        async def run_error_test():
            agent = OceanPlasticSentinel()
            await agent.initialize()
            
            mission_params = {
                'region': 'test_region',
                'coordinates': {'lat': 0.0, 'lon': 0.0},
                'search_radius_km': 10,
                'mission_type': 'debris_detection'
            }
            
            # This should handle errors gracefully
            try:
                result = await agent.execute_mission(mission_params)
                # Should still return a result with error information
                self.assertIsNotNone(result)
                print("✓ Error handling verified")
                return True
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
                return False
        
        # Run async error test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(run_error_test())
            self.assertTrue(success)
        finally:
            loop.close()
    
    def test_database_operations(self):
        """Test database operations independently."""
        print("\n=== Testing Database Operations ===")
        
        # Test mission storage
        mission_data = {
            'region': 'test_region',
            'coordinates': {'lat': 10.0, 'lon': 20.0},
            'status': 'completed',
            'parameters': {'test': 'param'}
        }
        
        async def run_db_test():
            mission_id = await self.memory.store_mission(mission_data)
            self.assertIsNotNone(mission_id)
            
            # Test detection storage
            detection_data = {
                'mission_id': mission_id,
                'latitude': 10.0,
                'longitude': 20.0,
                'confidence': 0.85,
                'debris_type': 'plastic_bottle',
                'analysis': 'Test detection'
            }
            
            detection_id = await self.memory.store_detection(detection_data)
            self.assertIsNotNone(detection_id)
            
            # Test retrieval
            missions = await self.memory.get_recent_missions(limit=1)
            self.assertEqual(len(missions), 1)
            
            detections = await self.memory.get_detections_for_mission(mission_id)
            self.assertEqual(len(detections), 1)
            
            print("✓ Database operations verified")
        
        # Run async database test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_db_test())
        finally:
            loop.close()


def run_e2e_test():
    """Run the end-to-end test and return results."""
    print("Ocean Plastic Sentinel - End-to-End Workflow Test")
    print("=" * 50)
    
    # Run main workflow test
    test_case = TestE2EWorkflow()
    test_case.setUp()
    
    try:
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(test_case.test_complete_mission_workflow())
            print(f"\nTest Results: {json.dumps(result, indent=2)}")
            return result
        finally:
            loop.close()
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        return {'test_status': 'FAILED', 'error': str(e)}
    
    finally:
        test_case.tearDown()


if __name__ == '__main__':
    # Run the test
    result = run_e2e_test()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if result.get('test_status') == 'PASSED':
        print("✓ End-to-end workflow test PASSED")
        print(f"✓ Missions executed: {result.get('missions_executed', 0)}")
        print(f"✓ Total detections: {result.get('total_detections', 0)}")
        print(f"✓ Learning system active: {result.get('learning_active', False)}")
    else:
        print("✗ End-to-end workflow test FAILED")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\nReady for hackathon submission!")