"""
Memory System for Ocean Plastic Sentinel

This module handles data persistence, learning from mission outcomes, and providing
historical context for strategic decisions. It maintains both short-term operational
data and long-term patterns that improve system performance over time.
"""
import logging
import sqlite3
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from src.config import config


logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Persistent memory and learning system for ocean cleanup operations.
    
    This system stores mission data, learns from prediction accuracy, and provides
    historical context to improve future operations. It acts as the institutional
    memory that makes the system smarter over time.
    """
    
    def __init__(self, db_path: str = "data/ocean_sentinel.db"):
        """Initialize the memory system with SQLite database."""
        self.db_path = Path(db_path)
        self.connection = None
        self.is_initialized = False
        
        # Learning parameters that evolve over time
        self.learning_state = {
            'total_missions': 0,
            'successful_predictions': 0,
            'average_accuracy': 0.0,
            'drift_coefficient_alpha': config.system.default_alpha_coefficient,
            'drift_coefficient_beta': config.system.default_beta_coefficient,
            'last_learning_update': None
        }
        
        # Cache for frequently accessed data
        self.regional_cache = {}
        self.cache_ttl_hours = 6
    
    async def initialize(self) -> bool:
        """Initialize the database and create required tables."""
        try:
            logger.info("Initializing Memory System...")
            
            # Ensure data directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create database connection
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create database schema
            await self._create_database_schema()
            
            # Load learning state from database
            await self._load_learning_state()
            
            self.is_initialized = True
            logger.info("Memory System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Memory System initialization failed: {str(e)}")
            return False
    
    async def _create_database_schema(self) -> None:
        """Create the database tables for storing mission data."""
        cursor = self.connection.cursor()
        
        # Missions table - stores basic mission information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                mission_id TEXT PRIMARY KEY,
                target_region_json TEXT NOT NULL,
                priority_level TEXT NOT NULL,
                vessel_constraints_json TEXT,
                time_window_hours INTEGER,
                requested_by TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                execution_time_seconds REAL
            )
        """)
        
        # Detections table - stores all debris detections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id TEXT PRIMARY KEY,
                mission_id TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                confidence REAL NOT NULL,
                estimated_size_m2 REAL,
                debris_type TEXT,
                detection_timestamp TIMESTAMP,
                FOREIGN KEY (mission_id) REFERENCES missions (mission_id)
            )
        """)
        
        # Predictions table - stores drift predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                detection_id TEXT NOT NULL,
                mission_id TEXT NOT NULL,
                initial_lat REAL NOT NULL,
                initial_lon REAL NOT NULL,
                predicted_trajectory_json TEXT NOT NULL,
                time_horizon_hours INTEGER,
                model_version TEXT,
                coefficients_json TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (detection_id) REFERENCES detections (detection_id)
            )
        """)
        
        # Validations table - stores accuracy validation results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validations (
                validation_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                mission_id TEXT NOT NULL,
                validation_timestamp TIMESTAMP,
                spatial_accuracy REAL,
                temporal_accuracy REAL,
                actual_position_json TEXT,
                prediction_error_km REAL,
                validation_method TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id)
            )
        """)
        
        # Learning state table - stores system learning parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_state (
                parameter_name TEXT PRIMARY KEY,
                parameter_value REAL,
                last_updated TIMESTAMP,
                update_count INTEGER DEFAULT 1
            )
        """)
        
        # Routes table - stores optimized routes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                route_id TEXT PRIMARY KEY,
                mission_id TEXT NOT NULL,
                target_prediction_id TEXT,
                waypoints_json TEXT NOT NULL,
                estimated_fuel_consumption REAL,
                estimated_travel_time_hours REAL,
                optimization_target TEXT,
                created_at TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("Database schema created successfully")
    
    async def store_mission_results(self, mission_id: str, results: Dict[str, Any]) -> None:
        """Store complete mission results in the database."""
        if not self.is_initialized:
            raise RuntimeError("MemorySystem not initialized")
        
        cursor = self.connection.cursor()
        
        try:
            # Store mission record
            cursor.execute("""
                INSERT OR REPLACE INTO missions 
                (mission_id, completed_at, status, execution_time_seconds)
                VALUES (?, ?, ?, ?)
            """, (
                mission_id,
                datetime.utcnow(),
                'completed',
                0  # Will be updated by caller
            ))
            
            # Store detections
            for detection in results.get('detections', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO detections
                    (detection_id, mission_id, latitude, longitude, confidence,
                     estimated_size_m2, debris_type, detection_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection.get('id', f"{mission_id}_det_{len(results.get('detections', []))}"),
                    mission_id,
                    detection['latitude'],
                    detection['longitude'],
                    detection['confidence'],
                    detection.get('estimated_size_m2'),
                    detection.get('debris_type'),
                    datetime.utcnow()
                ))
            
            # Store predictions
            for prediction in results.get('predictions', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions
                    (prediction_id, detection_id, mission_id, initial_lat, initial_lon,
                     predicted_trajectory_json, time_horizon_hours, model_version,
                     coefficients_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{mission_id}_pred_{prediction.get('detection_id', '')}",
                    prediction.get('detection_id'),
                    mission_id,
                    prediction['initial_position']['latitude'],
                    prediction['initial_position']['longitude'],
                    json.dumps(prediction['trajectory']),
                    prediction['time_horizon_hours'],
                    'adaptive_physics_v1',
                    json.dumps(results.get('coefficients_used', {})),
                    datetime.utcnow()
                ))
            
            # Store routes
            for route in results.get('routes', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO routes
                    (route_id, mission_id, target_prediction_id, waypoints_json,
                     estimated_fuel_consumption, estimated_travel_time_hours,
                     optimization_target, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    route['route_id'],
                    mission_id,
                    route.get('target_prediction_id'),
                    json.dumps(route['waypoints']),
                    route['estimated_fuel_consumption'],
                    route['estimated_travel_time_hours'],
                    'fuel_efficiency',
                    datetime.utcnow()
                ))
            
            self.connection.commit()
            
            # Update learning state
            await self._update_mission_statistics()
            
            logger.info(f"Mission results stored: {mission_id}")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to store mission results: {str(e)}")
            raise
    
    async def update_predictive_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Update predictive model parameters based on recent results."""
        if not results.get('detections'):
            return {'message': 'No detections to learn from'}
        
        # Calculate detection density for this mission
        detection_count = len(results['detections'])
        avg_confidence = sum(d['confidence'] for d in results['detections']) / detection_count
        
        # Update learning parameters
        updates = {
            'detection_density': detection_count,
            'average_confidence': avg_confidence,
            'learning_timestamp': datetime.utcnow().isoformat()
        }
        
        # Adaptive learning: adjust confidence if consistently high or low
        if avg_confidence > 0.85:
            # High confidence - can be slightly more aggressive
            self.learning_state['drift_coefficient_alpha'] *= 1.02
        elif avg_confidence < 0.6:
            # Low confidence - be more conservative
            self.learning_state['drift_coefficient_alpha'] *= 0.98
        
        # Save updated learning parameters
        await self._save_learning_state()
        
        logger.info(f"Predictive models updated with {detection_count} detections")
        return updates
    
    async def get_regional_history(self, region: Dict[str, float]) -> Dict[str, Any]:
        """Get historical data for a specific region."""
        region_key = f"{region['north']:.1f},{region['south']:.1f},{region['east']:.1f},{region['west']:.1f}"
        
        # Check cache first
        if region_key in self.regional_cache:
            cache_entry = self.regional_cache[region_key]
            if datetime.utcnow() - cache_entry['timestamp'] < timedelta(hours=self.cache_ttl_hours):
                return cache_entry['data']
        
        cursor = self.connection.cursor()
        
        # Find missions in this region (with some tolerance)
        cursor.execute("""
            SELECT m.mission_id, m.created_at, m.status, 
                   COUNT(d.detection_id) as detection_count,
                   AVG(d.confidence) as avg_confidence
            FROM missions m
            LEFT JOIN detections d ON m.mission_id = d.mission_id
            WHERE m.target_region_json LIKE ?
            GROUP BY m.mission_id
            ORDER BY m.created_at DESC
            LIMIT 10
        """, (f'%{region["north"]:.0f}%',))  # Simplified region matching
        
        historical_missions = cursor.fetchall()
        
        # Calculate regional statistics
        if historical_missions:
            total_detections = sum(row['detection_count'] or 0 for row in historical_missions)
            avg_confidence = np.mean([row['avg_confidence'] or 0.5 for row in historical_missions])
            success_rate = len([r for r in historical_missions if r['status'] == 'completed']) / len(historical_missions)
        else:
            total_detections = 0
            avg_confidence = 0.5
            success_rate = 0.0
        
        history_data = {
            'region': region,
            'historical_missions': len(historical_missions),
            'total_detections': total_detections,
            'average_confidence': avg_confidence,
            'success_rate': success_rate,
            'detection_density': total_detections / max(len(historical_missions), 1)
        }
        
        # Cache the result
        self.regional_cache[region_key] = {
            'data': history_data,
            'timestamp': datetime.utcnow()
        }
        
        return history_data
    
    async def get_current_conditions(self, region: Dict[str, float]) -> Dict[str, Any]:
        """Get current environmental conditions for the region."""
        # In a real system, this would fetch real-time data
        # For now, return reasonable defaults
        
        return {
            'region': region,
            'weather_conditions': 'moderate',
            'sea_state': 'calm',
            'visibility': 'good',
            'optimal_detection_window': True,
            'current_timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_learning_parameters(self) -> Dict[str, Any]:
        """Get current learning parameters for system optimization."""
        return {
            'drift_coefficients': {
                'alpha': self.learning_state['drift_coefficient_alpha'],
                'beta': self.learning_state['drift_coefficient_beta']
            },
            'performance_metrics': {
                'total_missions': self.learning_state['total_missions'],
                'average_accuracy': self.learning_state['average_accuracy'],
                'successful_predictions': self.learning_state['successful_predictions']
            },
            'last_update': self.learning_state['last_learning_update']
        }
    
    async def update_from_validation(self, mission_id: str, validation_result: Dict[str, Any]) -> None:
        """Update learning parameters based on validation results."""
        if not self.is_initialized:
            return
        
        cursor = self.connection.cursor()
        
        try:
            # Store validation result
            validation_id = f"val_{mission_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute("""
                INSERT INTO validations
                (validation_id, prediction_id, mission_id, validation_timestamp,
                 spatial_accuracy, temporal_accuracy, validation_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                validation_id,
                f"{mission_id}_pred",
                mission_id,
                datetime.utcnow(),
                validation_result.get('spatial_accuracy', 0.0),
                validation_result.get('temporal_accuracy', 0.0),
                'satellite_comparison'
            ))
            
            # Update learning coefficients based on accuracy
            spatial_accuracy = validation_result.get('spatial_accuracy', 0.5)
            
            if spatial_accuracy > 0.8:
                # Good prediction - reinforce current coefficients
                self.learning_state['successful_predictions'] += 1
            elif spatial_accuracy < 0.5:
                # Poor prediction - adjust coefficients
                self.learning_state['drift_coefficient_alpha'] *= 0.95
                self.learning_state['drift_coefficient_beta'] *= 0.95
            
            # Update average accuracy
            total = self.learning_state['total_missions']
            current_avg = self.learning_state['average_accuracy']
            self.learning_state['average_accuracy'] = (
                (current_avg * total + spatial_accuracy) / (total + 1)
            )
            
            self.connection.commit()
            await self._save_learning_state()
            
            logger.info(f"Learning updated from validation: {spatial_accuracy:.2f} accuracy")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to update from validation: {str(e)}")
    
    async def retrieve_mission_data(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete data for a specific mission."""
        if not self.is_initialized:
            return None
        
        cursor = self.connection.cursor()
        
        try:
            # Get mission basic info
            cursor.execute("""
                SELECT * FROM missions WHERE mission_id = ?
            """, (mission_id,))
            
            mission_row = cursor.fetchone()
            if not mission_row:
                return None
            
            # Get detections
            cursor.execute("""
                SELECT * FROM detections WHERE mission_id = ?
            """, (mission_id,))
            detections = [dict(row) for row in cursor.fetchall()]
            
            # Get predictions
            cursor.execute("""
                SELECT * FROM predictions WHERE mission_id = ?
            """, (mission_id,))
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Get routes
            cursor.execute("""
                SELECT * FROM routes WHERE mission_id = ?
            """, (mission_id,))
            routes = [dict(row) for row in cursor.fetchall()]
            
            return {
                'mission': dict(mission_row),
                'detections': detections,
                'predictions': predictions,
                'routes': routes
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve mission data: {str(e)}")
            return None
    
    async def _load_learning_state(self) -> None:
        """Load learning parameters from database."""
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT parameter_name, parameter_value FROM learning_state")
        params = cursor.fetchall()
        
        for param in params:
            if param['parameter_name'] in self.learning_state:
                self.learning_state[param['parameter_name']] = param['parameter_value']
        
        if not params:
            # Initialize with defaults
            await self._save_learning_state()
    
    async def _save_learning_state(self) -> None:
        """Save current learning parameters to database."""
        cursor = self.connection.cursor()
        
        for param_name, param_value in self.learning_state.items():
            if isinstance(param_value, (int, float)):
                cursor.execute("""
                    INSERT OR REPLACE INTO learning_state
                    (parameter_name, parameter_value, last_updated)
                    VALUES (?, ?, ?)
                """, (param_name, param_value, datetime.utcnow()))
        
        self.connection.commit()
    
    async def _update_mission_statistics(self) -> None:
        """Update global mission statistics."""
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM missions WHERE status = 'completed'")
        total_missions = cursor.fetchone()['total']
        
        self.learning_state['total_missions'] = total_missions
        self.learning_state['last_learning_update'] = datetime.utcnow().isoformat()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current memory system status."""
        cursor = self.connection.cursor()
        
        # Get database statistics
        cursor.execute("SELECT COUNT(*) as count FROM missions")
        mission_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM detections")
        detection_count = cursor.fetchone()['count']
        
        return {
            'initialized': self.is_initialized,
            'database_path': str(self.db_path),
            'total_missions': mission_count,
            'total_detections': detection_count,
            'learning_state': self.learning_state.copy(),
            'cache_size': len(self.regional_cache)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the memory system and close database connection."""
        logger.info("Shutting down Memory System...")
        
        if self.connection:
            # Save final learning state
            await self._save_learning_state()
            
            # Close database connection
            self.connection.close()
            self.connection = None
        
        # Clear caches
        self.regional_cache.clear()
        
        self.is_initialized = False
        logger.info("Memory System shutdown complete")