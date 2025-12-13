"""
Gemini API integration for Ocean Plastic Sentinel.

This module provides a clean interface for interacting with Google's Gemini API,
specifically optimized for satellite imagery analysis and marine debris detection.
"""
from datetime import datetime
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from PIL import Image
import google.generativeai as genai

from src.config import config


logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for plastic debris detection results."""
    
    latitude: float
    longitude: float
    confidence: float
    debris_type: str
    estimated_size_m2: Optional[float] = None
    description: Optional[str] = None


@dataclass
class AnalysisRequest:
    """Container for image analysis request parameters."""
    
    image_data: Union[Image.Image, str]  # PIL Image or file path
    analysis_type: str  # 'detection', 'validation', 'drift_assessment'
    region_bounds: Optional[Dict[str, float]] = None  # lat/lon boundaries
    priority_level: str = 'standard'  # 'urgent', 'standard', 'background'


class GeminiAPIClient:
    """
    High-level client for Gemini API interactions.
    
    This class encapsulates all Gemini API communication and provides
    specialized methods for marine debris analysis tasks.
    """
    
    def __init__(self):
        """Initialize the Gemini API client."""
        self._model = None
        self._is_initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the Gemini API connection.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
             # TEST MODE - Skip real API initialization
            if config.system.test_mode:
                logger.info("🧪 TEST MODE: Using mock Gemini responses (no API calls)")
                self._is_initialized = True
                self._model = None  # No real model needed
                return True
            
            if not config.api.gemini_api_key:
                logger.error("Gemini API key not found in configuration")
                return False
                
            genai.configure(api_key=config.api.gemini_api_key)
            self._model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Test connection with a simple request
            await self._test_connection()
            
            self._is_initialized = True
            logger.info(f"Gemini API initialized successfully with model: {config.api.gemini_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            return False
    
    async def _test_connection(self) -> None:
        """Test the API connection with a simple request."""
        try:
            logger.info("Testing Gemini API connection...")
            
            # Add timeout
            response = await asyncio.wait_for(
                self._model.generate_content_async("Hello"),
                timeout=10.0  # 10 second timeout
            )
            
            logger.info("Gemini API connection test successful")
            
        except asyncio.TimeoutError:
            logger.error("Gemini API connection test timed out after 10 seconds")
            raise Exception("Gemini API connection test failed: Timeout")
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {str(e)}")
            raise Exception(f"Gemini API connection test failed: {str(e)}")
    
    async def analyze_satellite_image(self, request: AnalysisRequest) -> List[DetectionResult]:
        """
        Analyze satellite imagery for plastic debris detection.
        
        Args:
            request: Analysis request with image data and parameters
            
        Returns:
            List of DetectionResult objects containing detected debris locations
        """
        # TEST MODE - Return realistic mock data
        if config.system.test_mode or self._model is None:
            logger.info("🧪 Returning mock detection results")
            
            # Extract location from region_bounds if available
            if request.region_bounds:
                center_lat = (request.region_bounds.get('north', 0) + request.region_bounds.get('south', 0)) / 2
                center_lon = (request.region_bounds.get('east', 0) + request.region_bounds.get('west', 0)) / 2
            else:
                center_lat, center_lon = 0.0, 0.0
            
            return DetectionResult(
                latitude=center_lat,
                longitude=center_lon,
                confidence=0.87,
                debris_type="mixed_plastics",
                estimated_size_m2=3200.0,  # 3.2 km² = 3,200,000 m²
                description="High plastic concentration detected - Priority cleanup area"
            )
        
        if not self._is_initialized:
            raise RuntimeError("GeminiAPIClient not initialized. Call initialize() first.")
        
        try:
            prompt = self._build_detection_prompt(request)
            
            # Prepare image for analysis
            image = await self._prepare_image(request.image_data)
            
            # Execute analysis
            response = await asyncio.to_thread(
                self._model.generate_content,
                [prompt, image]
            )
            
            # Parse response into structured results
            detections = await self._parse_detection_response(response.text, request.region_bounds)
            
            logger.info(f"Detected {len(detections)} potential debris locations")
            return detections
            
        except Exception as e:
            logger.error(f"Satellite image analysis failed: {str(e)}")
            raise
    
    async def validate_prediction(self, 
                                 prediction_coords: tuple, 
                                 validation_image: Union[Image.Image, str]) -> Dict[str, Any]:
        """
        Validate a previous debris prediction against new imagery.
        
        Args:
            prediction_coords: (latitude, longitude) of predicted location
            validation_image: New satellite image for validation
            
        Returns:
            Dictionary containing validation results and accuracy metrics
        """
        if not self._is_initialized:
            raise RuntimeError("GeminiAPIClient not initialized. Call initialize() first.")
        
        try:
            prompt = self._build_validation_prompt(prediction_coords)
            image = await self._prepare_image(validation_image)
            
            response = await asyncio.to_thread(
                self._model.generate_content,
                [prompt, image]
            )
            
            return await self._parse_validation_response(response.text, prediction_coords)
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {str(e)}")
            raise
    
    def _build_detection_prompt(self, request: AnalysisRequest) -> str:
        """Build the detection prompt for Gemini analysis."""
        base_prompt = """
        Analyze this Sentinel-2 satellite image for marine plastic debris detection.
        
        DETECTION CRITERIA:
        - Look for floating plastic accumulations (bright reflective patches)
        - Distinguish from natural phenomena (foam, algae, sediment)
        - Focus on areas with unnatural geometric patterns
        - Consider spectral signatures typical of plastic materials
        
        OUTPUT FORMAT:
        For each detection, provide:
        - Latitude/Longitude coordinates (decimal degrees)
        - Confidence score (0.0-1.0)
        - Debris type classification
        - Estimated size in square meters
        - Brief description of visual indicators
        
        Return results in structured JSON format.
        """
        
        if request.region_bounds:
            bounds_info = f"""
            SEARCH REGION BOUNDS:
            North: {request.region_bounds.get('north')}°
            South: {request.region_bounds.get('south')}°  
            East: {request.region_bounds.get('east')}°
            West: {request.region_bounds.get('west')}°
            """
            base_prompt += bounds_info
        
        if request.priority_level == 'urgent':
            base_prompt += "\nPRIORITY: URGENT - Focus on largest, most accessible debris patches."
        
        return base_prompt
    
    def _build_validation_prompt(self, coords: tuple) -> str:
        """Build the validation prompt for prediction checking."""
        lat, lon = coords
        return f"""
        Validate marine debris prediction at coordinates: {lat}°, {lon}°
        
        VALIDATION TASK:
        - Examine the specified location for debris presence
        - Compare against typical plastic debris characteristics
        - Assess whether prediction was accurate
        - Identify any changes since prediction was made
        
        OUTPUT FORMAT:
        - Debris present: Yes/No
        - Accuracy assessment: High/Medium/Low
        - Change description: What has changed since prediction
        - Confidence in validation: 0.0-1.0
        
        Return results in structured JSON format.
        """
    
    async def _prepare_image(self, image_data: Union[Image.Image, str]) -> Image.Image:
        """Prepare image data for Gemini analysis."""
        if isinstance(image_data, str):
            # Load from file path
            image = Image.open(image_data)
        else:
            image = image_data
        
        # Ensure image is in compatible format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    async def _parse_detection_response(self, 
                                       response_text: str, 
                                       region_bounds: Optional[Dict[str, float]]) -> List[DetectionResult]:
        """Parse Gemini response into structured detection results."""
        # This is a simplified parser - in production, you'd want robust JSON parsing
        # with error handling and validation
        
        detections = []
        try:
            # Extract structured data from response
            # Implementation would depend on actual Gemini response format
            logger.debug("Parsing detection response from Gemini")
            
            # Placeholder implementation - replace with actual parsing logic
            # This would parse JSON or structured text from Gemini response
            
        except Exception as e:
            logger.warning(f"Failed to parse detection response: {str(e)}")
        
        return detections
    
    async def _parse_validation_response(self, 
                                        response_text: str, 
                                        coords: tuple) -> Dict[str, Any]:
        """Parse validation response into structured results."""
        try:
            # Parse validation results from Gemini response
            # Implementation would extract accuracy metrics and validation status
            
            return {
                'prediction_accurate': False,  # Placeholder
                'confidence': 0.0,
                'debris_present': False,
                'changes_detected': False,
                'validation_timestamp': None
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse validation response: {str(e)}")
            return {
                'prediction_accurate': None,
                'confidence': 0.0,
                'error': str(e)
            }