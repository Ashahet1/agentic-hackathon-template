"""
Configuration module for Ocean Plastic Sentinel.

This module manages all configuration settings, environment variables,
and system constants used throughout the application.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfiguration:
    """Configuration for external API integrations."""
    
    # Google Gemini API settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-pro"
    
    # NOAA API settings
    noaa_base_url: str = "https://api.tidesandcurrents.noaa.gov"
    
    # Sentinel-2 API settings (via Google Earth Engine)
    earth_engine_project: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.earth_engine_project = os.getenv("EARTH_ENGINE_PROJECT")


@dataclass
class SystemConfiguration:
    """Configuration for system behavior and limits."""
    test_mode: bool = True
    # Processing limits
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Memory management
    memory_retention_days: int = 30
    prediction_history_limit: int = 1000
    
    # Drift modeling parameters
    default_alpha_coefficient: float = 0.7  # Ocean current influence
    default_beta_coefficient: float = 0.3   # Wind influence
    learning_rate: float = 0.01
    
    # Geographic bounds (degrees)
    max_search_area_km2: float = 10000.0
    min_confidence_threshold: float = 0.6


@dataclass
class ApplicationConfiguration:
    """Main application configuration container."""
    
    api: APIConfiguration
    system: SystemConfiguration
    
    def __init__(self):
        self.api = APIConfiguration()
        self.system = SystemConfiguration()
    
    def validate(self) -> bool:
        """
        Validate that all required configuration values are present.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        if not self.api.gemini_api_key:
            return False
        
        if self.system.max_concurrent_requests <= 0:
            return False
            
        return True


# Global configuration instance
config = ApplicationConfiguration()