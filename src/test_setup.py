"""
Installation Test for Ocean Plastic Sentinel

This script verifies that all components can be imported and basic functionality works.
Run this after setup to ensure everything is configured correctly.
"""
import sys
import asyncio
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import google.generativeai as genai
        print("  [OK] Google Generative AI library")
    except ImportError:
        print("  [FAIL] Google Generative AI library - run: pip install google-generativeai")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  [OK] Python dotenv")
    except ImportError:
        print("  [FAIL] Python dotenv - run: pip install python-dotenv")
        return False
    
    try:
        import requests
        print("  [OK] Requests library")
    except ImportError:
        print("  [FAIL] Requests library - run: pip install requests")
        return False
    
    try:
        from src.config import config
        print("  [OK] Ocean Sentinel configuration")
    except ImportError as e:
        print(f"  [FAIL] Ocean Sentinel configuration - {e}")
        return False
    
    try:
        from src.core import OceanPlasticSentinel
        print("  [OK] Ocean Sentinel core components")
    except ImportError as e:
        print(f"  [FAIL] Ocean Sentinel core components - {e}")
        return False
    
    return True

def test_configuration():
    """Test that configuration is properly loaded."""
    print("\nTesting configuration...")
    
    try:
        from src.config import config
        
        # Check if .env file exists
        if not Path('.env').exists():
            print("  [FAIL] .env file not found - run setup.py first")
            return False
        
        print("  [OK] .env file exists")
        
        # Check API key (without revealing it)
        if config.api.gemini_api_key and config.api.gemini_api_key != "your_api_key_here":
            print("  [OK] Gemini API key configured")
        else:
            print("  [FAIL] Gemini API key not configured - edit .env file")
            return False
        
        # Test configuration validation
        if config.validate():
            print("  [OK] Configuration validation passed")
        else:
            print("  [FAIL] Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Configuration test failed: {e}")
        return False

async def test_system_initialization():
    """Test that the main system can initialize."""
    print("\nTesting system initialization...")
    
    try:
        from src.core import OceanPlasticSentinel
        
        # Create sentinel instance
        sentinel = OceanPlasticSentinel()
        print("  [OK] OceanPlasticSentinel instance created")
        
        # Test initialization (this will test Gemini API connection)
        if await sentinel.initialize():
            print("  [OK] System initialization successful")
            print("  [OK] Gemini API connection working")
            
            # Get system status
            status = await sentinel.get_system_status()
            print(f"  [OK] System status: {status['initialized']}")
            
            # Cleanup
            await sentinel.shutdown()
            print("  [OK] System shutdown completed")
            
            return True
        else:
            print("  [FAIL] System initialization failed - check API key and network connection")
            return False
            
    except Exception as e:
        print(f"  [FAIL] System test failed: {e}")
        return False

def display_test_results(import_ok, config_ok, system_ok):
    """Display final test results and next steps."""
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    print(f"Imports: {'PASS' if import_ok else 'FAIL'}")
    print(f"Configuration: {'PASS' if config_ok else 'FAIL'}")
    print(f"System: {'PASS' if system_ok else 'FAIL'}")
    
    if import_ok and config_ok and system_ok:
        print("\nALL TESTS PASSED!")
        print("Your Ocean Plastic Sentinel is ready for missions!")
        print("\nTry running a test mission:")
        print("   python -m src.main")
    else:
        print("\nSOME TESTS FAILED")
        print("Please fix the issues above before running missions.")
        
        if not import_ok:
            print("- Install missing dependencies with: pip install -r requirements.txt")
        if not config_ok:
            print("- Configure your .env file with valid API keys")
        if not system_ok:
            print("- Check your internet connection and API key validity")

async def main():
    """Run all tests."""
    print("Ocean Plastic Sentinel - Installation Test")
    print("=" * 50)
    
    # Run tests
    import_ok = test_imports()
    config_ok = test_configuration() if import_ok else False
    system_ok = await test_system_initialization() if config_ok else False
    
    # Display results
    display_test_results(import_ok, config_ok, system_ok)

if __name__ == "__main__":
    asyncio.run(main())