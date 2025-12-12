"""
Ocean Plastic Sentinel Setup Script

Run this script to set up the project on a new machine.
It will create the necessary configuration files and install dependencies.
"""
import os
import shutil
from pathlib import Path

def setup_environment():
    """Create .env file with required API keys."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """# Ocean Plastic Sentinel Configuration
# Copy this file and add your actual API keys

# Google Gemini API Key (Required)
# Get from: https://ai.google.dev/
GEMINI_API_KEY=your_api_key_here

# Google Earth Engine Project (Optional)
# Get from: https://earthengine.google.com/
EARTH_ENGINE_PROJECT=your_project_id_here

# System Configuration (Optional)
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT_SECONDS=30
"""
        env_file.write_text(env_content)
        print("✓ Created .env file - Please add your API keys")
    else:
        print("✓ .env file already exists")

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        'logs',
        'data/cache',
        'data/models',
        'outputs/missions',
        'outputs/routes'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_requirements():
    """Check if requirements.txt exists and guide user on installation."""
    if Path('requirements.txt').exists():
        print("✓ requirements.txt found")
        print("📌 Next step: Run 'pip install -r requirements.txt' to install dependencies")
    else:
        print("❌ requirements.txt not found - dependencies may not install properly")

def display_next_steps():
    """Display what user needs to do next."""
    print("\n" + "="*60)
    print("🌊 OCEAN PLASTIC SENTINEL - SETUP COMPLETE")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Get a Gemini API key from: https://ai.google.dev/")
    print("3. Edit .env file and add your GEMINI_API_KEY")
    print("4. Test the installation: python -m src.test_setup")
    print("5. Run your first mission: python -m src.main")
    print("\n🔧 TROUBLESHOOTING:")
    print("- If pip install fails, try: pip install --upgrade pip")
    print("- For Python environment issues, use: python -m venv ocean_sentinel")
    print("- Check logs/ directory for detailed error messages")
    print("\n📚 DOCUMENTATION:")
    print("- README.md: Project overview and mission examples")
    print("- ARCHITECTURE.md: Technical system design")
    print("- EXPLANATION.md: How the AI agents work")

if __name__ == "__main__":
    print("🌊 Setting up Ocean Plastic Sentinel...")
    
    setup_environment()
    create_directories()
    check_requirements()
    display_next_steps()