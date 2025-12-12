# Ocean Plastic Sentinel - Technical Explanation

## 1. Agent Workflow

### Step-by-Step Processing Flow

**Input**: Ocean region coordinates or "global scan" request

1. **Receive Mission Parameters**
   - Target ocean region (lat/lon bounds) or global monitoring request
   - Priority level (emergency cleanup vs. routine monitoring)
   - Available cleanup vessel information (location, capacity, speed)

2. **Scout Agent: Debris Detection Phase**
   - Retrieve latest Sentinel-2 satellite imagery for target region
   - Convert multispectral bands (NIR/SWIR) to Gemini-readable RGB pseudo-images
   - Use Gemini vision model to identify plastic debris signatures
   - Generate confidence-scored detection coordinates

3. **Navigator Agent: Drift Prediction Phase**
   - Fetch real-time NOAA ocean current and wind data
   - Apply physics-based drift modeling: `V_drift = α*V_current + β*V_wind`
   - Calculate 48-72 hour trajectory predictions for detected debris
   - Optimize intercept routes for available cleanup vessels

4. **Critic Agent: Validation & Learning Phase**
   - Compare previous predictions against new satellite observations
   - Calculate prediction accuracy and spatial error margins
   - Update drift coefficients (α, β) based on validation results
   - Store learning outcomes in long-term memory

5. **Final Output Generation**
   - Synthesize results from all three agents
   - Generate GeoJSON cleanup routes for vessel navigation systems
   - Provide confidence intervals and recommended action timelines
   - Export mission reports with cost-benefit analysis

## 2. Key Modules

### **Scout Agent** (`scout_agent.py`)
**Purpose**: Plastic debris detection from satellite imagery
- **Input Processing**: Downloads and preprocesses Sentinel-2 multispectral data
- **Spectral Analysis**: Converts invisible NIR/SWIR bands to RGB for Gemini processing
- **Pattern Recognition**: Uses Gemini's vision capabilities to identify plastic signatures
- **Confidence Scoring**: Assigns probability scores to potential debris patches
- **Spatial Clustering**: Groups nearby detections into actionable cleanup targets

### **Navigator Agent** (`navigator_agent.py`)  
**Purpose**: Oceanographic drift prediction and route optimization
- **Data Fusion**: Combines NOAA ocean currents, wind data, and debris characteristics
- **Physics Modeling**: Implements drift equations with adaptive coefficient learning
- **Trajectory Calculation**: Projects debris movement over 24-72 hour windows
- **Route Optimization**: Calculates fuel-efficient vessel intercept paths
- **Time Window Analysis**: Determines optimal cleanup scheduling

### **Critic Agent** (`critic_agent.py`)
**Purpose**: System validation and continuous improvement
- **Prediction Validation**: Compares forecasts against new satellite observations
- **Error Analysis**: Calculates spatial/temporal accuracy metrics
- **Parameter Tuning**: Adjusts detection thresholds and drift coefficients
- **Learning Integration**: Updates system knowledge based on real-world outcomes
- **Performance Reporting**: Tracks improvement trends over time

### **Memory System** (`memory_store.py`)
**Purpose**: Context preservation and learning persistence
- **Short-term Memory**: Active session tracking of detections and predictions
- **Long-term Memory**: Historical patterns, accuracy data, and learned parameters
- **Context Retrieval**: Provides relevant background for agent reasoning
- **Learning Storage**: Maintains coefficient updates and pattern libraries

### **Orchestrator** (`ocean_coordinator.py`)
**Purpose**: Multi-agent coordination and workflow management
- **Agent Scheduling**: Coordinates timing between Scout, Navigator, and Critic
- **Data Flow Management**: Routes information between agents and external APIs
- **Resource Allocation**: Manages API rate limits and computational resources
- **Error Handling**: Provides graceful degradation when components fail  

## 3. Tool Integration

List each external tool or API and how you call it:
- **Search API**: function `search(query)`  
- **Calculator**: LLM function calling  

## 4. Observability & Testing

Explain your logging and how judges can trace decisions:
- Logs saved in `logs/` directory  
- `TEST.sh` exercises main path  

## 5. Known Limitations

Be honest about edge cases or performance bottlenecks:
- Long-running API calls  
- Handling of ambiguous user inputs  

