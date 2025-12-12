# Ocean Plastic Sentinel - Technical Explanation

## 1. Agent Workflow

### Step-by-Step Processing Flow

**Input**: Ocean region coordinates or "global scan" request

1. **Receive Mission Parameters**
   - Target ocean region (lat/lon bounds) or global monitoring request
   - Priority level (emergency cleanup vs. routine monitoring)
   - Available cleanup vessel information (location, capacity, speed)

2. **Planner: Detection Planning Phase**
   - Retrieve latest Sentinel-2 satellite imagery for target region
   - Convert multispectral bands (NIR/SWIR) to Gemini-readable RGB pseudo-images
   - Use Gemini vision model to identify plastic debris signatures
   - Generate confidence-scored detection coordinates

3. **Executor: Drift Prediction Phase**
   - Fetch real-time NOAA ocean current and wind data
   - Apply physics-based drift modeling: $\vec{V}_{drift} = \alpha \cdot \vec{V}_{current} + \beta \cdot \vec{V}_{wind}$
   - Calculate 48-72 hour trajectory predictions for detected debris
   - Optimize intercept routes for available cleanup vessels

4. **Memory: Validation & Learning Phase**
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

### **Planner** (`planner.py`)
**Purpose**: Task decomposition and strategy planning
- **Task Analysis**: Breaks down user requests into executable sub-tasks
- **Detection Planning**: Plans satellite imagery analysis strategies  
- **Prediction Planning**: Designs drift calculation approaches
- **Validation Planning**: Schedules accuracy checking and learning cycles
- **Resource Allocation**: Manages API quotas and computational resources

### **Executor** (`executor.py`)  
**Purpose**: Task execution and tool coordination
- **Detection Execution**: Processes satellite imagery and calls Gemini for analysis
- **Prediction Execution**: Integrates ocean data with physics modeling
- **Validation Execution**: Compares predictions against new observations
- **Tool Management**: Coordinates calls to external APIs (Earth Engine, NOAA)
- **Result Synthesis**: Combines outputs from all tasks into final response

### **Memory System** (`memory.py`)
**Purpose**: Context preservation and learning persistence
- **Short-term Memory**: Active session tracking of detections and predictions
- **Long-term Memory**: Historical patterns, accuracy data, and learned parameters
- **Context Retrieval**: Provides relevant background for task execution
- **Learning Storage**: Maintains coefficient updates and pattern libraries

### **Agent Coordinator** (`agent.py`)
**Purpose**: Overall system orchestration and user interface
- **Request Processing**: Handles user inputs and mission parameters
- **Workflow Management**: Coordinates planner → executor → memory cycles
- **Response Generation**: Synthesizes final outputs and recommendations
- **Error Handling**: Provides graceful degradation when components fail
- **Performance Monitoring**: Tracks system metrics and improvement trends  

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

