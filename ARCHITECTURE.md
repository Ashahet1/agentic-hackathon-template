# Ocean Plastic Sentinel - System Architecture

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OCEAN PLASTIC SENTINEL                       │
│                          COMMAND CENTER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │     PLANNER     │  │    EXECUTOR     │  │     MEMORY      │      │
│  │  Task Planning  │  │  Agent Control  │  │   Context &     │      │
│  │   & Strategy    │  │  & Coordination │  │   Learning      │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────────────────────┬───────────────────────┬───────────────────┘
                          │                       │
            ┌─────────────┴─┐                 ┌───▼────┐
            │ USER REQUEST  │                 │ MEMORY │
            │ (Region/Task) │                 │ SYSTEM │
            └─────────────┬─┘                 └────────┘
                          ▼
      ┌───────────────────────────────────────────────┐
      │              AGENT WORKFLOW                   │
      │  ┌─────────┐    ┌─────────┐    ┌─────────┐   │
      │  │   1     │───▶│    2    │───▶│    3    │   │
      │  │ SCOUT   │    │NAVIGATR │    │ CRITIC  │   │
      │  │ DETECT  │    │PREDICT  │    │VALIDATE │   │
      │  └─────────┘    └─────────┘    └─────────┘   │
      │       │              │              │        │
      └───────┼──────────────┼──────────────┼────────┘
              ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ GEMINI API  │ │ GEMINI API  │ │ GEMINI API  │
    │ + Sentinel-2│ │ + NOAA Data │ │ + Historical│
    │ Satellite   │ │ Ocean Model │ │ Validation  │
    │ Analysis    │ │ Physics     │ │ Learning    │
    └─────────────┘ └─────────────┘ └─────────────┘
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   DEBRIS    │ │   DRIFT     │ │  UPDATED    │
    │ COORDINATES │ │ PREDICTIONS │ │ PARAMETERS  │
    │ + Confidence│ │ + Routes    │ │ + Accuracy  │
    └─────────────┘ └─────────────┘ └─────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ FINAL OUTPUT:   │
                    │ • Cleanup Route │
                    │ • Time Windows  │
                    │ • Cost Analysis │
                    │ • GeoJSON File  │
                    └─────────────────┘
```

## Agentic Workflow Detail

```
USER: "Find plastic debris in Pacific Gyre"
  │
  ▼
PLANNER: Breaks down into sub-tasks:
  ├─ Task 1: Scan satellite imagery for debris
  ├─ Task 2: Predict debris movement  
  └─ Task 3: Validate and optimize
  │
  ▼
EXECUTOR: Coordinates agents in sequence:
  │
  ├─ SCOUT AGENT ─────────────────────────────┐
  │  ├─ Gets Sentinel-2 imagery               │
  │  ├─ Converts spectral data to text        │
  │  ├─ Asks Gemini: "Analyze this data       │
  │  │   for plastic signatures"              │
  │  └─ Returns: [(lat, lon, confidence)]     │
  │                                           │
  ├─ NAVIGATOR AGENT ──────────────────────┐  │
  │  ├─ Takes debris coordinates from Scout│  │
  │  ├─ Gets NOAA current/wind data       │  │
  │  ├─ Asks Gemini: "Calculate optimal   │  │
  │  │   drift prediction and intercept"  │  │
  │  └─ Returns: [routes, time_windows]   │  │
  │                                       │  │
  ├─ CRITIC AGENT ─────────────────────┐  │  │
  │  ├─ Compares old predictions vs     │  │  │
  │  │   new satellite observations    │  │  │
  │  ├─ Asks Gemini: "How accurate     │  │  │
  │  │   was our prediction?"         │  │  │
  │  └─ Updates: [coefficients, model] │  │  │
  │                                    │  │  │
  └─ MEMORY stores all results ─────────┘  │  │
     for future context                    │  │
                                          │  │
FINAL RESPONSE: ◄─────────────────────────┘  │
  • "Found 3 debris patches"                 │
  • "Optimal intercept: 34.5°N, 140.2°W"    │  
  • "Best window: Tomorrow 2-6 PM"          │
  • "Fuel savings: 60% vs random search"    │
  • Downloads: cleanup_route.geojson        │
                                            │
LEARNING LOOP: ◄──────────────────────────┘
  Critic validates results and improves
  system accuracy over time
```

## Core Components

### 1. **Multi-Agent Orchestration System**

**Agent Coordinator (Main Controller)**
- Manages the workflow between three specialized agents
- Handles inter-agent communication and data flow
- Coordinates timing for satellite data pulls and predictions
- Powered by **Gemini 1.5 Pro** for decision-making

### 2. **Scout Agent - Plastic Detection**

**Responsibilities:**
- Processes multispectral satellite imagery from Sentinel-2
- Converts NIR/SWIR bands into visible RGB pseudo-images
- Identifies plastic debris "fingerprints" using spectral analysis
- Flags potential cleanup targets with confidence scores

**Technical Implementation:**
- **Input**: Sentinel-2 L2A imagery (10m resolution)
- **Processing**: Gemini vision model analyzes converted spectral data
- **Output**: Geographic coordinates of detected plastic patches
- **Tools**: Google Earth Engine API, Gemini Multimodal API

### 3. **Navigator Agent - Drift Prediction**

**Responsibilities:**
- Calculates debris drift trajectories using oceanographic data
- Applies physics-based modeling: `V_drift = α*V_current + β*V_wind`
- Generates optimal intercept routes for cleanup vessels
- Provides time-sensitive cleanup windows

**Technical Implementation:**
- **Input**: NOAA ocean current and wind data, initial debris coordinates
- **Processing**: Physics modeling combined with Gemini reasoning
- **Output**: Predicted drift paths and optimal intercept coordinates
- **Tools**: NOAA API, HYCOM ocean model data

### 4. **Critic Agent - Self-Improvement Loop**

**Responsibilities:**
- Validates predictions against new satellite observations (48h later)
- Calculates prediction accuracy and error margins
- Updates drift coefficients (α, β) based on real-world outcomes
- Maintains system learning and improvement over time

**Technical Implementation:**
- **Input**: Original predictions + new satellite imagery
- **Processing**: Gemini-powered error analysis and coefficient optimization
- **Output**: Updated model parameters and accuracy metrics
- **Storage**: BigQuery for historical pattern analysis

## Data Flow Architecture

### Primary Data Pipeline
```
Satellite Data → Scout Agent → Debris Coordinates
                                      ↓
Ocean Data → Navigator Agent → Drift Predictions
                                      ↓
New Satellite Data → Critic Agent → Model Updates
                                      ↓
                              Updated System Parameters
```

### Memory & Learning System
- **Short-term Memory**: Current session debris tracking (Redis cache)
- **Long-term Memory**: Historical drift patterns and accuracy metrics (BigQuery)
- **Learning Mechanism**: Continuous parameter tuning based on validation results

## Technology Stack

### Core AI & Processing
- **Gemini 1.5 Pro**: Primary reasoning engine for all three agents
- **Google Earth Engine**: Satellite imagery access and processing
- **Vertex AI**: Agent orchestration and deployment

### Data Sources
- **Sentinel-2 Constellation**: Multispectral satellite imagery (ESA)
- **NOAA Ocean Service**: Real-time ocean currents and wind data
- **HYCOM Model**: High-resolution ocean circulation data

### Backend & Storage
- **Python**: Primary development language
- **BigQuery**: Historical data warehouse and analytics
- **Cloud Functions**: Serverless agent execution
- **Firestore**: Real-time coordination between agents

### User Interface
- **Streamlit Dashboard**: Real-time monitoring and visualization
- **GeoJSON Export**: Direct integration with ship navigation systems
- **REST API**: Third-party integration capabilities

## Scalability & Performance

### Geographic Coverage
- **Global Monitoring**: Sentinel-2 provides worldwide coverage every 5 days
- **Priority Regions**: Great Pacific Garbage Patch, Mediterranean, Caribbean
- **Resolution**: 10m pixel resolution for precise debris detection

### Processing Efficiency
- **Parallel Processing**: Independent agent operation for multiple regions
- **Smart Filtering**: Pre-filtering reduces Gemini API calls by 70%
- **Caching Strategy**: 24-hour cache for ocean data to minimize API costs

## Observability & Monitoring

### Agent Performance Tracking
- **Detection Accuracy**: Scout agent plastic identification rates
- **Drift Precision**: Navigator prediction error margins
- **Learning Progress**: Critic agent improvement metrics over time

### System Monitoring
- **API Usage**: Gemini, Earth Engine, and NOAA API call tracking
- **Response Times**: End-to-end processing latency monitoring
- **Error Handling**: Comprehensive retry logic and failure recovery

### Business Metrics
- **Cost Efficiency**: Fuel savings vs. traditional search methods
- **Cleanup Effectiveness**: Tons of plastic intercepted using predictions
- **Environmental Impact**: Ocean area coverage and debris reduction

## Security & Compliance

### Data Privacy
- **No Personal Data**: System processes only environmental/satellite data
- **API Security**: Encrypted connections and secure key management
- **Access Control**: Role-based permissions for different user types

### Reliability
- **Fault Tolerance**: Multiple agent backup systems
- **Data Validation**: Cross-verification between multiple data sources
- **Graceful Degradation**: System continues operation with limited data  

