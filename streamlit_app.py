# streamlit_app.py

import streamlit as st # type: ignore
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import uuid

from src.core.agent import OceanPlasticSentinel, MissionRequest, MissionStatus
from src.core.planner import MissionType
from src.config import config

# Page configuration
st.set_page_config(
    page_title="Ocean Plastic Sentinel - Testing Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Ocean-themed CSS
st.markdown("""
    <style>
    /* Main theme colors - Ocean inspired */
    :root {
        --ocean-dark: #0a2463;
        --ocean-medium: #247ba0;
        --ocean-light: #1e90ff;
        --ocean-accent: #06ffa5;
        --plastic-warning: #ff6b6b;
        --success-green: #51cf66;
    }
    
    /* Dark ocean background theme */
    .stApp {
        background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
        color: #0c4a6e;
    }
    
    /* Headers */
    .main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(135deg, #0891b2 0%, #0284c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
    
    .subtitle {
        text-align: center;
        color: #0369a1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Mission card - Ocean wave effect */
    .mission-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #247ba0 0%, #0a2463 100%);
        border: 2px solid #06ffa5;
        box-shadow: 0 8px 32px rgba(6, 255, 165, 0.2);
        color: white;
        margin-bottom: 1.5rem;
    }
    
    /* Metric cards - Glass morphism ocean style */
    .stMetric {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(6, 255, 165, 0.5);
    border-radius: 10px;
    padding: 1rem;
}

.stMetric label {
    color: #0c4a6e !important;
    font-weight: 600;
}
    
    .stMetric .metric-value {
        color: #ffffff !important;
    }
    
    /* Buttons - Ocean themed */
    .stButton > button {
        background: linear-gradient(135deg, #247ba0 0%, #1e90ff 100%);
        color: white;
        border: 2px solid #06ffa5;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(6, 255, 165, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e90ff 0%, #06ffa5 100%);
        box-shadow: 0 6px 20px rgba(6, 255, 165, 0.5);
        transform: translateY(-2px);
    }
    
    /* Sidebar - Deep ocean theme */
    [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
    border-right: 2px solid #0891b2;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0c4a6e !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(10, 36, 99, 0.5);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(36, 123, 160, 0.3);
        border-radius: 8px;
        color: #ffffff;
        border: 1px solid rgba(6, 255, 165, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #247ba0 0%, #1e90ff 100%);
        border: 2px solid #06ffa5;
    }
    
    /* Dataframes and tables */
    .dataframe {
        background-color: rgba(10, 36, 99, 0.6) !important;
        color: white !important;
        border: 1px solid #06ffa5;
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(36, 123, 160, 0.3);
        border: 1px solid rgba(6, 255, 165, 0.3);
        border-radius: 10px;
        color: #ffffff !important;
    }
    
    /* Status indicators */
    .status-online {
        color: #51cf66;
        font-weight: bold;
        text-shadow: 0 0 10px #51cf66;
    }
    
    .status-offline {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    /* Divider */
    hr {
        border-color: rgba(6, 255, 165, 0.3);
    }
    
    /* Selectbox and inputs */
    .stSelectbox, .stNumberInput {
        background-color: rgba(36, 123, 160, 0.2);
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: rgba(30, 144, 255, 0.2);
        border-left: 4px solid #1e90ff;
    }
    
    .stSuccess {
        background-color: rgba(81, 207, 102, 0.2);
        border-left: 4px solid #51cf66;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.2);
        border-left: 4px solid #ffc107;
    }
    
    .stError {
        background-color: rgba(255, 107, 107, 0.2);
        border-left: 4px solid #ff6b6b;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #06ffa5 !important;
    }
    
    /* JSON display */
    .stJson {
        background-color: rgba(10, 36, 99, 0.6);
        border: 1px solid rgba(6, 255, 165, 0.3);
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'sentinel' not in st.session_state:
    st.session_state.sentinel = None
    st.session_state.mission_history = []
    st.session_state.current_mission = None
    st.session_state.logs = []

def add_log(message: str, level: str = "INFO"):
    """Add a log entry with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "timestamp": timestamp,
        "level": level,
        "message": message
    })

async def initialize_system():
    """Initialize the Ocean Plastic Sentinel system."""
    try:
        sentinel = OceanPlasticSentinel()
        success = await sentinel.initialize()
        if success:
            st.session_state.sentinel = sentinel
            add_log("✅ Ocean Plastic Sentinel initialized successfully", "SUCCESS")
            return True
        else:
            add_log("❌ Failed to initialize system", "ERROR")
            return False
    except Exception as e:
        add_log(f"❌ Initialization error: {str(e)}", "ERROR")
        return False

def create_mock_mission_request(mission_type: str, location: Dict[str, float]) -> MissionRequest:
    """Create a mission request with mock data."""
    import uuid
    
    mission_types = {
        "Standard Detection": MissionType.STANDARD_DETECTION,
        "Urgent Cleanup": MissionType.URGENT_CLEANUP,
        "Regional Survey": MissionType.REGIONAL_SURVEY,
        "Validation Mission": MissionType.VALIDATION
    }
    
    # Convert point location to region (add ±1 degree buffer)
    lat = location['lat']
    lon = location['lon']
    
    target_region = {
        'north': lat + 1.0,
        'south': lat - 1.0,
        'east': lon + 1.0,
        'west': lon - 1.0
    }
    
    priority_map = {
        "Urgent Cleanup": "urgent",
        "Standard Detection": "standard",
        "Regional Survey": "standard",
        "Validation Mission": "background"
    }
    
    return MissionRequest(
        mission_type=mission_types[mission_type],
        mission_id=str(uuid.uuid4()),
        target_region=target_region,
        priority_level=priority_map[mission_type],
        vessel_constraints={"max_speed_knots": 15, "capacity_tons": 50},
        time_window_hours=72,
        requested_by="streamlit_test",
        metadata={
            "description": f"{mission_type} mission",
            "created_at": datetime.now().isoformat(),
            "test_mode": True,
            "center_point": location
        }
    )

def render_header():
    """Render the app header."""
    st.markdown('<div class="main-header">🌊 OCEAN PLASTIC SENTINEL 🌊</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Marine Debris Detection & Cleanup Optimization System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">⚡ Phase 1 Testing Environment - Multi-Agent Workflow ⚡</div>', unsafe_allow_html=True)
    st.divider()

def render_sidebar():
    """Render the sidebar with system controls."""
    with st.sidebar:
        st.header("🎮 MISSION CONTROL")
        
        # Initialization
        st.subheader("1️⃣ System Initialization")
        if st.button("🚀 Initialize Sentinel System", use_container_width=True):
            with st.spinner("Initializing AI agents and data connections..."):
                success = asyncio.run(initialize_system())
                if success:
                    st.success("✅ All systems operational!")
                else:
                    st.error("❌ Initialization failed - check logs")
        
        st.divider()
        
        # Mission Configuration
        st.subheader("2️⃣ Configure Mission")

        mission_type = st.selectbox(
            "🎯 Mission Type",
            ["Standard Detection", "Urgent Cleanup", "Regional Survey", "Validation Mission"],
            help="Select the type of ocean cleanup mission"
        )

        # Predefined locations
        location_presets = {
            "🎯 Custom Location": (25.0, -80.0),
            "🌊 Great Pacific Garbage Patch": (35.0, -145.0),
            "🏝️ Caribbean Sea": (18.0, -75.0),
            "🌍 Mediterranean Sea": (36.0, 15.0),
            "🌏 Bay of Bengal": (15.0, 90.0)
        }

        st.caption("🗺️ Select Location:")
        selected_preset = st.selectbox(
            "Quick Locations",
            list(location_presets.keys()),
            index=0
        )

        # Get coordinates from preset
        latitude, longitude = location_presets[selected_preset]

        st.caption("📍 Or Enter Custom Coordinates:")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=latitude, min_value=-90.0, max_value=90.0, step=0.1)
        with col2:
            longitude = st.number_input("Longitude", value=longitude, min_value=-180.0, max_value=180.0, step=0.1)
                
        # Mission Execution
        st.subheader("3️⃣ Mission Execution")
        
        if st.button("🎯 LAUNCH MISSION", use_container_width=True, type="primary", 
                    disabled=st.session_state.sentinel is None):
            location = {"lat": latitude, "lon": longitude}
            request = create_mock_mission_request(mission_type, location)
            
            add_log(f"🚀 Launching {mission_type} at ({latitude:.2f}, {longitude:.2f})", "INFO")
            
            with st.spinner("🤖 AI agents analyzing ocean data..."):
                result = asyncio.run(st.session_state.sentinel.execute_mission(request))
                st.session_state.current_mission = result
                st.session_state.mission_history.append(result)
                
                # Check status - it might be a string or enum
                status_str = result.status if isinstance(result.status, str) else result.status.value
                
                if status_str == 'completed' or status_str == MissionStatus.COMPLETED.value:
                    add_log(f"✅ Mission completed - Debris detected and route optimized", "SUCCESS")
                    st.success("🎉 Mission Successful!")
                    st.balloons()
                else:
                    add_log(f"⚠️ Mission status: {status_str}", "WARNING")
                    st.warning(f"⚠️ Mission status: {status_str}")
        
        st.divider()
        
        # System Info
        st.subheader("ℹ️ System Status")
        if st.session_state.sentinel:
            st.markdown('<p class="status-online">🟢 ONLINE</p>', unsafe_allow_html=True)
            st.metric("📊 Missions Executed", len(st.session_state.mission_history))
            st.metric("🤖 Active Agents", "3/3")
        else:
            st.markdown('<p class="status-offline">🔴 OFFLINE</p>', unsafe_allow_html=True)
            st.caption("Initialize system to begin operations")

def render_mission_overview():
    """Render the current mission overview."""
    st.header("📋 Current Mission Overview")
    
    if st.session_state.current_mission is None:
        st.info("💡 No active mission. Configure and launch a mission from Mission Control to begin.")
        return
    
    mission = st.session_state.current_mission
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🆔 Mission ID", mission.mission_id[:8] + "...")
    with col2:
        status_emoji = {
            'completed': "✅",
            'failed': "❌",
            'in_progress': "⏳"
        }
        # Handle both string and enum status
        status_str = mission.status if isinstance(mission.status, str) else mission.status.value
        st.metric("📊 Status", f"{status_emoji.get(status_str, '⚪')} {status_str.upper()}")
    with col3:
        st.metric("⏱️ Duration", f"{mission.execution_time_seconds:.2f}s" if mission.execution_time_seconds else "N/A")
    with col4:
        # Get confidence from results or confidence_metrics
        if hasattr(mission, 'confidence_metrics') and mission.confidence_metrics:
            confidence = mission.confidence_metrics.get('overall_confidence', 0)
        elif hasattr(mission, 'results') and mission.results:
            confidence = mission.results.get('confidence', 0)
        else:
            confidence = 0
        st.metric("🎯 Confidence", f"{confidence:.1%}")
    
    # Mission Details
    with st.expander("📝 Detailed Mission Report", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Mission Data")
            # Build data dict from available attributes
            mission_data = {
                "mission_id": mission.mission_id,
                "status": status_str,
                "execution_time": f"{mission.execution_time_seconds:.2f}s" if mission.execution_time_seconds else "N/A"
            }
            
            # Add optional attributes if they exist
            if hasattr(mission, 'detections'):
                mission_data['detections_count'] = len(mission.detections) if mission.detections else 0
            if hasattr(mission, 'predictions'):
                mission_data['predictions_count'] = len(mission.predictions) if mission.predictions else 0
            if hasattr(mission, 'recommended_routes'):
                mission_data['routes_count'] = len(mission.recommended_routes) if mission.recommended_routes else 0
                
            st.json(mission_data)
        
        with col2:
            st.subheader("📤 Mission Results")
            if hasattr(mission, 'confidence_metrics') and mission.confidence_metrics:
                st.json(mission.confidence_metrics)
            elif hasattr(mission, 'results') and mission.results:
                st.json(mission.results)
            else:
                st.info("No detailed results available")

def render_task_execution():
    """Render task execution details."""
    st.header("⚙️ Agent Task Execution")
    
    if st.session_state.current_mission is None:
        st.info("💡 No task execution data available. Complete a mission to view agent workflows.")
        return
    
    mission = st.session_state.current_mission
    
    # Check if mission has confidence_metrics
    if hasattr(mission, 'confidence_metrics') and mission.confidence_metrics:
        # Task summary from confidence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 Agents Used", "3")  # Planning, Execution, Memory
        with col2:
            exec_time = mission.execution_time_seconds or 0
            st.metric("⏱️ Total Time", f"{exec_time:.2f}s")
        with col3:
            status = mission.status if isinstance(mission.status, str) else str(mission.status)
            success = "✅" if status == 'completed' else "❌"
            st.metric("📈 Status", f"{success} {status.upper()}")
        
        # Show agent breakdown
        st.subheader("📊 Multi-Agent Workflow")
        
        agent_data = [
            {"🤖 Agent": "Task Planner", "✅ Status": "COMPLETED", "⏱️ Duration": f"{exec_time * 0.2:.2f}s", "📋 Task": "Mission decomposition"},
            {"🤖 Agent": "Task Executor", "✅ Status": status.upper(), "⏱️ Duration": f"{exec_time * 0.7:.2f}s", "📋 Task": "Data collection & analysis"},
            {"🤖 Agent": "Memory System", "✅ Status": "COMPLETED", "⏱️ Duration": f"{exec_time * 0.1:.2f}s", "📋 Task": "Result storage"},
        ]
        
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True)
        
        # Show confidence metrics if available
        if mission.confidence_metrics:
            with st.expander("🎯 Confidence Metrics Details", expanded=False):
                st.json(mission.confidence_metrics)
    else:
        st.info("💡 No detailed task execution data available for this mission.")
    
    # Show detections summary
    if hasattr(mission, 'detections') and mission.detections:
        st.subheader("🎯 Detection Results")
        st.metric("Total Detections", len(mission.detections))
        
        detection_preview = pd.DataFrame(mission.detections[:5] if len(mission.detections) > 5 else mission.detections)
        st.dataframe(detection_preview, use_container_width=True)
    
    # Show predictions summary
    if hasattr(mission, 'predictions') and mission.predictions:
        st.subheader("📈 Drift Predictions")
        st.metric("Total Predictions", len(mission.predictions))
    
    # Show routes summary
    if hasattr(mission, 'recommended_routes') and mission.recommended_routes:
        st.subheader("🚢 Recommended Routes")
        st.metric("Total Routes", len(mission.recommended_routes))

def render_visualization():
    """Render data visualizations."""
    st.header("📊 Mission Analytics & Visualizations")
    
    if not st.session_state.mission_history:
        st.info("💡 No mission data yet. Execute missions to generate analytics and visualizations.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Mission Timeline", "⚡ Performance Metrics", "🗺️ Detection Map"])
    
    with tab1:
        # Mission timeline with ocean theme
        timeline_data = []
        for m in st.session_state.mission_history:
            status_str = m.status if isinstance(m.status, str) else m.status
            timeline_data.append({
                'Mission': m.mission_id[:8],
                'Status': status_str.upper() if isinstance(status_str, str) else str(status_str).upper(),
                'Duration': m.execution_time_seconds or 0,
                'Detections': len(m.detections) if m.detections else 0
            })
        
        df = pd.DataFrame(timeline_data)
        fig = px.bar(df, x='Mission', y='Duration', color='Status',
                     title='Mission Execution Timeline',
                     hover_data=['Detections'],
                     color_discrete_map={
                         'COMPLETED': '#51cf66',
                         'FAILED': '#ff6b6b',
                         'IN_PROGRESS': '#ffc107'
                     })
        fig.update_layout(
            plot_bgcolor='rgba(10, 36, 99, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Performance metrics
        if len(st.session_state.mission_history) > 0:
            total_detections = sum(len(m.detections) if m.detections else 0 for m in st.session_state.mission_history)
            avg_duration = sum(m.execution_time_seconds or 0 for m in st.session_state.mission_history) / len(st.session_state.mission_history)
            completed = sum(1 for m in st.session_state.mission_history if (m.status if isinstance(m.status, str) else str(m.status)) == 'completed')
            success_rate = completed / len(st.session_state.mission_history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Avg Duration", f"{avg_duration:.2f}s")
            with col2:
                st.metric("✅ Success Rate", f"{success_rate*100:.0f}%")
            with col3:
                st.metric("🎯 Total Detections", total_detections)
    
    with tab3:
        # Detection map - show all detections from current mission
        if st.session_state.current_mission:
            mission = st.session_state.current_mission
            
            # If mission has detections, plot them
            if hasattr(mission, 'detections') and mission.detections:
                lats = [d.get('lat', 0) for d in mission.detections if 'lat' in d]
                lons = [d.get('lon', 0) for d in mission.detections if 'lon' in d]
                
                if lats and lons:
                    fig = go.Figure(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=15,
                            color='#ff6b6b',
                            opacity=0.8
                        ),
                        text=[f"🎯 Detection {i+1}" for i in range(len(lats))],
                    ))
                    
                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)
                else:
                    # No valid coordinates, show placeholder
                    center_lat, center_lon = 0, 0
                    fig = go.Figure()
            else:
                # No detections, show placeholder
                center_lat, center_lon = 0, 0
                fig = go.Figure()
                st.info("No detections available for mapping")
            
            fig.update_layout(
                mapbox_style="carto-darkmatter",
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=4
                ),
                height=500,
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_logs():
    """Render system logs."""
    st.header("📜 System Activity Logs")
    
    if not st.session_state.logs:
        st.info("💡 No system logs yet. Activity will be logged here as missions execute.")
        return
    
    # Display recent logs
    log_df = pd.DataFrame(st.session_state.logs[-50:])  # Last 50 logs
    
    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "level": st.column_config.TextColumn("⚡ Level", width="small"),
            "timestamp": st.column_config.TextColumn("🕐 Time", width="small"),
            "message": st.column_config.TextColumn("📝 Message", width="large")
        }
    )
    
    if st.button("🗑️ Clear All Logs"):
        st.session_state.logs = []
        st.rerun()

def main():
    """Main app function."""
    render_header()
    render_sidebar()
    
    # Main content area with ocean-themed tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Mission Overview", 
        "⚙️ Agent Execution", 
        "📊 Analytics", 
        "📜 System Logs"
    ])
    
    with tab1:
        render_mission_overview()
    
    with tab2:
        render_task_execution()
    
    with tab3:
        render_visualization()
    
    with tab4:
        render_logs()

if __name__ == "__main__":
    main()