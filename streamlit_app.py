# streamlit_app.py

import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    mission_types = {
        "Standard Detection": MissionType.STANDARD_DETECTION,
        "Urgent Cleanup": MissionType.URGENT_CLEANUP,
        "Regional Survey": MissionType.REGIONAL_SURVEY,
        "Validation Mission": MissionType.VALIDATION
    }
    
    return MissionRequest(
        mission_type=mission_types[mission_type],
        target_location=location,
        priority=1 if mission_type == "Urgent Cleanup" else 3,
        metadata={
            "description": f"{mission_type} mission",
            "created_at": datetime.now().isoformat(),
            "test_mode": True
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
        st.subheader("2️⃣ Mission Configuration")
        
        mission_type = st.selectbox(
            "🎯 Mission Type",
            ["Standard Detection", "Urgent Cleanup", "Regional Survey", "Validation Mission"],
            help="Select the type of ocean cleanup mission"
        )
        
        st.caption("📍 Target Location Coordinates")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=25.0, min_value=-90.0, max_value=90.0, step=0.1)
        with col2:
            longitude = st.number_input("Longitude", value=-80.0, min_value=-180.0, max_value=180.0, step=0.1)
        
        # Predefined locations
        st.caption("🗺️ Quick Location Presets:")
        location_presets = {
            "🌊 Great Pacific Garbage Patch": (35.0, -145.0),
            "🏝️ Caribbean Sea": (18.0, -75.0),
            "🌍 Mediterranean Sea": (36.0, 15.0),
            "🌏 Bay of Bengal": (15.0, 90.0)
        }
        
        selected_preset = st.selectbox("Load Preset", ["🎯 Custom Location"] + list(location_presets.keys()))
        if selected_preset != "🎯 Custom Location":
            latitude, longitude = location_presets[selected_preset]
            st.rerun()
        
        st.divider()
        
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
                
                if result.status == MissionStatus.COMPLETED:
                    add_log(f"✅ Mission completed - Debris detected and route optimized", "SUCCESS")
                    st.success("🎉 Mission Successful!")
                    st.balloons()
                else:
                    add_log(f"⚠️ Mission status: {result.status.value}", "WARNING")
                    st.warning(f"⚠️ Mission status: {result.status.value}")
        
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
            MissionStatus.COMPLETED: "✅",
            MissionStatus.FAILED: "❌",
            MissionStatus.IN_PROGRESS: "⏳"
        }
        st.metric("📊 Status", f"{status_emoji.get(mission.status, '⚪')} {mission.status.value.upper()}")
    with col3:
        st.metric("⏱️ Duration", f"{mission.execution_time_seconds:.2f}s" if mission.execution_time_seconds else "N/A")
    with col4:
        confidence = mission.results.get('confidence', 0) if mission.results else 0
        st.metric("🎯 Confidence", f"{confidence:.1%}")
    
    # Mission Details
    with st.expander("📝 Detailed Mission Report", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Request Parameters")
            st.json({
                "type": mission.mission_type.value,
                "location": mission.target_location,
                "priority": mission.priority,
                "timestamp": mission.timestamp
            })
        
        with col2:
            st.subheader("📤 Mission Results")
            if mission.results:
                st.json(mission.results)
            else:
                st.info("Awaiting results...")

def render_task_execution():
    """Render task execution details."""
    st.header("⚙️ Agent Task Execution")
    
    if st.session_state.current_mission is None or not st.session_state.current_mission.results:
        st.info("💡 No task execution data available. Complete a mission to view agent workflows.")
        return
    
    mission = st.session_state.current_mission
    
    # Create task timeline
    if 'tasks_completed' in mission.results:
        tasks = mission.results['tasks_completed']
        
        # Task summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Tasks Completed", len(tasks))
        with col2:
            total_time = sum(t.get('execution_time', 0) for t in tasks)
            st.metric("⏱️ Total Time", f"{total_time:.2f}s")
        with col3:
            successful = sum(1 for t in tasks if t.get('status') == 'completed')
            st.metric("📈 Success Rate", f"{successful/len(tasks)*100:.0f}%")
        
        # Task details table
        st.subheader("📊 Task Breakdown by AI Agent")
        task_df = pd.DataFrame([{
            '🤖 Agent': t.get('type', 'Unknown').replace('_', ' ').title(),
            '✅ Status': t.get('status', 'Unknown').upper(),
            '⏱️ Duration (s)': f"{t.get('execution_time', 0):.2f}",
            '⚡ Priority': t.get('priority', 'N/A')
        } for t in tasks])
        
        st.dataframe(task_df, use_container_width=True)

def render_visualization():
    """Render data visualizations."""
    st.header("📊 Mission Analytics & Visualizations")
    
    if not st.session_state.mission_history:
        st.info("💡 No mission data yet. Execute missions to generate analytics and visualizations.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Mission Timeline", "⚡ Performance Metrics", "🗺️ Detection Map"])
    
    with tab1:
        # Mission timeline with ocean theme
        timeline_data = [{
            'Mission': m.mission_id[:8],
            'Type': m.mission_type.value.replace('_', ' ').title(),
            'Status': m.status.value.upper(),
            'Duration': m.execution_time_seconds or 0,
            'Time': m.timestamp
        } for m in st.session_state.mission_history]
        
        df = pd.DataFrame(timeline_data)
        fig = px.bar(df, x='Mission', y='Duration', color='Status',
                     title='Mission Execution Timeline',
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
            avg_duration = sum(m.execution_time_seconds or 0 for m in st.session_state.mission_history) / len(st.session_state.mission_history)
            success_rate = sum(1 for m in st.session_state.mission_history if m.status == MissionStatus.COMPLETED) / len(st.session_state.mission_history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Avg Duration", f"{avg_duration:.2f}s")
            with col2:
                st.metric("✅ Success Rate", f"{success_rate*100:.0f}%")
            with col3:
                st.metric("📊 Total Missions", len(st.session_state.mission_history))
    
    with tab3:
        # Detection map with ocean styling
        if st.session_state.current_mission and st.session_state.current_mission.target_location:
            loc = st.session_state.current_mission.target_location
            
            fig = go.Figure(go.Scattermapbox(
                lat=[loc['lat']],
                lon=[loc['lon']],
                mode='markers+text',
                marker=go.scattermapbox.Marker(
                    size=20,
                    color='#ff6b6b',
                    opacity=0.8
                ),
                text=[f"🎯 Mission Zone"],
                textposition="top center",
                textfont=dict(size=14, color='white')
            ))
            
            fig.update_layout(
                mapbox_style="carto-darkmatter",  # Dark ocean-like map
                mapbox=dict(
                    center=dict(lat=loc['lat'], lon=loc['lon']),
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