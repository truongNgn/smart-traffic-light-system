import streamlit as st
import pandas as pd
import plotly.express as px
import time
import state

# Must be the first Streamlit command
st.set_page_config(page_title="AI Traffic Controller", page_icon="🚦", layout="wide")

# Map Enum indices to string names since Pydantic outputs integers for our Enum
DIR_MAP = {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH", None: None}

# Initialize websocket thread (runs only once)
state.start_background_thread()

st.title("🚦 AI Traffic Controller - Live Telemetry")

# ================================
# 1. TRAFFIC LIGHTS UI
# ================================
st.header("Intersection Status")
phase = state.latest_phase_state
active_dir_int = phase.get("active_direction")
active_dir = DIR_MAP.get(active_dir_int)

cols = st.columns(4)
directions = ["NORTH", "SOUTH", "EAST", "WEST"]

for col, d in zip(cols, directions):
    with col:
        st.subheader(f"Route: {d}")
        
        # Determine color
        if phase.get("is_all_red"):
            color_hex = "#FF3B30" # Red
            label = "RED (SAFETY BUFFER)"
        elif active_dir == d:
            if phase.get("is_yellow"):
                color_hex = "#FFCC00" # Yellow
                label = "YELLOW"
            else:
                color_hex = "#34C759" # Green
                label = "GREEN"
        else:
            color_hex = "#FF3B30" # Red
            label = "RED"
            
        # Draw a beautiful circle indicator
        html = f"""
        <div style="
            width: 100px; 
            height: 100px; 
            border-radius: 50%; 
            background-color: {color_hex};
            box-shadow: 0 0 20px {color_hex};
            margin: 0 auto;
            border: 4px solid #333;
        "></div>
        <p style="text-align: center; margin-top: 10px; font-weight: bold; color: {color_hex}">{label}</p>
        """
        st.markdown(html, unsafe_allow_html=True)

st.divider()

# ================================
# 2. REAL-TIME QUEUE CHART
# ================================
st.header("Live Lane Occupancy")

# Combine all deques into a single DataFrame for Plotly
df_list = []
for d in directions:
    q = state.vehicle_counts[d]
    for item in q:
        df_list.append({
            "Direction": d,
            "Time": pd.to_datetime(item["time"], unit='s'),
            "Count": item["count"]
        })

if df_list:
    df = pd.DataFrame(df_list)
    # Sort by time
    df = df.sort_values("Time")
    
    fig = px.line(
        df, 
        x="Time", 
        y="Count", 
        color="Direction",
        title="Vehicles in ROI per Direction",
        markers=True,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    )
    fig.update_layout(yaxis_title="Vehicle Count", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for vehicle count data from vision producer...")

# ================================
# Auto-Refresh Loop
# ================================
# Rerun the script every 0.5 seconds to pull new data from state
time.sleep(0.5)
st.rerun()
