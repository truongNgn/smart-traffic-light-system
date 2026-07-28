import time
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import state


st.set_page_config(page_title="AI Traffic Operations", page_icon="🚦", layout="wide")

DIR_MAP = {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH", None: None}
PHASE_MAP = {0: ["EAST", "WEST"], 1: ["NORTH", "SOUTH"], None: []}
PHASE_NAME = {0: "EAST_WEST", 1: "NORTH_SOUTH", None: "ALL_RED"}
DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]
PHASE_COLORS = {
    "GREEN": "#12b981",
    "YELLOW": "#f4c430",
    "RED": "#ef4444",
    "ALL_RED": "#991b1b",
}


def normalize_phase(value):
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if value == "EAST_WEST":
            return 0
        if value == "NORTH_SOUTH":
            return 1
    return value


def active_directions(phase):
    active_phase = normalize_phase(phase.get("active_phase"))
    dirs = [DIR_MAP.get(d, d) for d in phase.get("active_directions", [])]
    dirs = [d for d in dirs if d]
    if not dirs:
        dirs = PHASE_MAP.get(active_phase, [])
    if not dirs:
        active_dir = DIR_MAP.get(phase.get("active_direction"))
        dirs = [active_dir] if active_dir else []
    return dirs


def signal_for(direction, phase, active_dirs):
    if phase.get("is_all_red"):
        return "ALL_RED"
    if direction in active_dirs:
        return "YELLOW" if phase.get("is_yellow") else "GREEN"
    return "RED"


def count_for(direction):
    queue = state.vehicle_counts[direction]
    if not queue:
        return 0
    return int(queue[-1]["count"])


def queue_dataframe():
    rows = []
    for direction in DIRECTIONS:
        for item in state.vehicle_counts[direction]:
            rows.append(
                {
                    "Direction": direction,
                    "Time": pd.to_datetime(item["time"], unit="s"),
                    "Vehicle Count": item["count"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Direction", "Time", "Vehicle Count"])
    return pd.DataFrame(rows).sort_values("Time")


def reasoning_dataframe(latest):
    q_values = latest.get("q_values", {}) if latest else {}
    rows = []
    for phase, value in q_values.items():
        rows.append({"Phase": str(phase), "Q-value": float(value)})
    return pd.DataFrame(rows)


def reasoning_history_dataframe():
    rows = []
    for item in reversed(state.latest_reasoning):
        sim_time = item.get("sim_time_s")
        waiting = item.get("total_waiting_time_s")
        if sim_time is None or waiting is None:
            continue
        action = PHASE_NAME.get(normalize_phase(item.get("chosen_action")), item.get("chosen_action"))
        rows.append(
            {
                "SUMO Time": float(sim_time),
                "Network Waiting Time": float(waiting),
                "Chosen Phase": str(action).replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def build_intersection_html(phase, active_dirs):
    lane_counts = {direction: count_for(direction) for direction in DIRECTIONS}
    signal = {direction: signal_for(direction, phase, active_dirs) for direction in DIRECTIONS}

    def lamp(direction):
        status = signal[direction]
        color = PHASE_COLORS[status]
        label = "ALL RED" if status == "ALL_RED" else status
        return (
            f'<div class="signal signal-{direction.lower()}">'
            f'<span class="lamp" style="background:{color}; box-shadow:0 0 22px {color};"></span>'
            f'<strong>{escape(direction.title())}</strong>'
            f'<small>{label}</small>'
            "</div>"
        )

    def vehicles(direction):
        count = min(lane_counts[direction], 9)
        flow_class = "is-moving" if signal[direction] == "GREEN" else "is-caution" if signal[direction] == "YELLOW" else "is-stopped"
        dots = "".join(
            f'<span class="vehicle-dot" style="--i:{index};"></span>' for index in range(count)
        )
        if count == 0:
            dots = '<span class="vehicle-placeholder"></span>'
        return f'<div class="vehicles vehicles-{direction.lower()} {flow_class}">{dots}</div>'

    phase_label = PHASE_NAME.get(normalize_phase(phase.get("active_phase")), "UNKNOWN")
    if phase.get("is_all_red"):
        phase_label = "ALL_RED SAFETY BUFFER"
    elif phase.get("is_yellow"):
        phase_label = f"{phase_label} YELLOW CLEARANCE"

    return f"""
    <div class="intersection-shell">
      <div class="phase-ribbon">{escape(phase_label.replace("_", " "))}</div>
      <div class="intersection">
        <div class="road vertical"></div>
        <div class="road horizontal"></div>
        <div class="crossbox"></div>
        <div class="lane-mark v1"></div><div class="lane-mark v2"></div>
        <div class="lane-mark h1"></div><div class="lane-mark h2"></div>
        {lamp("NORTH")}
        {lamp("EAST")}
        {lamp("SOUTH")}
        {lamp("WEST")}
        {vehicles("NORTH")}
        {vehicles("EAST")}
        {vehicles("SOUTH")}
        {vehicles("WEST")}
        <div class="count-badge north-count">{lane_counts["NORTH"]}</div>
        <div class="count-badge east-count">{lane_counts["EAST"]}</div>
        <div class="count-badge south-count">{lane_counts["SOUTH"]}</div>
        <div class="count-badge west-count">{lane_counts["WEST"]}</div>
      </div>
    </div>
    """


def build_queue_chart(df):
    fig = go.Figure()
    colors = {
        "NORTH": "#2563eb",
        "EAST": "#16a34a",
        "SOUTH": "#dc2626",
        "WEST": "#7c3aed",
    }
    for direction in DIRECTIONS:
        part = df[df["Direction"] == direction]
        fig.add_trace(
            go.Scatter(
                x=part["Time"],
                y=part["Vehicle Count"],
                name=direction.title(),
                mode="lines+markers",
                line={"width": 3, "color": colors[direction]},
                marker={"size": 7},
            )
        )
    fig.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis_title="Vehicles",
        xaxis_title=None,
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#111827"},
    )
    fig.update_yaxes(gridcolor="#e5e7eb", rangemode="tozero")
    fig.update_xaxes(gridcolor="#f1f5f9")
    return fig


def build_waiting_chart(df):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["SUMO Time"],
                y=df["Network Waiting Time"],
                mode="lines+markers",
                name="Network waiting",
                line={"width": 3, "color": "#ea580c"},
                marker={"size": 7, "color": "#0f766e"},
                hovertemplate="SUMO %{x:.0f}s<br>Waiting %{y:.0f}s<extra></extra>",
            )
        )
    fig.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        yaxis_title="Waiting Time (s)",
        xaxis_title="SUMO Time (s)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#111827"},
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#e5e7eb", rangemode="tozero", tickfont={"color": "#334155"}, titlefont={"color": "#334155"})
    fig.update_xaxes(gridcolor="#f1f5f9", tickfont={"color": "#334155"}, titlefont={"color": "#334155"})
    return fig


def build_q_chart(q_df, chosen_name):
    fig = go.Figure()
    if not q_df.empty:
        colors = ["#0f766e" if phase == chosen_name else "#94a3b8" for phase in q_df["Phase"]]
        fig.add_trace(
            go.Bar(
                x=q_df["Phase"],
                y=q_df["Q-value"],
                marker_color=colors,
                text=[f"{value:.2f}" for value in q_df["Q-value"]],
                textposition="outside",
                textfont={"color": "#111827", "size": 12},
            )
        )
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        yaxis_title="Q-value",
        xaxis_title=None,
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#111827"},
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#e5e7eb", tickfont={"color": "#334155"}, titlefont={"color": "#334155"})
    fig.update_xaxes(tickfont={"color": "#334155"})
    return fig


st.markdown(
    """
    <style>
      .stApp { background: #f8fafc; color: #0f172a; }
      [data-testid="stHeader"] { background: rgba(248,250,252,0.92); }
      [data-testid="stToolbar"] { right: 1rem; }
      .main .block-container { padding-top: 1.2rem; max-width: 1500px; }
      h1, h2, h3 { letter-spacing: 0; }
      .ops-header {
        display: flex; align-items: flex-end; justify-content: space-between;
        gap: 24px; padding: 8px 0 18px 0; border-bottom: 1px solid #e5e7eb;
      }
      .ops-title h1 { font-size: 2rem; margin: 0; color: #0f172a; }
      .ops-title p { margin: 6px 0 0 0; color: #64748b; font-size: 0.98rem; }
      .status-pill {
        display: inline-flex; align-items: center; gap: 9px; padding: 8px 12px;
        border: 1px solid #bbf7d0; background: #f0fdf4; color: #166534;
        border-radius: 999px; font-weight: 700; white-space: nowrap;
      }
      .pulse {
        width: 10px; height: 10px; border-radius: 999px; background: #16a34a;
        box-shadow: 0 0 0 7px rgba(22, 163, 74, 0.13);
      }
      .metric-panel {
        border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff;
        padding: 14px 16px; min-height: 104px;
      }
      .metric-panel small {
        display: block; color: #64748b; text-transform: uppercase; font-size: 0.72rem;
        font-weight: 800; letter-spacing: 0.06em;
      }
      .metric-panel strong { display: block; color: #0f172a; font-size: 1.55rem; margin-top: 7px; }
      .metric-panel span { display: block; color: #475569; margin-top: 4px; font-size: 0.9rem; }
      .section-title { font-size: 1.05rem; color: #0f172a; font-weight: 800; margin: 0 0 12px; }
      .intersection-shell {
        border: 1px solid #e5e7eb; background: #ffffff; border-radius: 8px;
        padding: 14px; min-height: 580px;
      }
      .phase-ribbon {
        display: inline-flex; padding: 8px 12px; border-radius: 999px;
        background: #0f172a; color: #ffffff; font-size: 0.8rem; font-weight: 800;
        margin-bottom: 12px;
      }
      .intersection {
        position: relative; height: 510px; overflow: hidden; border-radius: 8px;
        background:
          linear-gradient(90deg, transparent 0 39%, #334155 39% 61%, transparent 61%),
          linear-gradient(0deg, transparent 0 39%, #334155 39% 61%, transparent 61%),
          #dbeafe;
        border: 1px solid #cbd5e1;
      }
      .road { position: absolute; background: #334155; }
      .vertical { width: 22%; height: 100%; left: 39%; top: 0; }
      .horizontal { width: 100%; height: 22%; left: 0; top: 39%; }
      .crossbox {
        position: absolute; left: 39%; top: 39%; width: 22%; height: 22%;
        background: #475569; border: 2px solid rgba(255,255,255,0.25);
      }
      .lane-mark { position: absolute; background: rgba(255,255,255,0.72); border-radius: 4px; }
      .v1 { left: 49.5%; top: 3%; width: 1%; height: 34%; }
      .v2 { left: 49.5%; bottom: 3%; width: 1%; height: 34%; }
      .h1 { top: 49.5%; left: 3%; height: 1%; width: 34%; }
      .h2 { top: 49.5%; right: 3%; height: 1%; width: 34%; }
      .signal {
        position: absolute; z-index: 4; min-width: 104px; padding: 9px 10px;
        background: rgba(15,23,42,0.94); color: #fff; border-radius: 8px;
        display: grid; grid-template-columns: 22px 1fr; column-gap: 8px; align-items: center;
      }
      .signal strong { font-size: 0.82rem; line-height: 1; }
      .signal small { grid-column: 2; color: #cbd5e1; font-size: 0.68rem; font-weight: 800; }
      .lamp { grid-row: 1 / span 2; width: 18px; height: 18px; border-radius: 999px; border: 2px solid rgba(255,255,255,0.72); }
      .signal-north { left: 62%; top: 27%; }
      .signal-east { right: 27%; top: 62%; }
      .signal-south { right: 62%; bottom: 27%; }
      .signal-west { left: 27%; bottom: 62%; }
      .vehicles {
        position: absolute; z-index: 3; display: flex; gap: 7px; flex-wrap: wrap;
        max-width: 152px; align-content: flex-start;
      }
      .vehicle-dot {
        width: 17px; height: 28px; border-radius: 5px; background: #f8fafc;
        border: 2px solid #38bdf8; box-shadow: 0 3px 8px rgba(15,23,42,0.35);
        animation-delay: calc(var(--i) * -0.32s);
      }
      .vehicle-placeholder { width: 1px; height: 28px; opacity: 0; }
      .is-moving .vehicle-dot { animation: vehicle-flow 1.7s linear infinite; border-color: #22c55e; }
      .is-caution .vehicle-dot { animation: vehicle-flow 2.8s linear infinite; border-color: #f59e0b; }
      .is-stopped .vehicle-dot { animation: vehicle-idle 1.8s ease-in-out infinite; border-color: #94a3b8; }
      .vehicles-north { left: 43%; top: 6%; width: 80px; transform: rotate(90deg); }
      .vehicles-south { right: 43%; bottom: 6%; width: 80px; transform: rotate(90deg); }
      .vehicles-east { right: 6%; top: 44%; width: 142px; }
      .vehicles-west { left: 6%; bottom: 44%; width: 142px; }
      @keyframes vehicle-flow {
        0% { transform: translateX(0); opacity: 0.25; }
        12% { opacity: 1; }
        100% { transform: translateX(74px); opacity: 0.15; }
      }
      @keyframes vehicle-idle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-1px); }
      }
      .count-badge {
        position: absolute; z-index: 5; width: 38px; height: 38px; border-radius: 999px;
        display: grid; place-items: center; background: #fff; color: #0f172a;
        border: 2px solid #0f172a; font-weight: 900;
      }
      .north-count { left: 43%; top: 23%; }
      .east-count { right: 23%; top: 43%; }
      .south-count { right: 43%; bottom: 23%; }
      .west-count { left: 23%; bottom: 43%; }
      .panel {
        border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff;
        padding: 14px 16px; min-height: 100%;
      }
      .phase-row {
        display: flex; justify-content: space-between; gap: 10px; padding: 10px 0;
        border-bottom: 1px solid #eef2f7; color: #334155;
      }
      .phase-row:last-child { border-bottom: 0; }
      .phase-row strong { color: #0f172a; }
      .empty-notice {
        border: 1px dashed #93c5fd; background: #eff6ff; color: #1e3a8a;
        border-radius: 8px; padding: 18px; font-weight: 700;
      }
      @media (max-width: 900px) {
        .ops-header { display: block; }
        .status-pill { margin-top: 12px; }
        .intersection { height: 420px; }
        .signal { min-width: 86px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

state.start_background_thread()

phase = state.latest_phase_state
phase["active_phase"] = normalize_phase(phase.get("active_phase"))
active_dirs = active_directions(phase)
latest_reasoning = state.latest_reasoning[0] if state.latest_reasoning else {}
chosen = normalize_phase(latest_reasoning.get("chosen_action"))
chosen_name = PHASE_NAME.get(chosen, str(chosen)) if latest_reasoning else "Waiting"
total_queue = sum(count_for(direction) for direction in DIRECTIONS)
waiting = latest_reasoning.get("total_waiting_time_s")
sim_time = latest_reasoning.get("sim_time_s")
signal_mode = "All-red buffer" if phase.get("is_all_red") else "Yellow clearance" if phase.get("is_yellow") else "Green service"

st.markdown(
    """
    <div class="ops-header">
      <div class="ops-title">
        <h1>AI Traffic Operations</h1>
        <p>Realtime intersection control powered by the trained DQN policy, Redis Streams, and SUMO telemetry.</p>
      </div>
      <div class="status-pill"><span class="pulse"></span> Live telemetry</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metrics = [
    ("Active Phase", PHASE_NAME.get(phase.get("active_phase"), "ALL_RED").replace("_", " "), signal_mode),
    ("Model Decision", chosen_name.replace("_", " "), "Latest policy action"),
    ("Observed Queue", str(total_queue), "Vehicles in active ROI windows"),
    (
        "SUMO Time",
        f"{float(sim_time):.0f}s" if sim_time is not None else "Waiting",
        f"Network waiting: {float(waiting):.0f}s" if waiting is not None else "Awaiting agent log",
    ),
]
for col, (label, value, caption) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-panel">
              <small>{escape(label)}</small>
              <strong>{escape(value)}</strong>
              <span>{escape(caption)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1.2, 1])
with left:
    st.markdown('<div class="section-title">Live Intersection View</div>', unsafe_allow_html=True)
    st.markdown(build_intersection_html(phase, active_dirs), unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Traffic Load Trend</div>', unsafe_allow_html=True)
    df = queue_dataframe()
    waiting_df = reasoning_history_dataframe()
    if not df.empty:
        st.plotly_chart(build_queue_chart(df), use_container_width=True)
    elif not waiting_df.empty:
        st.plotly_chart(build_waiting_chart(waiting_df), use_container_width=True)
    else:
        st.markdown('<div class="empty-notice">Waiting for live traffic statistics.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Policy Q-values</div>', unsafe_allow_html=True)
    q_df = reasoning_dataframe(latest_reasoning)
    if q_df.empty:
        st.markdown('<div class="empty-notice">Waiting for DQN reasoning logs.</div>', unsafe_allow_html=True)
    else:
        st.plotly_chart(build_q_chart(q_df, chosen_name), use_container_width=True)

bottom_left, bottom_right = st.columns([1, 1])
with bottom_left:
    st.markdown('<div class="panel"><div class="section-title">Directional Snapshot</div>', unsafe_allow_html=True)
    for direction in DIRECTIONS:
        status = signal_for(direction, phase, active_dirs)
        st.markdown(
            f"""
            <div class="phase-row">
              <span>{escape(direction.title())}</span>
              <strong>{escape(status.replace("_", " "))}</strong>
              <span>{count_for(direction)} vehicles</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown('<div class="panel"><div class="section-title">Recent Model Reasoning</div>', unsafe_allow_html=True)
    if state.latest_reasoning:
        rows = []
        for item in state.latest_reasoning[:8]:
            action = PHASE_NAME.get(normalize_phase(item.get("chosen_action")), item.get("chosen_action"))
            rows.append(
                {
                    "sim_time_s": item.get("sim_time_s"),
                    "chosen_phase": str(action).replace("_", " "),
                    "waiting_time_s": item.get("total_waiting_time_s"),
                    "exploration": item.get("exploration", False),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Waiting for model decisions.")
    st.markdown("</div>", unsafe_allow_html=True)

time.sleep(0.75)
st.rerun()
