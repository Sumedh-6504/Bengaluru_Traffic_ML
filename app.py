import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import folium
from streamlit_folium import st_folium

# --- 1. PRO CONFIGURATION (WIDE MODE) ---
st.set_page_config(
    page_title="Bengaluru Traffic AI",
    layout="wide",  # Uses full screen width
    page_icon="🚦"
)


# --- 2. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    base_path = 'D:/Temp/Bengaluru_Traffic_ML/.venv/eda_and_model/models/saved_models/'
    vol_model = joblib.load(base_path + 'xgb_volume_model.joblib')
    cong_model = joblib.load(base_path + 'xgb_congestion_model.joblib')
    le_area = joblib.load(base_path + 'le_area.joblib')
    le_road = joblib.load(base_path + 'le_road.joblib')
    return vol_model, cong_model, le_area, le_road


try:
    vol_model, cong_model, le_area, le_road = load_artifacts()
except Exception as e:
    st.error(f"System Error: Models not found.\n{e}")
    st.stop()


# --- 3. LOAD HISTORY ---
@st.cache_data
def load_data():
    df = pd.read_csv('D:/Temp/Bengaluru_Traffic_ML/.venv/data/processed/traffic_ml_ready.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    rename_map = {'Area_ID': 'Area ID'}
    df.rename(columns=rename_map, inplace=True)
    return df


history_df = load_data()


# --- 4. PREDICTION ENGINE (HELPER) ---
def get_prediction(area_id_input, road_id_input, date_val, weather_val, roadwork_val, incident_val):
    # Time Encoding
    sel_date = pd.to_datetime(date_val)
    sel_month = sel_date.month
    day_of_week = sel_date.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    month_sin = np.sin(2 * np.pi * sel_month / 12)
    month_cos = np.cos(2 * np.pi * sel_month / 12)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Weather Encoding
    weather_dict = {'weather_Clear': 0, 'weather_Fog': 0, 'weather_Overcast': 0, 'weather_Rain': 0, 'weather_Windy': 0}
    w_key = f"weather_{weather_val}"
    if w_key in weather_dict: weather_dict[w_key] = 1

    # Interaction
    w_impact = 1 * (0 if weather_val == "Clear" else 1) if incident_val else 0

    # Smart Lag Lookup
    lag_date = sel_date - datetime.timedelta(days=7)
    mask = (history_df['Date'] == lag_date) & (history_df['Area ID'] == area_id_input) & (
                history_df['Road_ID'] == road_id_input)
    row = history_df[mask]

    # Backtracking
    backtrack_count = 0
    if row.empty:
        max_db_date = history_df['Date'].max()
        temp_date = lag_date
        while temp_date > max_db_date or row.empty:
            if backtrack_count > 3: break
            temp_date = temp_date - datetime.timedelta(days=364)
            mask = (history_df['Date'] == temp_date) & (history_df['Area ID'] == area_id_input) & (
                        history_df['Road_ID'] == road_id_input)
            row = history_df[mask]
            backtrack_count += 1

    if not row.empty:
        v_lag = row['Log_Traffic_Volume'].values[0]
        c_lag = row['Congestion Level'].values[0]
        # Calculate historical volume for comparison
        hist_vol_display = int(np.expm1(v_lag))
    else:
        # Average Fallback
        subset = history_df[(history_df['Area ID'] == area_id_input) & (history_df['Road_ID'] == road_id_input)]
        if not subset.empty:
            v_lag = subset['Log_Traffic_Volume'].mean()
            c_lag = subset['Congestion Level'].mean()
        else:
            v_lag = history_df['Log_Traffic_Volume'].mean()
            c_lag = history_df['Congestion Level'].mean()
        hist_vol_display = int(np.expm1(v_lag))

    # Construct Input
    input_data = pd.DataFrame([{
        'month_sin': month_sin, 'month_cos': month_cos,
        'day_sin': day_sin, 'day_cos': day_cos,
        'is_weekend': is_weekend,
        'Area ID': area_id_input,
        'Road_ID': road_id_input,
        'Roadwork_encoded': 1 if roadwork_val else 0,
        'weather_incident_impact': w_impact,
        'weather_Clear': weather_dict['weather_Clear'],
        'weather_Fog': weather_dict['weather_Fog'],
        'weather_Overcast': weather_dict['weather_Overcast'],
        'weather_Rain': weather_dict['weather_Rain'],
        'weather_Windy': weather_dict['weather_Windy'],
        'Volume_Lag_7': v_lag,
        'Congestion_Lag_7': c_lag
    }])

    # Predict
    log_vol = vol_model.predict(input_data)[0]
    act_vol = int(np.expm1(log_vol))
    act_cong = cong_model.predict(input_data)[0]

    return act_vol, act_cong, hist_vol_display


# --- 5. UI LAYOUT (SIDEBAR INPUTS) ---

st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure the scenario below.")

# Sidebar Inputs
selected_area = st.sidebar.selectbox("📍 Select Area", le_area.classes_)
selected_road = st.sidebar.selectbox("🛣️ Select Road", le_road.classes_)
selected_date = st.sidebar.date_input("📅 Date", datetime.date(2025, 1, 15))
selected_weather = st.sidebar.selectbox("☁️ Weather", ['Clear', 'Rain', 'Fog', 'Overcast', 'Windy'])

st.sidebar.divider()
is_roadwork = st.sidebar.toggle("🚧 Roadwork Activity", value=False)
is_incident = st.sidebar.toggle("🚨 Reported Incident", value=False)

# Initialize Session State
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_vol' not in st.session_state: st.session_state['res_vol'] = 0
if 'res_cong' not in st.session_state: st.session_state['res_cong'] = 0
if 'hist_vol' not in st.session_state: st.session_state['hist_vol'] = 0

# --- 6. MAIN ACTION BUTTON ---
if st.sidebar.button("🚀 Analyze Traffic Scenario", type="primary"):
    with st.spinner("Crunching historical data & predicting congestion..."):
        # 1. Transform IDs
        a_id = le_area.transform([selected_area])[0]
        r_id = le_road.transform([selected_road])[0]

        # 2. Get Single Prediction
        p_vol, p_cong, h_vol = get_prediction(a_id, r_id, selected_date, selected_weather, is_roadwork, is_incident)

        # 3. Store in State
        st.session_state['res_vol'] = p_vol
        st.session_state['res_cong'] = p_cong
        st.session_state['hist_vol'] = h_vol
        st.session_state['analyzed'] = True

# --- 7. DASHBOARD DISPLAY ---

st.title("🚦 Bangalore Traffic Intelligence Dashboard")

if st.session_state['analyzed']:

    # --- ROW 1: METRICS ---
    col1, col2, col3 = st.columns(3)

    vol = st.session_state['res_vol']
    cong = st.session_state['res_cong']
    hist = st.session_state['hist_vol']

    # Calculate difference from history (Lag)
    delta_vol = vol - hist

    with col1:
        st.metric("Predicted Volume", f"{vol:,} Cars", delta=f"{delta_vol:,} vs Avg", delta_color="inverse")

    with col2:
        # Logic for Color/Status
        if cong < 40:
            status, color_code = "Smooth Flow", "normal"  # normal = green in streamlit metric
        elif cong < 55:
            status, color_code = "Moderate", "off"  # off = gray/yellowish
        else:
            status, color_code = "Heavy Congestion", "inverse"  # inverse = red

        st.metric("Congestion Level", f"{cong:.1f} / 100", status, delta_color=color_code)

    with col3:
        # Date Context
        day_name = pd.to_datetime(selected_date).strftime("%A")
        st.info(f"**Context:** Prediction for a {selected_weather} {day_name}.")

    st.divider()

    # --- ROW 2: SPLIT VIEW (DETAILS + MAP) ---
    map_col, details_col = st.columns([2, 1])

    with map_col:
        st.subheader("🗺️ City-Wide Congestion Heatmap")

        # Generate Map Logic
        LOCATION_COORDS = {
            "Indiranagar": [12.9784, 77.6408], "Whitefield": [12.9698, 77.7500],
            "Koramangala": [12.9352, 77.6245], "M.G. Road": [12.9756, 77.6066],
            "Jayanagar": [12.9308, 77.5838], "Hebbal": [13.0354, 77.5988],
            "Yeshwanthpur": [13.0238, 77.5529], "Electronic City": [12.8452, 77.6602]
        }

        m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)

        # Batch Prediction Loop
        for map_area, coords in LOCATION_COORDS.items():
            try:
                # Get ID & Dummy Road
                map_a_id = le_area.transform([map_area])[0]
                dummy_row = history_df[history_df['Area ID'] == map_a_id].iloc[0]
                map_r_id = dummy_row['Road_ID']

                # Predict
                _, map_cong, _ = get_prediction(map_a_id, map_r_id, selected_date, selected_weather, is_roadwork,
                                                is_incident)

                # Color Logic
                if map_cong < 40:
                    c, fc = 'green', '#00FF00'
                elif map_cong < 55:
                    c, fc = 'orange', '#FFA500'
                else:
                    c, fc = 'red', '#FF0000'

                # Draw
                folium.Circle(
                    location=coords, radius=800, color=c, fill=True, fill_color=fc, fill_opacity=0.4,
                    popup=f"<b>{map_area}</b><br>Congestion: {map_cong:.1f}"
                ).add_to(m)
            except:
                continue

        st_folium(m, width=None, height=400)  # Full width of column

    with details_col:
        st.subheader("💡 Analysis")
        st.markdown(f"""
        **Target Road:** {selected_road}

        **Factors considered:**
        - **Seasonality:** {pd.to_datetime(selected_date).strftime('%B')} Patterns
        - **Weather:** {selected_weather} Impact
        - **Activity:** {'Roadwork Detected' if is_roadwork else 'No Roadwork'}
        - **History:** Based on 7-day lagged trends.
        """)

        if cong > 60:
            st.warning("⚠️ High delay probability. Consider alternate routes.")
        else:
            st.success("✅ Traffic flow expected to be normal.")

else:
    # Initial State
    st.info("👈 Select a date and location from the sidebar, then click 'Analyze Traffic Scenario'.")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Bangalore_Traffic.jpg/640px-Bangalore_Traffic.jpg",
        caption="Bangalore Traffic Analytics")