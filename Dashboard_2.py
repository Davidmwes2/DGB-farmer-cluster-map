import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
from fpdf import FPDF
import io
import requests

# 1. Page Config
st.set_page_config(layout="wide", page_title="DGB Global Farmer Hub")

# 2. Live Data Loading
@st.cache_data(ttl=600)
def load_live_data():
    sheet_id = "15V_IFm4Q14tMsg-vUV7r7IhtiGzNSjvBC58QkX9ydEY"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        response = requests.get(sheet_url)
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={'District_2': 'District', 'Subcounty_2': 'Subcounty', 'Parish_2': 'Parish'})
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        return df.dropna(subset=['Latitude', 'Longitude'])
    except:
        st.error("Connection Error with Google Sheets.")
        st.stop()

df = load_live_data()

# --- CLUSTERING LOGIC (Subcounty Locked) ---
def assign_stable_clusters(data, target_size=40, min_size=10):
    processed_list = []
    for (dist, sub), sub_data in data.groupby(['District', 'Subcounty']):
        sub_data = sub_data.copy()
        total = len(sub_data)
        dist_code = "HC" if "hoima city" in dist.lower() else dist[:2].upper()
        sub_code = str(sub)[:3].upper() if pd.notnull(sub) else "UNK"

        if total <= min_size:
            n_clusters = 1
        else:
            n_clusters = max(1, total // target_size)
            if total / n_clusters < min_size:
                n_clusters = max(1, total // min_size)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        sub_data.loc[:, 'Cluster_ID_Num'] = kmeans.fit_predict(sub_data[['Latitude', 'Longitude']])
        sub_data.loc[:, 'Cluster_Label'] = sub_data['Cluster_ID_Num'].apply(
            lambda x: f"UG-{dist_code}{sub_code}-{str(int(x) + 1).zfill(3)}"
        )
        processed_list.append(sub_data)
    return pd.concat(processed_list)

full_data = assign_stable_clusters(df)

# --- LEAD FARMER LOGIC ---
def get_lead_farmers(dataframe):
    leads = []
    for cluster in dataframe['Cluster_Label'].unique():
        subset = dataframe[dataframe['Cluster_Label'] == cluster]
        mean_center = subset[['Latitude', 'Longitude']].mean().values.reshape(1, -1)
        dists = distance.cdist(mean_center, subset[['Latitude', 'Longitude']].values)
        lead_farmer = subset.iloc[dists.argmin()]
        leads.append({
            'Cluster_ID': cluster, 'District': lead_farmer['District'],
            'Subcounty': lead_farmer['Subcounty'], 'Lead_Farmer': lead_farmer['Name'],
            'Farmer_Count': len(subset), 'Latitude': lead_farmer['Latitude'],
            'Longitude': lead_farmer['Longitude']
        })
    return pd.DataFrame(leads)

# --- SIDEBAR: SEARCH & INTERACTIVE FOCUS ---
st.sidebar.title("🔍 Map Controls")
search_query = st.sidebar.text_input("Find Farmer by Name", "")

# Filter by District
dist_options = ["All Districts"] + sorted(full_data['District'].unique().tolist())
selected_dist = st.sidebar.selectbox("Filter District", dist_options)

display_df = full_data.copy()
if selected_dist != "All Districts":
    display_df = display_df[display_df['District'] == selected_dist]

lead_farmer_df = get_lead_farmers(display_df)

# Interactive Focus: Link Table to Map
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Cluster Focus")
focus_cluster = st.sidebar.selectbox("Select Cluster ID to Highlight", ["None"] + sorted(lead_farmer_df['Cluster_ID'].tolist()))

# Set Dynamic Map Center/Zoom
zoom_lat, zoom_lon, zoom_level = display_df['Latitude'].mean(), display_df['Longitude'].mean(), 9

if focus_cluster != "None":
    focus_data = lead_farmer_df[lead_farmer_df['Cluster_ID'] == focus_cluster].iloc[0]
    zoom_lat, zoom_lon, zoom_level = focus_data['Latitude'], focus_data['Longitude'], 13
elif search_query:
    res = full_data[full_data['Name'].str.contains(search_query, case=False, na=False)]
    if not res.empty:
        zoom_lat, zoom_lon, zoom_level = res.iloc[0]['Latitude'], res.iloc[0]['Longitude'], 14

# --- MAP ---
st.title("🚜 Farmer Intelligence Dashboard")
m = folium.Map(location=[zoom_lat, zoom_lon], zoom_start=zoom_level, tiles="cartodbpositron")

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(sorted(display_df['Cluster_Label'].unique()))}

for _, row in display_df.iterrows():
    # If a cluster is focused, grey out everyone else
    opacity = 1.0
    color = color_map[row['Cluster_Label']]
    if focus_cluster != "None" and row['Cluster_Label'] != focus_cluster:
        opacity = 0.1
        color = 'grey'
    
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=color, fill=True, fill_opacity=opacity,
        tooltip=f"<b>{row['Name']}</b><br>ID: {row['Cluster_Label']}"
    ).add_to(m)

for _, row in lead_farmer_df.iterrows():
    if focus_cluster == "None" or row['Cluster_ID'] == focus_cluster:
        label_num = row['Cluster_ID'].split('-')[-1]
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.DivIcon(html=f'<div style="font-family: sans-serif; color: white; background: #000; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 8pt; font-weight: bold; border: 2px solid white;">{int(label_num)}</div>'),
            tooltip=f"HUB: {row['Lead_Farmer']}"
        ).add_to(m)

st_folium(m, width=1300, height=550)

# --- REGISTRY TABLE (With Farmer Count) ---
st.subheader("Cluster Registry & Hub Identification")
st.markdown("_Click column headers to sort. Use the sidebar 'Cluster Focus' to find these on the map._")
st.dataframe(lead_farmer_df[['Cluster_ID', 'Subcounty', 'Lead_Farmer', 'Farmer_Count', 'Latitude', 'Longitude']], width="stretch")