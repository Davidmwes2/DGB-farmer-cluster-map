import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
from fpdf import FPDF
import numpy as np

# 1. Page Config
st.set_page_config(layout="wide", page_title="DGB Live Farmer Pipeline")

# 2. Live Data Connection (Google Sheets)
@st.cache_data(ttl=600) # Refreshes every 10 mins
def load_live_data():
    # Replace the URL below with your 'Publish to Web' CSV link
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRh8cGfXDbYcmgH-YTq0ZS-C_QASjwjXDNNoJwhVSULBQJum4nncf_slx6ivJBdNEHi6B60plva-hLZ/pub?gid=0&single=true&output=csv"
    df = pd.read_csv(sheet_url)
    df = df.rename(columns={'District_2': 'District', 'Subcounty_2': 'Subcounty', 'Parish_2': 'Parish'})
    # Critical: Ensure Lat/Long are numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    return df.dropna(subset=['Latitude', 'Longitude'])

df = load_live_data()

# --- OPTION B: NEAREST NEIGHBOR LOGIC ---
def assign_stable_clusters(data, target_size=40):
    processed_list = []
    
    for dist in data['District'].unique():
        dist_data = data[data['District'] == dist].copy()
        
        # Determine how many clusters we ALREADY should have based on district size
        n_clusters = max(1, len(dist_data) // target_size)
        
        # 1. Establish the "Base" Cluster Centers using KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        dist_data.loc[:, 'Cluster_ID_Num'] = kmeans.fit_predict(dist_data[['Latitude', 'Longitude']])
        centers = kmeans.cluster_centers_
        
        # 2. Assign naming convention
        dist_code = "HC" if "hoima city" in dist.lower() else dist[:2].upper()
        
        def get_label(row):
            # Find closest center (Nearest Neighbor)
            pt = [row['Latitude'], row['Longitude']]
            closest_idx = distance.cdist([pt], centers).argmin()
            # Find subcounty for the label
            sub_code = str(row['Subcounty'])[:3].upper() if pd.notnull(row['Subcounty']) else "UNK"
            return f"UG-{dist_code}{sub_code}-{str(closest_idx + 1).zfill(3)}"
        
        dist_data.loc[:, 'Cluster_Label'] = dist_data.apply(get_label, axis=1)
        processed_list.append(dist_data)
        
    return pd.concat(processed_list)

full_data = assign_stable_clusters(df)

# --- SIDEBAR & HEALTH CHECK ---
st.sidebar.title("🛡️ Pipeline Monitor")
st.sidebar.info(f"Last Sync: {pd.Timestamp.now().strftime('%H:%M:%S')}")

# Threshold Warning
cluster_counts = full_data['Cluster_Label'].value_counts()
overloaded = cluster_counts[cluster_counts > 60]
if not overloaded.empty:
    st.sidebar.error(f"⚠️ {len(overloaded)} Clusters exceed 60-farmer limit!")
    st.sidebar.write(overloaded)

# --- VISUALIZATION ---
selected_view = st.sidebar.selectbox("View District", ["All"] + sorted(full_data['District'].unique().tolist()))
view_df = full_data if selected_view == "All" else full_data[full_data['District'] == selected_view]

st.title(f"🌍 Live Farmer Network: {selected_view}")

# Map Logic
m = folium.Map(location=[view_df['Latitude'].mean(), view_df['Longitude'].mean()], zoom_start=9, tiles="cartodbpositron")

# Color and Badge Logic
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(view_df['Cluster_Label'].unique()))}

for _, row in view_df.iterrows():
    count = cluster_counts[row['Cluster_Label']]
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6 if count > 60 else 4,
        color='red' if count > 60 else color_map[row['Cluster_Label']],
        fill=True,
        tooltip=f"<b>{row['Name']}</b><br>ID: {row['Cluster_Label']}<br>Group Size: {count}"
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- PDF & DATA TABLE ---
st.subheader("Cluster Capacity Registry")
summary = full_data.groupby(['District', 'Cluster_Label']).size().reset_index(name='Farmer Count')
st.dataframe(summary.style.apply(lambda x: ['background-color: #ffcccc' if v > 60 else '' for v in x], subset=['Farmer Count']), width="stretch")