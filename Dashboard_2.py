import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
from fpdf import FPDF
import io

# 1. Page Config
st.set_page_config(layout="wide", page_title="DGB Farmer Logistics Hub")

@st.cache_data(ttl=600)
def load_live_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/15V_IFm4Q14tMsg-vUV7r7IhtiGzNSjvBC58QkX9ydEY/export?format=csv&gid=0"
    df = pd.read_csv(sheet_url)
    df = df.rename(columns={'District_2': 'District', 'Subcounty_2': 'Subcounty', 'Parish_2': 'Parish'})
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    return df.dropna(subset=['Latitude', 'Longitude'])

df = load_live_data()

# --- STABLE CLUSTERING LOGIC ---
def assign_stable_clusters(data, target_size=40):
    processed_list = []
    for dist in data['District'].unique():
        dist_data = data[data['District'] == dist].copy()
        n_clusters = max(1, len(dist_data) // target_size)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        dist_data.loc[:, 'Cluster_ID_Num'] = kmeans.fit_predict(dist_data[['Latitude', 'Longitude']])
        centers = kmeans.cluster_centers_
        dist_code = "HC" if "hoima city" in dist.lower() else dist[:2].upper()
        
        def get_label(row):
            pt = [row['Latitude'], row['Longitude']]
            closest_idx = distance.cdist([pt], centers).argmin()
            sub_code = str(row['Subcounty'])[:3].upper() if pd.notnull(row['Subcounty']) else "UNK"
            return f"UG-{dist_code}{sub_code}-{str(closest_idx + 1).zfill(3)}"
        
        dist_data.loc[:, 'Cluster_Label'] = dist_data.apply(get_label, axis=1)
        processed_list.append(dist_data)
    return pd.concat(processed_list)

full_data = assign_stable_clusters(df)

# --- SIDEBAR: DOWNLOAD TOOLS ---
st.sidebar.title("📦 Logistics Exports")
selected_view = st.sidebar.selectbox("Filter by District", ["All"] + sorted(full_data['District'].unique().tolist()))
view_df = full_data if selected_view == "All" else full_data[full_data['District'] == selected_view]

# Calculate Centroids for the Map and Export
centroids = view_df.groupby(['District', 'Cluster_Label']).agg({
    'Latitude': 'mean', 
    'Longitude': 'mean'
}).reset_index()

# Download Meeting Points CSV
csv_centroids = centroids.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📍 Download Meeting Point CSV",
    data=csv_centroids,
    file_name=f"Meeting_Points_{selected_view}.csv",
    mime="text/csv",
    help="Download a list of central GPS coordinates for each cluster badge."
)

# --- MAP VISUALIZATION ---
st.title(f"🌍 Farmer Network Hub: {selected_view}")

m = folium.Map(location=[view_df['Latitude'].mean(), view_df['Longitude'].mean()], zoom_start=9, tiles="cartodbpositron")

# Cluster Colors
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(view_df['Cluster_Label'].unique()))}

# 1. Farmer Dots
for _, row in view_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        color=color_map[row['Cluster_Label']],
        fill=True,
        tooltip=f"<b>{row['Name']}</b><br>ID: {row['Cluster_Label']}"
    ).add_to(m)

# 2. Black Badge Meeting Points
for _, row in centroids.iterrows():
    label_num = row['Cluster_Label'].split('-')[-1]
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: sans-serif; color: white; background: #000; 
            border-radius: 50%; width: 28px; height: 28px; display: flex; 
            align-items: center; justify-content: center; font-size: 8pt; 
            font-weight: bold; border: 2px solid white; box-shadow: 0px 0px 5px rgba(0,0,0,0.5);">
            {int(label_num)}</div>"""),
        tooltip=f"MEETING POINT: {row['Cluster_Label']}"
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- REGISTRY TABLE ---
st.subheader("Cluster Capacity Registry")
summary = full_data.groupby(['District', 'Cluster_Label']).size().reset_index(name='Farmer Count')
st.dataframe(summary, width="stretch")