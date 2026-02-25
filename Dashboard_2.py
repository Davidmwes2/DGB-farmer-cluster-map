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
st.set_page_config(layout="wide", page_title="DGB Farmer cluster Hub")

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

# --- CLUSTERING LOGIC (Min 10, Max 40) ---
def assign_stable_clusters(data, target_size=40, min_size=10):
    processed_list = []
    for dist in data['District'].unique():
        dist_data = data[data['District'] == dist].copy()
        
        # Calculate optimal number of clusters
        total_farmers = len(dist_data)
        
        if total_farmers <= min_size:
            n_clusters = 1 # Keep as one group if below minimum
        else:
            # We target 40, but ensure we don't create clusters < 10
            n_clusters = max(1, total_farmers // target_size)
            # Check if this split would result in clusters smaller than 10 on average
            if total_farmers / n_clusters < min_size:
                n_clusters = max(1, total_farmers // min_size)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        dist_data.loc[:, 'Cluster_ID_Num'] = kmeans.fit_predict(dist_data[['Latitude', 'Longitude']])
        
        dist_code = "HC" if "hoima city" in dist.lower() else dist[:2].upper()
        
        def get_label(row):
            sub_code = str(row['Subcounty'])[:3].upper() if pd.notnull(row['Subcounty']) else "UNK"
            return f"UG-{dist_code}{sub_code}-{str(int(row['Cluster_ID_Num']) + 1).zfill(3)}"
        
        dist_data.loc[:, 'Cluster_Label'] = dist_data.apply(get_label, axis=1)
        processed_list.append(dist_data)
    return pd.concat(processed_list)

full_data = assign_stable_clusters(df)

# --- IDENTIFY LEAD FARMER (MEDOID) ---
def get_lead_farmers(dataframe):
    leads = []
    for cluster in dataframe['Cluster_Label'].unique():
        subset = dataframe[dataframe['Cluster_Label'] == cluster]
        mean_center = subset[['Latitude', 'Longitude']].mean().values.reshape(1, -1)
        dists = distance.cdist(mean_center, subset[['Latitude', 'Longitude']].values)
        lead_farmer = subset.iloc[dists.argmin()]
        leads.append({
            'Cluster_Label': cluster,
            'Latitude': lead_farmer['Latitude'],
            'Longitude': lead_farmer['Longitude'],
            'Lead_Farmer': lead_farmer['Name'],
            'Parish': lead_farmer['Parish'],
            'Size': len(subset)
        })
    return pd.DataFrame(leads)

# --- UI & DOWNLOADS ---
st.sidebar.title("📦 Logistics Center")
selected_view = st.sidebar.selectbox("Filter District", ["All"] + sorted(full_data['District'].unique().tolist()))
view_df = full_data if selected_view == "All" else full_data[full_data['District'] == selected_view]
lead_farmer_df = get_lead_farmers(view_df)

# Sidebar PDF and CSV buttons (restored from previous steps)
csv_leads = lead_farmer_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📍 Download Hub GPS (CSV)", csv_leads, f"Hubs_{selected_view}.csv", "text/csv")

# --- MAP VISUALIZATION ---
st.title(f"🌍 Farmer Network: {selected_view}")
st.markdown(f"**Target Cluster Size:** 10 to 40 Farmers")

m = folium.Map(location=[view_df['Latitude'].mean(), view_df['Longitude'].mean()], zoom_start=9, tiles="cartodbpositron")

# Cluster Colors
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(view_df['Cluster_Label'].unique()))}

for _, row in view_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=color_map[row['Cluster_Label']], fill=True,
        tooltip=f"Farmer: {row['Name']}<br>Cluster: {row['Cluster_Label']}"
    ).add_to(m)

for _, row in lead_farmer_df.iterrows():
    label_num = row['Cluster_Label'].split('-')[-1]
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: sans-serif; color: white; background: #000; 
            border-radius: 50%; width: 28px; height: 28px; display: flex; 
            align-items: center; justify-content: center; font-size: 8pt; 
            font-weight: bold; border: 2px solid white;">{int(label_num)}</div>"""),
        tooltip=f"HUB: {row['Lead_Farmer']} ({row['Cluster_Label']}) - {row['Size']} farmers"
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- CLUSTER REGISTRY TABLE ---
st.subheader("Cluster Performance Summary")
st.dataframe(lead_farmer_df[['Cluster_Label', 'Lead_Farmer', 'Size', 'Parish', 'Latitude', 'Longitude']], width="stretch")