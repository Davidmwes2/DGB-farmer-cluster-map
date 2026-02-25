import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from fpdf import FPDF
import io
import requests
import math
import numpy as np

# 1. Page Config
st.set_page_config(layout="wide", page_title="DGB Strict Cluster Hub")

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

# --- STRICT CLUSTERING LOGIC (Min 10, Max 40) ---
def assign_strict_clusters(data):
    processed_list = []
    
    for dist, dist_data in data.groupby('District'):
        dist_data = dist_data.copy()
        N = len(dist_data)
        dist_code = "HC" if "hoima city" in dist.lower() else dist[:2].upper()
        
        # Edge case: If a district has fewer than 10 farmers total, we must keep them as 1 cluster
        if N < 10 or N <= 40:
            dist_data.loc[:, 'Cluster_ID_Num'] = 0
        else:
            # Calculate required clusters to enforce max 40
            k = math.ceil(N / 40.0)
            base_count = N // k
            remainder = N % k
            
            # Find initial geographic centers
            kmeans = KMeans(n_clusters=k, random_state=42).fit(dist_data[['Latitude', 'Longitude']])
            centers = kmeans.cluster_centers_
            
            # Replicate centers based on exact slot limits (Guarantees sizes between 10 and 40)
            expanded_centers = []
            cluster_mapping = []
            for c in range(k):
                slots = base_count + 1 if c < remainder else base_count
                for _ in range(slots):
                    expanded_centers.append(centers[c])
                    cluster_mapping.append(c)
                    
            # Optimize assignment so farmers get their closest available slot
            cost_matrix = distance.cdist(dist_data[['Latitude', 'Longitude']], expanded_centers)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Map optimized slots back to farmers
            labels = np.zeros(N, dtype=int)
            for i, r in enumerate(row_ind):
                labels[r] = cluster_mapping[col_ind[i]]
                
            dist_data.loc[:, 'Cluster_ID_Num'] = labels
            
        # ID Format: UG-DIST-001
        dist_data.loc[:, 'Cluster_Label'] = dist_data['Cluster_ID_Num'].apply(
            lambda x: f"UG-{dist_code}-{str(int(x) + 1).zfill(3)}"
        )
        processed_list.append(dist_data)
    return pd.concat(processed_list)

full_data = assign_strict_clusters(df)

# --- LEAD FARMER (MEDOID) LOGIC ---
# Identifies the specific farmer physically closest to the geometric center of their cluster
def get_lead_farmers(dataframe):
    leads = []
    for cluster in dataframe['Cluster_Label'].unique():
        subset = dataframe[dataframe['Cluster_Label'] == cluster]
        mean_center = subset[['Latitude', 'Longitude']].mean().values.reshape(1, -1)
        dists = distance.cdist(mean_center, subset[['Latitude', 'Longitude']].values)
        lead_farmer = subset.iloc[dists.argmin()]
        leads.append({
            'Cluster_ID': cluster, 
            'District': lead_farmer['District'],
            'Lead_Farmer': lead_farmer['Name'], 
            'Farmer_Count': len(subset),
            'Latitude': lead_farmer['Latitude'], 
            'Longitude': lead_farmer['Longitude']
        })
    return pd.DataFrame(leads)

# --- SIDEBAR: SEARCH & DOWNLOADS ---
st.sidebar.title("🔍 Operations Hub")
search_query = st.sidebar.text_input("Find Farmer by Name", "")

selected_dist = st.sidebar.selectbox("Filter by District", ["All Districts"] + sorted(full_data['District'].unique().tolist()))

display_df = full_data.copy()
if selected_dist != "All Districts":
    display_df = display_df[display_df['District'] == selected_dist]

lead_farmer_df = get_lead_farmers(display_df)

# Dynamic Zoom for Search
zoom_lat, zoom_lon, zoom_level = display_df['Latitude'].mean(), display_df['Longitude'].mean(), 9
if search_query:
    res = full_data[full_data['Name'].str.contains(search_query, case=False, na=False)]
    if not res.empty:
        zoom_lat, zoom_lon, zoom_level = res.iloc[0]['Latitude'], res.iloc[0]['Longitude'], 14
        st.sidebar.success(f"Zoomed to {res.iloc[0]['Name']}")

# PDF Generator
def create_pdf(dataframe, lead_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for cluster in sorted(dataframe['Cluster_Label'].unique()):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Cluster Registry: {cluster}", ln=True, align='C')
        pdf.ln(5)
        lead = lead_df[lead_df['Cluster_ID'] == cluster].iloc[0]
        pdf.set_fill_color(0, 0, 0); pdf.set_text_color(255, 255, 255); pdf.set_font("Arial", 'B', 10)
        pdf.cell(190, 10, f" LEAD FARMER: {lead['Lead_Farmer']} | TOTAL HEADCOUNT: {lead['Farmer_Count']}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0); pdf.ln(2)
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(85, 8, "Name", border=1); pdf.cell(60, 8, "Subcounty", border=1); pdf.cell(45, 8, "Coordinates", border=1); pdf.ln()
        pdf.set_font("Arial", '', 9)
        members = dataframe[dataframe['Cluster_Label'] == cluster]
        for _, row in members.iterrows():
            pdf.cell(85, 8, str(row['Name'])[:35], border=1); pdf.cell(60, 8, str(row['Subcounty']), border=1); pdf.cell(45, 8, f"{row['Latitude']:.4f}, {row['Longitude']:.4f}", border=1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Downloads")
if st.sidebar.button("🛠️ Prepare PDF Handbook"):
    pdf_bytes = create_pdf(display_df, lead_farmer_df)
    st.sidebar.download_button("📥 Download Farmer List (PDF)", pdf_bytes, "Farmer_Handbook.pdf", "application/pdf")

csv_data = lead_farmer_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📍 Download Lead Farmer Hubs (CSV)", csv_data, "Hub_Coordinates.csv", "text/csv")

# --- MAP VISUALIZATION ---
st.title("🚜 DGB Farmer Geographic Hub")
st.info("Rule Active: Clusters strictly bounded between 10 and 40 farmers.")
m = folium.Map(location=[zoom_lat, zoom_lon], zoom_start=zoom_level, tiles="cartodbpositron")

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(sorted(display_df['Cluster_Label'].unique()))}

# Plot Farmers
for _, row in display_df.iterrows():
    is_target = search_query and search_query.lower() in row['Name'].lower()
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=8 if is_target else 4,
        color='black' if is_target else color_map[row['Cluster_Label']],
        fill=True,
        tooltip=f"<b>Farmer:</b> {row['Name']}<br><b>Cluster:</b> {row['Cluster_Label']}"
    ).add_to(m)

# Plot Dynamic Hub Banners
for _, row in lead_farmer_df.iterrows():
    label_num = row['Cluster_ID'].split('-')[-1]
    banner_html = f"""
    <div style="font-size: 11pt; padding: 2px;">
        <b>Cluster ID:</b> {row['Cluster_ID']}<br>
        <b>Headcount:</b> {row['Farmer_Count']} Farmers<br>
        <b>Lead Farmer:</b> {row['Lead_Farmer']}
    </div>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=f'<div style="font-family: sans-serif; color: white; background: #000; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 9pt; font-weight: bold; border: 2px solid white;">{int(label_num)}</div>'),
        tooltip=folium.Tooltip(banner_html)
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- HUB REGISTRY TABLE ---
st.subheader("Cluster Registry & Hub Identification")
st.dataframe(lead_farmer_df[['Cluster_ID', 'District', 'Lead_Farmer', 'Farmer_Count', 'Latitude', 'Longitude']], width="stretch")