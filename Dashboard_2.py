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
st.set_page_config(layout="wide", page_title="DGB Farmer Logistics Hub")

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

# --- CLUSTERING LOGIC ---
def assign_stable_clusters(data, target_size=40):
    processed_list = []
    for dist in data['District'].unique():
        dist_data = data[data['District'] == dist].copy()
        n_clusters = max(1, round(len(dist_data) / target_size))
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
            'Parish': lead_farmer['Parish']
        })
    return pd.DataFrame(leads)

# --- SIDEBAR & VIEW ---
st.sidebar.title("📦 Logistics Center")
selected_view = st.sidebar.selectbox("Filter District", ["All"] + sorted(full_data['District'].unique().tolist()))
view_df = full_data if selected_view == "All" else full_data[full_data['District'] == selected_view]
lead_farmer_df = get_lead_farmers(view_df)

# PDF Handbook Generator
def create_pdf(dataframe, lead_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for cluster in sorted(dataframe['Cluster_Label'].unique()):
        pdf.add_page()
        lead = lead_df[lead_df['Cluster_Label'] == cluster].iloc[0]
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"DGB Field Handbook: {cluster}", ln=True, align='C')
        pdf.ln(5)
        
        # Lead Farmer Highlight Box
        pdf.set_fill_color(0, 0, 0) # Black box
        pdf.set_text_color(255, 255, 255) # White text
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(190, 10, f" CENTRAL MEETING HUB: {lead['Lead_Farmer']}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 10)
        pdf.cell(190, 8, f" Location: {lead['Latitude']:.5f}, {lead['Longitude']:.5f} (Parish: {lead['Parish']})", border='LRB', ln=True)
        pdf.ln(10)
        
        # Member Table
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(80, 10, "Farmer Name", border=1)
        pdf.cell(60, 10, "Parish", border=1)
        pdf.cell(50, 10, "Coordinates", border=1)
        pdf.ln()
        
        pdf.set_font("Arial", size=9)
        members = dataframe[dataframe['Cluster_Label'] == cluster]
        for _, row in members.iterrows():
            pdf.cell(80, 10, str(row['Name'])[:35], border=1)
            pdf.cell(60, 10, str(row['Parish']), border=1)
            pdf.cell(50, 10, f"{row['Latitude']:.4f}, {row['Longitude']:.4f}", border=1)
            pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# Sidebar Downloads
if st.sidebar.button("🛠️ Prepare PDF Handbooks"):
    pdf_bytes = create_pdf(view_df, lead_farmer_df)
    st.sidebar.download_button("📥 Download PDF", pdf_bytes, f"DGB_Handbooks_{selected_view}.pdf", "application/pdf")

csv_leads = lead_farmer_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📍 Download Hub GPS (CSV)", csv_leads, f"Hubs_{selected_view}.csv", "text/csv")

# --- MAP VISUALIZATION ---
st.title(f"🌍 Lead-Farmer Meeting Hubs")
m = folium.Map(location=[view_df['Latitude'].mean(), view_df['Longitude'].mean()], zoom_start=9, tiles="cartodbpositron")

# Cluster Colors
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(view_df['Cluster_Label'].unique()))}

# Plot Farmers
for _, row in view_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=color_map[row['Cluster_Label']], fill=True,
        tooltip=f"Farmer: {row['Name']}<br>Cluster: {row['Cluster_Label']}"
    ).add_to(m)

# Plot Lead-Farmer Hubs (Black Badges)
for _, row in lead_farmer_df.iterrows():
    label_num = row['Cluster_Label'].split('-')[-1]
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: sans-serif; color: white; background: #000; 
            border-radius: 50%; width: 28px; height: 28px; display: flex; 
            align-items: center; justify-content: center; font-size: 8pt; 
            font-weight: bold; border: 2px solid white; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            {int(label_num)}</div>"""),
        tooltip=f"HUB: {row['Lead_Farmer']} ({row['Cluster_Label']})"
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- CLUSTER REGISTRY ---
st.subheader("Cluster Registry & Hub Identification")
st.dataframe(lead_farmer_df[['Cluster_Label', 'Lead_Farmer', 'Parish', 'Latitude', 'Longitude']], width="stretch")