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

# 2. Robust Live Data Loading
@st.cache_data(ttl=600)
def load_live_data():
    sheet_id = "15V_IFm4Q14tMsg-vUV7r7IhtiGzNSjvBC58QkX9ydEY"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        response = requests.get(sheet_url)
        if response.status_code != 200:
            st.error(f"Google Sheets Access Denied (Error {response.status_code}). Check Sharing settings.")
            st.stop()
        
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={'District_2': 'District', 'Subcounty_2': 'Subcounty', 'Parish_2': 'Parish'})
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        return df.dropna(subset=['Latitude', 'Longitude'])
    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

df = load_live_data()

# 3. Stable Clustering Logic
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

# --- SIDEBAR & PDF LOGIC ---
st.sidebar.title("📊 Field Reports")
selected_view = st.sidebar.selectbox("Filter District", ["All"] + sorted(full_data['District'].unique().tolist()))
view_df = full_data if selected_view == "All" else full_data[full_data['District'] == selected_view]

def create_pdf(dataframe):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    clusters = sorted(dataframe['Cluster_Label'].unique())
    
    for cluster in clusters:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"DGB Field List: {cluster}", ln=True, align='C')
        pdf.ln(5)
        
        # Table Header
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(85, 10, "Farmer Name", border=1, fill=True)
        pdf.cell(60, 10, "Parish", border=1, fill=True)
        pdf.cell(45, 10, "Coordinates", border=1, fill=True)
        pdf.ln()
        
        # Farmer Rows
        pdf.set_font("Arial", size=9)
        cluster_farmers = dataframe[dataframe['Cluster_Label'] == cluster]
        for _, row in cluster_farmers.iterrows():
            pdf.cell(85, 10, str(row['Name'])[:40], border=1)
            pdf.cell(60, 10, str(row['Parish']), border=1)
            pdf.cell(45, 10, f"{row['Latitude']:.4f}, {row['Longitude']:.4f}", border=1)
            pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

if st.sidebar.button("🛠️ Prepare PDF Handbook"):
    pdf_output = create_pdf(view_df)
    st.sidebar.download_button(
        label="📥 Download PDF",
        data=pdf_output,
        file_name=f"Field_Report_{selected_view}.pdf",
        mime="application/pdf"
    )

# --- MAP VISUALIZATION ---
st.title(f"🌍 Farmer Network Hub")
m = folium.Map(location=[view_df['Latitude'].mean(), view_df['Longitude'].mean()], zoom_start=9, tiles="cartodbpositron")

# Color Logic
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6']
color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(view_df['Cluster_Label'].unique()))}

for _, row in view_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=color_map[row['Cluster_Label']], fill=True,
        tooltip=f"<b>{row['Name']}</b><br>ID: {row['Cluster_Label']}"
    ).add_to(m)

# Meeting Point Badges
centroids = view_df.groupby('Cluster_Label').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
for _, row in centroids.iterrows():
    label_num = row['Cluster_Label'].split('-')[-1]
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(html=f"""
            <div style="font-family: sans-serif; color: white; background: #000; 
            border-radius: 50%; width: 28px; height: 28px; display: flex; 
            align-items: center; justify-content: center; font-size: 8pt; 
            font-weight: bold; border: 2px solid white;">{int(label_num)}</div>"""),
        tooltip=f"MEETING POINT: {row['Cluster_Label']}"
    ).add_to(m)

st_folium(m, width=1300, height=600)

# --- REGISTRY TABLE ---
st.subheader("Cluster Registry Summary")
summary = full_data.groupby(['District', 'Cluster_Label']).size().reset_index(name='Farmer Count')
st.dataframe(summary, width="stretch")