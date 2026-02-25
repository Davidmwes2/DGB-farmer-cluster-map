# 🚜Farmer Logistics & Clustering Dashboard

This interactive web application is designed to manage and visualize farmer networks across 8 districts in Uganda. It uses geographic clustering (K-Means and Nearest Neighbor) to organize farmers into efficient groups for training, logistics, and monitoring.

## 🌟 Key Features

- **Live Data Sync:** Connects directly to the master Google Sheet for real-time registration updates.
- **Stable Clustering:** Uses a "Nearest Neighbor" logic to ensure existing farmer IDs remain static while new farmers are assigned to the closest existing group.
- **Unique ID System:** Automatically generates IDs following the `UG-DIST-SUB-001` format (e.g., `UG-HCHAM-002`).
- **Capacity Monitoring:** Highlights clusters that exceed the **60-farmer threshold** in red for split-management.
- **Field Reporting:** Generates printable PDF handbooks for Field Monitors, sorted by cluster.
- **Global Search:** Find any farmer by name and instantly zoom to their location and cluster neighbors.

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Framework:** [Streamlit](https://streamlit.io/)
- **Mapping:** [Folium](https://python-visualization.github.io/folium/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (K-Means)
- **PDF Generation:** [FPDF](http://pyfpdf.github.io/fpdf2/)

## 🚀 Deployment Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/dgb-farmer-logistics.git](https://github.com/YOUR_USERNAME/dgb-farmer-logistics.git)

   Install Dependencies:

Bash
pip install -r requirements.txt
Run Locally:

Bash
streamlit run app_final.py
Web Deployment: Connect this GitHub repo to Streamlit Community Cloud for free hosting.

📊 Data Structure Requirements
The dashboard expects a CSV input (via Google Sheets) with the following headers:

Name: Full name of the farmer.

District_2: Primary district location.

Subcounty_2: Subcounty location.

Parish_2: Parish location.

Latitude / Longitude: Decimal GPS coordinates.

🛡️ Maintenance & Alerts
Red Dots on Map: Indicate clusters that have grown too large (>60 farmers) and need to be split.

Data Health Sidebar: Will notify you if farmers are registered with missing or invalid GPS coordinates.

Developed for the DGB Project - 2026
