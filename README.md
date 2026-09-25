# RoadGuard AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-100%25-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-FF6B6B?style=for-the-badge" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Status-Active-00C853?style=for-the-badge" alt="Status" />
</p>

<p align="center">
  <strong>AI-Powered Road Accident Live CCTV Application</strong>
</p>

<p align="center">
  A real-time intelligent traffic monitoring and emergency response dashboard designed to detect road accidents from CCTV feeds, classify severity, and support dispatch operations.
</p>

---

## 🚨 Project Highlights

- Real-time accident detection from live camera and recorded video streams
- YOLOv8-based vehicle accident recognition
- Severity classification: Minor, Substantial, Critical
- Near-miss motion analysis using optical flow heuristics
- Automated alerting through SMS, WhatsApp, email, and webhook systems
- SQLite-powered incident and dispatch logging
- Streamlit dashboard with multiple operational tabs

---

## 🧭 Overview

RoadGuard AI is a smart road safety command system built for continuous vehicle traffic analysis. It monitors urban road conditions, identifies possible accident events, classifies risk severity, and promotes quick emergency coordination using automated alerts.

This repository contains a full demo-style application that mimics a traffic command center and provides:

- live CCTV monitoring interfaces
- model-based accident inference
- severity analysis
- historical trend analytics
- GIS-style hotspot mapping
- emergency dispatch reporting

---

## 🏗️ System Architecture

```text
CCTV / Video Input
        ↓
YOLOv8 Accident Detection Model
        ↓
Severity Classification Model
        ↓
Traffic Analysis + Near-Miss Logic
        ↓
SQLite Incident Database
        ↓
Streamlit Command Dashboard
        ↓
Alerts / Dispatch / Reporting
```

---

## ✨ Core Features

| Feature | Description |
|---|---|
| Accident Detection | Detects accident events in live or recorded footage |
| Severity Analysis | Identifies whether an incident is Minor, Substantial, or Critical |
| Optical Flow Monitoring | Tracks abnormal motion and near-miss risk indicators |
| Emergency Alerts | Logs and triggers SMS, WhatsApp, email, and webhook dispatch actions |
| Incident Dashboard | Displays vehicle flow, trends, alerts, and operational status |
| Reporting | Generates printable incident and report exports |
| Data Storage | Uses SQLite to store history and dispatch records |

---

## 📁 Repository Structure

```text
AI_PoweredRoadAccidentLiveCCTVApplication/
├── app.py                          # Main Streamlit application
├── db_manager.py                  # Database utilities and incident storage
├── README.md                      # Project documentation
├── roadguard.db                   # SQLite database
├── roadguard_history.json         # Historical incident records
├── roadguard_dispatch_logs.json   # Dispatch event logs
├── yolov8n.pt                     # Base YOLOV8 weights
├── CNN_Models/                    # CNN model assets
├── YOLOv8_Accident_Model/         # Accident detection model folder
├── YOLOv8_Severity_Model/         # Severity classifier model folder
├── videos/                        # Demo traffic videos
├── .gitignore                     # Git ignore rules
├── .gitattributes                 # Git attributes
├── AI-Powered Road Accident Detection.code-workspace
└── LICENSE                       # Optional project license (if added later)
```

---

## 🧪 Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- Streamlit
- Plotly
- Pandas
- NumPy
- Pillow
- SQLite

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hoor-Fayaz/AI_PoweredRoadAccidentLiveCCTVApplication.git
cd AI_PoweredRoadAccidentLiveCCTVApplication
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows:

```bash
.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install ultralytics opencv-python pandas numpy pillow plotly streamlit
```

---

## 🧠 Model Requirements

The app expects trained YOLOv8 model files inside the project directories:

```text
YOLOv8_Accident_Model/best.pt
YOLOv8_Severity_Model/best.pt
```

If these models are missing, the application may show errors or fail to run inference.

---

## ▶️ Run the Application

From the repository root, run:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 🎛️ Application Workflow

1. Open the Streamlit app
2. Select a CCTV feed or upload a video/image file
3. Tune the detection sensitivity
4. Start the inference scan
5. View accident classification and severity results
6. Review dispatch logs, trends, and reports

---

## 🚑 Alerting Capabilities

The application can simulate dispatch workflows through:

- SMS alerts
- WhatsApp notifications
- Email dispatch instructions
- CAD-style webhook payloads

These systems are useful for emergency response planning and road traffic monitoring demonstrations.

---

## 📊 Dashboard Tabs

The app includes several operational tabs such as:

- Executive Landing Page
- CCTV Command Station
- Near-Miss Prediction
- Spatial & Trend Analytics
- Emergency Dispatch Hub
- Model Performance Curves
- Calibration Settings

---

## 📝 Notes

This project is designed for academic, research, and demonstration purposes. For production usage, it is recommended to:

- secure credentials and API keys
- validate the model on real-world traffic datasets
- deploy with robust monitoring and logging
- add proper access control and infrastructure protections

---

## 👤 Author

Hoor Fayaz

---

## 🔗 Repository

https://github.com/Hoor-Fayaz/AI_PoweredRoadAccidentLiveCCTVApplication

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python" />
</p>
