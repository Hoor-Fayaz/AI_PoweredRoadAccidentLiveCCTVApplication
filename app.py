import os
import tempfile
import numpy as np
import cv2
import pandas as pd
import datetime
import uuid
import json
import urllib.request
import urllib.parse
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(DATA_DIR, "roadguard_history.json")
DISPATCH_LOG_FILE = os.path.join(DATA_DIR, "roadguard_dispatch_logs.json")

RESPONDER_STATIONS = [
    {
        "name": "Station Alpha - Central Traffic HQ",
        "lat": 40.1218,
        "lon": -88.2290,
        "unit": "Police / EMS"
    },
    {
        "name": "Station Beta - Riverfront Medical Dispatch",
        "lat": 40.1132,
        "lon": -88.2384,
        "unit": "Hospital / Ambulance"
    },
    {
        "name": "Station Gamma - Metro Fire Command",
        "lat": 40.1090,
        "lon": -88.2265,
        "unit": "Fire Rescue"
    }
]

# ----------------------------------------------------
# Routing helpers
# ----------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return kilometers between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def find_nearest_responder(lat, lon, responders=RESPONDER_STATIONS):
    best = None
    best_distance = None
    for resp in responders:
        dist = haversine_distance(lat, lon, resp["lat"], resp["lon"])
        if dist is None:
            continue
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best = {**resp, "distance_km": dist}
    return best


def format_eta(distance_km, speed_kmh=35.0):
    if distance_km is None:
        return "N/A"
    eta_minutes = (distance_km / speed_kmh) * 60
    return f"{max(1, int(round(eta_minutes)))} min"

# ----------------------------------------------------
# Data persistence helpers
# ----------------------------------------------------

def load_saved_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_json(HISTORY_FILE)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            return df
        except Exception:
            return None
    return None


def save_history(df):
    try:
        df.to_json(HISTORY_FILE, orient="records", date_format="iso", force_ascii=False)
        return True
    except Exception:
        return False


def load_saved_dispatch_logs():
    if os.path.exists(DISPATCH_LOG_FILE):
        try:
            logs = pd.read_json(DISPATCH_LOG_FILE)
            return logs.to_dict(orient="records")
        except Exception:
            return None
    return None


def save_dispatch_logs(logs):
    try:
        pd.DataFrame(logs).to_json(DISPATCH_LOG_FILE, orient="records", date_format="iso", force_ascii=False)
        return True
    except Exception:
        return False


def get_demo_records(now: datetime.datetime):
    coords = {
        "Camera 01 - Expressway Loop (Exit 9)": (40.1241, -88.2300),
        "Camera 02 - Downtown Crossing (Main St)": (40.1108, -88.2339),
        "Camera 03 - Industrial Bypass Tunnel": (40.1052, -88.2476),
        "Camera 04 - Underpass Overpass Junction": (40.1179, -88.2211)
    }

    records = [
        {
            "id": "RA-98402",
            "timestamp": (now - datetime.timedelta(days=6, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Vehicle Collision",
            "severity": "High",
            "confidence": 0.984,
            "camera_id": "Camera 01 - Expressway Loop (Exit 9)",
            "location": "Expressway Loop - Mile 42.5",
            "dispatch_status": "Resolved (EMS & Patrol dispatched)"
        },
        {
            "id": "RA-98389",
            "timestamp": (now - datetime.timedelta(days=6, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.941,
            "camera_id": "Camera 02 - Downtown Crossing (Main St)",
            "location": "Main St & 4th Ave Intersection",
            "dispatch_status": "None"
        },
        {
            "id": "RA-98311",
            "timestamp": (now - datetime.timedelta(days=5, hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Rollover",
            "severity": "Critical",
            "confidence": 0.967,
            "camera_id": "Camera 03 - Industrial Bypass Tunnel",
            "location": "Industrial Bypass - South Entrance",
            "dispatch_status": "Resolved (Heavy Rescue dispatched)"
        },
        {
            "id": "RA-98242",
            "timestamp": (now - datetime.timedelta(days=4, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Vehicle Collision",
            "severity": "Critical",
            "confidence": 0.912,
            "camera_id": "Camera 04 - Underpass Overpass Junction",
            "location": "Route 9 Overpass Junction",
            "dispatch_status": "Resolved (Police, EMS & Fire dispatched)"
        },
        {
            "id": "RA-98201",
            "timestamp": (now - datetime.timedelta(days=4, hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.992,
            "camera_id": "Camera 01 - Expressway Loop (Exit 9)",
            "location": "Expressway Loop - Mile 42.5",
            "dispatch_status": "None"
        },
        {
            "id": "RA-98188",
            "timestamp": (now - datetime.timedelta(days=3, hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.884,
            "camera_id": "Camera 02 - Downtown Crossing (Main St)",
            "location": "Main St & 4th Ave Intersection",
            "dispatch_status": "None"
        },
        {
            "id": "RA-98105",
            "timestamp": (now - datetime.timedelta(days=2, hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Vehicle Collision",
            "severity": "Critical",
            "confidence": 0.953,
            "camera_id": "Camera 03 - Industrial Bypass Tunnel",
            "location": "Industrial Bypass - South Entrance",
            "dispatch_status": "Resolved (Patrol dispatched)"
        },
        {
            "id": "RA-98074",
            "timestamp": (now - datetime.timedelta(days=2, hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Vehicle Collision",
            "severity": "Substantial",
            "confidence": 0.725,
            "camera_id": "Camera 01 - Expressway Loop (Exit 9)",
            "location": "Expressway Loop - Mile 42.5",
            "dispatch_status": "False Alarm (Cancelled)"
        },
        {
            "id": "RA-98012",
            "timestamp": (now - datetime.timedelta(days=1, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.911,
            "camera_id": "Camera 04 - Underpass Overpass Junction",
            "location": "Route 9 Overpass Junction",
            "dispatch_status": "None"
        },
        {
            "id": "RA-97992",
            "timestamp": (now - datetime.timedelta(days=1, hours=18)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.965,
            "camera_id": "Camera 02 - Downtown Crossing (Main St)",
            "location": "Main St & 4th Ave Intersection",
            "dispatch_status": "None"
        },
        {
            "id": "RA-97940",
            "timestamp": (now - datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Accident",
            "type": "Vehicle Collision",
            "severity": "Substantial",
            "confidence": 0.924,
            "camera_id": "Camera 01 - Expressway Loop (Exit 9)",
            "location": "Expressway Loop - Mile 42.5",
            "dispatch_status": "Resolved (Patrol dispatched)"
        },
        {
            "id": "RA-97911",
            "timestamp": (now - datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "classification": "Normal",
            "type": "None",
            "severity": "None",
            "confidence": 0.985,
            "camera_id": "Camera 03 - Industrial Bypass Tunnel",
            "location": "Industrial Bypass - South Entrance",
            "dispatch_status": "None"
        }
    ]

    for rec in records:
        rec["lat"], rec["lon"] = coords.get(rec["camera_id"], (None, None))

    return records


def resolve_camera_coords(camera_id):
    camera = st.session_state.get("camera_states", {}).get(camera_id)
    if camera is None:
        return None, None
    return camera.get("lat"), camera.get("lon")

# ----------------------------------------------------
# Emergency Communication Manager Utility
# ----------------------------------------------------
class EmergencyCommManager:
    @staticmethod
    def send_sms(to_number, message, config):
        """Sends an SMS using Twilio HTTP POST endpoint (using urllib.request)."""
        sid = config.get("twilio_sid")
        token = config.get("twilio_token")
        from_num = config.get("twilio_from")
        if not (sid and token and from_num and to_number):
            return False, "Twilio credentials or destination number missing. In Simulation Mode."
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        data = {
            "To": to_number,
            "From": from_num,
            "Body": message
        }
        
        try:
            encoded_data = urllib.parse.urlencode(data).encode("utf-8")
            req = urllib.request.Request(url, data=encoded_data, method="POST")
            auth_str = f"{sid}:{token}"
            auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            req.add_header("Authorization", f"Basic {auth_b64}")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            
            with urllib.request.urlopen(req) as response:
                if response.status in [200, 201]:
                    return True, "SMS alert successfully sent via Twilio API."
                else:
                    return False, f"SMS failed (Status: {response.status})"
        except Exception as e:
            return False, f"Twilio SMS Error: {str(e)}"

    @staticmethod
    def prepare_whatsapp_link(to_number, message):
        """Build a free WhatsApp Web click-to-chat URL."""
        cleaned_number = "".join([ch for ch in to_number if ch.isdigit()])
        if cleaned_number.startswith("00"):
            cleaned_number = cleaned_number[2:]
        elif cleaned_number.startswith("0"):
            cleaned_number = "92" + cleaned_number[1:]
        encoded_text = urllib.parse.quote(message)
        return f"https://wa.me/{cleaned_number}?text={encoded_text}"

    @staticmethod
    def send_whatsapp(to_number, message, config):
        """Sends a WhatsApp message via Twilio API if configured, otherwise returns a free manual chat URL."""
        sid = config.get("twilio_sid")
        token = config.get("twilio_token")
        from_num = config.get("whatsapp_from")
        if sid and token and from_num:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            data = {
                "To": f"whatsapp:{to_number}",
                "From": f"whatsapp:{from_num}",
                "Body": message
            }

            try:
                encoded_data = urllib.parse.urlencode(data).encode("utf-8")
                req = urllib.request.Request(url, data=encoded_data, method="POST")
                auth_str = f"{sid}:{token}"
                auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
                req.add_header("Authorization", f"Basic {auth_b64}")
                req.add_header("Content-Type", "application/x-www-form-urlencoded")

                with urllib.request.urlopen(req) as response:
                    if response.status in [200, 201]:
                        return True, "WhatsApp alert successfully sent via Twilio API."
                    else:
                        return False, f"WhatsApp failed (Status: {response.status})"
            except Exception as e:
                return False, f"Twilio WhatsApp Error: {str(e)}"

        # Fallback to free WhatsApp Web click-to-chat
        manual_url = EmergencyCommManager.prepare_whatsapp_link(to_number, message)
        return True, f"Manual WhatsApp alert ready: {manual_url}"

    @staticmethod
    def send_email(to_email, subject, body, config):
        """Sends an HTML email using standard SMTP."""
        smtp_server = config.get("smtp_server")
        smtp_port = config.get("smtp_port", 587)
        smtp_user = config.get("smtp_user")
        smtp_pass = config.get("smtp_pass")
        from_email = config.get("smtp_from") or smtp_user
        
        if not (smtp_server and smtp_user and smtp_pass and to_email):
            return False, "SMTP server credentials or recipient email missing. In Simulation Mode."
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_email
            msg["To"] = to_email
            
            html_part = MIMEText(body, "html")
            msg.attach(html_part)
            
            server = smtplib.SMTP(smtp_server, int(smtp_port))
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_email, to_email, msg.as_string())
            server.quit()
            return True, "Email dispatch successfully sent via SMTP."
        except Exception as e:
            return False, f"SMTP Email Error: {str(e)}"

    @staticmethod
    def trigger_webhook(webhook_url, payload):
        """Triggers a POST webhook containing accident telemetry payload."""
        if not webhook_url:
            return False, "Webhook endpoint URL missing. In Simulation Mode."
        
        try:
            encoded_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(webhook_url, data=encoded_data, method="POST")
            req.add_header("Content-Type", "application/json")
            
            with urllib.request.urlopen(req) as response:
                if response.status in [200, 201, 204]:
                    return True, "Municipal Webhook successfully triggered."
                else:
                    return False, f"Webhook trigger failed (Status: {response.status})"
        except Exception as e:
            return False, f"Webhook Error: {str(e)}"
# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI - Real-Time Accident Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# Model Path & Definition  (YOLOv8)
# ----------------------------------------------------
MODEL_PATH = r"d:\Hoor\University stuff\6th Semester\Machine Learning Fundamentals\AI-Powered Road Accident Detection\YOLOv8_Accident_Model\best.pt"
SEVERITY_MODEL_PATH = r"d:\Hoor\University stuff\6th Semester\Machine Learning Fundamentals\AI-Powered Road Accident Detection\YOLOv8_Severity_Model\best.pt"

@st.cache_resource
def load_road_accident_model():
    """
    Loads the YOLOv8 model from the .pt weights file.
    """
    path = MODEL_PATH
    if not os.path.exists(path):
        path = r"c:\Users\hp\Downloads\YOLOv8_Accident_Model\best.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLOv8 accident model file not found at: {path}")
    return YOLO(path)

@st.cache_resource
def load_severity_model():
    """
    Loads the YOLOv8 severity classification model from the .pt weights file.
    """
    path = SEVERITY_MODEL_PATH
    if not os.path.exists(path):
        path = r"c:\Users\hp\Downloads\YOLOv8_Severity_Model\best.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLOv8 severity model file not found at: {path}")
    return YOLO(path)

# Load the models
try:
    model = load_road_accident_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

try:
    severity_model = load_severity_model()
    severity_model_loaded = True
except Exception as e:
    severity_model_loaded = False
    severity_model_error = str(e)

# ----------------------------------------------------
# YOLOv8 Inference & Annotation
# ----------------------------------------------------

def process_image(img, model, threshold=0.5, severity_model=None):
    """Run YOLOv8 inference (detection or classification) on a BGR image.
    Returns annotated image, heatmap overlay (unused), classification label,
    confidence score, severity label, and bounding box coordinates (None for classification).
    """
    _SEVERITY_MAP = {"1": "Minor", "2": "Substantial", "3": "Critical",
                     0: "Minor", 1: "Substantial", 2: "Critical"}

    h, w = img.shape[:2]
    annotated_img = img.copy()
    heatmap_overlay = None
    box_coords = None
    classification = "Normal"
    confidence = 0.0
    severity = "None"

    # Run YOLOv8 inference
    results = model.predict(img, conf=threshold, verbose=False)
    result = results[0]

    # Check if the model is a classification model
    is_classification_model = getattr(model, "task", "detect") == "classify" or (
        hasattr(result, "probs") and result.probs is not None
    )

    if is_classification_model:
        # Classification model logic (e.g. yolov8-cls)
        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs.data.tolist()
            
            # Find the probability of 'Accident' (class 0)
            accident_conf = 0.0
            if model.names:
                for idx, name in model.names.items():
                    if name.lower() == "accident":
                        accident_conf = probs[idx]
                        break
            else:
                accident_conf = probs[0] if len(probs) > 0 else 0.0

            # If the probability of Accident is greater than the threshold
            if accident_conf >= threshold:
                classification = "Accident"
                confidence = accident_conf
            else:
                classification = "Normal"
                confidence = 1.0 - accident_conf
            
            # Draw a banner on the image
            banner_height = max(35, int(h * 0.12))
            banner_color = (0, 0, 255) if classification == "Accident" else (0, 200, 100)
            cv2.rectangle(annotated_img, (0, 0), (w, banner_height), banner_color, -1)
            label = f"{classification.upper()} ({confidence:.1%})"
            cv2.putText(annotated_img, label, (15, int(banner_height * 0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            
    else:
        # Object detection model logic (e.g. yolov8-det)
        best_conf = 0.0
        best_box = None
        best_cls_name = "Normal"
        detected_accident = False

        if result.boxes is not None and len(result.boxes) > 0:
            # Separate accident and non-accident detections
            accident_detections = []
            non_accident_detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id] if model.names else str(cls_id)
                if cls_name.lower() == "accident" or cls_id == 0:
                    accident_detections.append((conf, cls_name, box.xyxy[0].cpu().numpy().astype(int)))
                else:
                    non_accident_detections.append((conf, cls_name, box.xyxy[0].cpu().numpy().astype(int)))
            
            # Prioritize Accident detections if any exist
            if accident_detections:
                accident_detections.sort(key=lambda x: x[0], reverse=True)
                best_conf, best_cls_name, best_box = accident_detections[0]
                detected_accident = True
            elif non_accident_detections:
                non_accident_detections.sort(key=lambda x: x[0], reverse=True)
                best_conf, best_cls_name, best_box = non_accident_detections[0]
                detected_accident = False

        if best_box is not None:
            if detected_accident:
                classification = "Accident"
                confidence = best_conf
                x1, y1, x2, y2 = best_box
                box_coords = (x1, y1, x2 - x1, y2 - y1)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                label = f"{best_cls_name.upper()} {confidence:.1%}"
                badge_w = max(200, len(label) * 11)
                cv2.rectangle(annotated_img, (x1, max(y1 - 36, 0)), (x1 + badge_w, max(y1, 36)), (0, 0, 220), -1)
                cv2.putText(annotated_img, label, (x1 + 5, max(y1 - 10, 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                classification = "Normal"
                confidence = best_conf
                x1, y1, x2, y2 = best_box
                box_coords = (x1, y1, x2 - x1, y2 - y1)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 200, 100), 4)
                label = f"NORMAL {confidence:.1%}"
                badge_w = max(200, len(label) * 11)
                cv2.rectangle(annotated_img, (x1, max(y1 - 36, 0)), (x1 + badge_w, max(y1, 36)), (0, 180, 80), -1)
                cv2.putText(annotated_img, label, (x1 + 5, max(y1 - 10, 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            classification = "No Detection"
            confidence = 0.0
            cv2.rectangle(annotated_img, (20, 20), (260, 62), (0, 200, 100), 2)
            cv2.putText(annotated_img, f"NO DETECTION {confidence:.1%}", (30, 49),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 20, 40), 2, cv2.LINE_AA)

    # Run severity model if accident detected and model is available
    if classification == "Accident" and severity_model is not None:
        try:
            sev_results = severity_model.predict(img, verbose=False)
            sev_result = sev_results[0]
            if hasattr(sev_result, "probs") and sev_result.probs is not None:
                sev_class_id = int(sev_result.probs.top1)
                sev_class_name = severity_model.names.get(sev_class_id, str(sev_class_id))
                severity = _SEVERITY_MAP.get(sev_class_name, _SEVERITY_MAP.get(sev_class_id, "Unknown"))
            else:
                severity = "Unknown"
        except Exception:
            severity = "Unknown"

    return annotated_img, heatmap_overlay, classification, confidence, severity, box_coords
# ----------------------------------------------------
# Session Database (Mock Data Initialization)
# ----------------------------------------------------
if "dispatch_config" not in st.session_state:
    st.session_state["dispatch_config"] = {
        "twilio_sid": "",
        "twilio_token": "",
        "twilio_from": "",
        "whatsapp_from": "",
        "whatsapp_to": "03141988998",
        "police_phone": "+15550199",
        "hospital_email": "emergency@traumacenter.org",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_pass": "",
        "smtp_from": "",
        "webhook_url": ""
    }

if "dispatch_logs" not in st.session_state:
    saved_logs = load_saved_dispatch_logs()
    st.session_state["dispatch_logs"] = saved_logs if saved_logs is not None else []

if "camera_states" not in st.session_state:
    st.session_state["camera_states"] = {
        "Camera 01 - Expressway Loop (Exit 9)": {
            "status": "ONLINE", "location": "Expressway Loop - Mile 42.5", "alert_active": False,
            "lat": 40.1241, "lon": -88.2300
        },
        "Camera 02 - Downtown Crossing (Main St)": {
            "status": "ONLINE", "location": "Main St & 4th Ave Intersection", "alert_active": False,
            "lat": 40.1108, "lon": -88.2339
        },
        "Camera 03 - Industrial Bypass Tunnel": {
            "status": "ONLINE", "location": "Industrial Bypass - South Entrance", "alert_active": False,
            "lat": 40.1052, "lon": -88.2476
        },
        "Camera 04 - Underpass Overpass Junction": {
            "status": "ONLINE", "location": "Route 9 Overpass Junction", "alert_active": False,
            "lat": 40.1179, "lon": -88.2211
        }
    }

if "history" not in st.session_state:
    saved_history = load_saved_history()
    if saved_history is not None and len(saved_history) > 0:
        st.session_state["history"] = saved_history
    else:
        now = datetime.datetime.now()
        st.session_state["history"] = pd.DataFrame(get_demo_records(now))

# Theme System
# ----------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def get_theme_colors(theme: str) -> dict:
    if theme == "light":
        return {
            "bg": "#F8FAFC",  # slate 50
            "bg_gradient": "linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)",  # Slate 50 to 100
            "surface": "rgba(255, 255, 255, 0.98)",
            "surface_alt": "#FFFFFF",
            "border": "#E2E8F0",  # slate 200
            "border_subtle": "#F1F5F9",  # slate 100
            "text": "#000000",  # pure black for maximum visibility
            "text_muted": "#334155",  # slate 700 for high contrast in light mode
            "accent": "#4F46E5",  # Indigo 600
            "accent_light": "#6366F1",  # Indigo 500
            "accent_glow": "rgba(79, 70, 229, 0.15)",
            "header_bg": "#FFFFFF",
            "success": "#10B981",  # Emerald 500
            "danger": "#EF4444",  # Red 500
            "warning": "#F59E0B",  # Amber 500
            "chart_line": "#FFFFFF",
            "shadow": "0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05)",
            "shadow_hover": "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
            "input_bg": "#F8FAFC",
            "code_bg": "#EEF2FF",  # Indigo 50 — visible code bg for light mode
            "scrollbar_track": "#F1F5F9",
            "scrollbar_thumb": "#CBD5E1",
            "primary_btn_text": "#FFFFFF",
        }
    # Dark Theme
    return {
        "bg": "#090D16",  # Very deep slate/black
        "bg_gradient": "radial-gradient(circle at 10% 20%, rgba(26, 32, 53, 0.9) 0%, rgba(9, 13, 22, 1) 90%)",
        "surface": "rgba(17, 24, 39, 0.7)",  # Gray 900 with transparency
        "surface_alt": "rgba(31, 41, 55, 0.5)",  # Gray 800 with transparency
        "border": "rgba(59, 130, 246, 0.2)",  # Translucent blue border
        "border_subtle": "rgba(59, 130, 246, 0.08)",
        "text": "#F8FAFC",
        "text_muted": "#94A3B8",
        "accent": "#3B82F6",  # Blue 500
        "accent_light": "#60A5FA",  # Blue 400
        "accent_glow": "rgba(59, 130, 246, 0.25)",
        "header_bg": "rgba(17, 24, 39, 0.9)",
        "success": "#10B981",
        "danger": "#EF4444",
        "warning": "#F59E0B",
        "chart_line": "#090D16",
        "shadow": "0 4px 20px rgba(0, 0, 0, 0.4)",
        "shadow_hover": "0 10px 30px rgba(0, 0, 0, 0.6)",
        "input_bg": "rgba(17, 24, 39, 0.8)",
        "code_bg": "rgba(31, 41, 55, 0.8)",  # Dark code block bg
        "scrollbar_track": "#090D16",
        "scrollbar_thumb": "rgba(59, 130, 246, 0.4)",
        "primary_btn_text": "#FFFFFF",
    }

def plotly_theme_layout(theme: str, height: int = 300, **overrides) -> dict:
    c = get_theme_colors(theme)
    xaxis_defaults = {
        "color": c["text"],
        "title_font": {"color": c["text"]},
        "tickfont": {"color": c["text"]},
    }
    yaxis_defaults = {
        "color": c["text"],
        "title_font": {"color": c["text"]},
        "tickfont": {"color": c["text"]},
    }
    layout = {
        "template": "plotly_white" if theme == "light" else "plotly_dark",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": c["text"]},
        "font_color": c["text"],
        "xaxis": xaxis_defaults,
        "yaxis": yaxis_defaults,
        "legend": {"font": {"color": c["text"]}},
        "margin": dict(l=20, r=20, t=10, b=10),
        "height": height,
    }
    for key, value in overrides.items():
        if key in ("xaxis", "yaxis", "legend") and isinstance(value, dict):
            nested = layout.get(key, {})
            nested.update(value)
            layout[key] = nested
        else:
            layout[key] = value
    return layout

def inject_app_styles(theme: str) -> None:
    c = get_theme_colors(theme)
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp {{
        background: {c["bg_gradient"]} !important;
        background-color: {c["bg"]} !important;
        color: {c["text"]} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px !important;
    }}

    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {c["scrollbar_track"]}; }}
    ::-webkit-scrollbar-thumb {{
        background: {c["scrollbar_thumb"]};
        border-radius: 99px;
    }}

    /* ---- ALL TEXT ELEMENTS — force correct color regardless of Streamlit's own CSS ---- */
    .stApp p, .stApp span:not([class*="emoji"]),
    .stApp li, .stApp strong, .stApp em,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {c["text"]} !important;
    }}

    /* Markdown & st.write() */
    .stMarkdown p, .stMarkdown span, .stMarkdown li,
    .stMarkdown strong, .stMarkdown em,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {{
        color: {c["text"]} !important;
    }}
    .stDataFrame td, .stDataFrame th,
    .stTable td, .stTable th,
    .stDataFrame caption {{
        color: {c["text"]} !important;
    }}
    /* code inline blocks */
    .stMarkdown code, [data-testid="stMarkdownContainer"] code {{
        background: {c["code_bg"]} !important;
        color: {c["accent"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 4px;
        padding: 1px 5px;
    }}

    /* Caption / small text */
    .stCaption, [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        color: {c["text_muted"]} !important;
    }}

    /* st.write() / generic text */
    [data-testid="stText"] {{
        color: {c["text"]} !important;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: {c["text_muted"]} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 18px !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s ease !important;
        border-bottom: 2px solid transparent !important;
    }}
    button[data-baseweb="tab"]:hover {{ color: {c["accent"]} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {c["accent"]} !important;
        border-bottom: 2px solid {c["accent"]} !important;
        background-color: {c["border_subtle"]} !important;
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: {c["accent"]} !important; }}
    div[role="tablist"] {{
        border-bottom: 1px solid {c["border"]} !important;
        padding-bottom: 1px !important;
        margin-bottom: 24px !important;
        gap: 6px !important;
    }}

    /* ---- RADIO BUTTONS ---- */
    .stRadio label p, .stRadio label span, .stRadio label div {{
        color: {c["text"]} !important;
    }}
    .stRadio [data-baseweb="radio"] ~ div,
    .stRadio [data-baseweb="radio"] + span {{
        color: {c["text"]} !important;
    }}
    div[role="radiogroup"] label div p {{
        color: {c["text"]} !important;
        font-weight: 500 !important;
    }}
    /* Radio widget label (the heading above options) */
    .stRadio > div > label p {{
        color: {c["text_muted"]} !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.3px;
    }}

    /* ---- ALL WIDGET LABELS ---- */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stSelectbox > label,
    .stTextInput > label,
    .stSlider > label,
    .stNumberInput > label,
    .stFileUploader > label,
    .stRadio > label {{
        color: {c["text_muted"]} !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
    }}

    /* ---- SELECTBOX ---- */
    [data-baseweb="select"] > div {{
        background-color: {c["input_bg"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="select"] span, [data-baseweb="select"] div {{
        color: {c["text"]} !important;
    }}
    /* Dropdown list items */
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background-color: {c["surface_alt"]} !important;
        border: 1px solid {c["border"]} !important;
    }}
    [role="option"] {{
        background-color: {c["surface_alt"]} !important;
        color: {c["text"]} !important;
    }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{
        background-color: {c["border_subtle"]} !important;
        color: {c["accent"]} !important;
    }}

    /* ---- TEXT INPUT ---- */
    .stTextInput > div > div > input {{
        background-color: {c["input_bg"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
        caret-color: {c["accent"]} !important;
    }}
    .stTextInput > div > div > input::placeholder {{
        color: {c["text_muted"]} !important;
        opacity: 0.7;
    }}

    /* ---- NUMBER INPUT ---- */
    .stNumberInput > div > div > input,
    [data-testid="stNumberInputField"] {{
        background-color: {c["input_bg"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
    }}

    /* ---- SLIDER ---- */
    [data-testid="stSliderThumb"] {{
        background-color: {c["accent"]} !important;
        border-color: {c["accent"]} !important;
    }}
    [data-testid="stSliderTickBar"],
    [data-testid="stThumbValue"] {{
        color: {c["text_muted"]} !important;
    }}
    /* Slider track */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background-color: {c["accent"]} !important;
    }}

    /* ---- FILE UPLOADER ---- */
    [data-testid="stFileUploader"] {{
        border: 1px dashed {c["border"]} !important;
        background-color: {c["surface_alt"]} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {c["accent"]} !important;
        box-shadow: 0 0 15px {c["accent_glow"]};
    }}
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] [role="button"] {{
        background-color: {c["accent"]} !important;
        color: {c["primary_btn_text"]} !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease, transform 0.2s ease !important;
    }}
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] [role="button"]:hover {{
        background-color: {c["accent_light"]} !important;
        color: {c["primary_btn_text"]} !important;
        transform: translateY(-1px) !important;
    }}
    [data-testid="stFileUploader"] button:focus-visible,
    [data-testid="stFileUploader"] [role="button"]:focus-visible {{
        outline: 2px solid {c["accent"]} !important;
        outline-offset: 2px !important;
    }}
    [data-testid="stFileUploader"] section {{ background-color: transparent !important; }}
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileDropzoneInstructions"] span,
    [data-testid="stFileDropzoneInstructions"] p {{
        color: {c["text_muted"]} !important;
    }}
    [data-testid="stFileUploaderFileName"] {{
        color: {c["text"]} !important;
        font-weight: 600 !important;
    }}

    /* ---- EXPANDER ---- */
    [data-testid="stExpander"] {{
        border: 1px solid {c["border"]} !important;
        border-radius: 10px !important;
        background-color: {c["surface_alt"]} !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: {c["text"]} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover p {{
        color: {c["accent"]} !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] div {{
        color: {c["text"]} !important;
    }}
    details > summary svg, details > summary span {{
        color: {c["text"]} !important;
        fill: {c["text_muted"]} !important;
    }}

    /* ---- ALERTS (st.info, st.warning, st.error, st.success) ---- */
    .stAlert {{
        border-radius: 8px !important;
        border: 1px solid {c["border"]} !important;
    }}
    .stAlert p, .stAlert span {{
        color: {c["text"]} !important;
    }}
    [data-testid="stAlertContainer"] p,
    [data-testid="stAlertContainer"] span {{
        color: {c["text"]} !important;
    }}

    /* ---- SPINNER ---- */
    [data-testid="stSpinner"] p,
    [data-testid="stSpinner"] span {{
        color: {c["text"]} !important;
    }}

    /* ---- DATAFRAME / TABLE ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {c["border"]} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] td,
    [data-testid="stTable"] th,
    .stDataFrame td,
    .stDataFrame th,
    .stTable td,
    .stTable th {{
        color: {c["text"]} !important;
        background-color: transparent !important;
    }}
    [data-testid="stDataFrameResizable"] {{
        border: 1px solid {c["border"]} !important;
    }}

    /* ---- DOWNLOAD BUTTON ---- */
    [data-testid="stDownloadButton"] button {{
        background: {c["surface_alt"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}
    [data-testid="stDownloadButton"] button:hover {{
        border-color: {c["accent"]} !important;
        color: {c["accent"]} !important;
    }}

    /* ---- BUTTONS ---- */
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {c["accent_light"]}, {c["accent"]}) !important;
        color: {c["primary_btn_text"]} !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 12px {c["accent_glow"]} !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        box-shadow: 0 6px 18px {c["accent_glow"]} !important;
        transform: translateY(-1px) !important;
    }}
    div.stButton > button[kind="secondary"] {{
        background: {c["surface_alt"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        border-color: {c["accent"]} !important;
        background: {c["border_subtle"]} !important;
        color: {c["accent"]} !important;
    }}

    /* Theme toggle button */
    div[data-testid="column"]:has(.theme-toggle-wrap) div.stButton > button {{
        background: {c["surface_alt"]} !important;
        color: {c["accent"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        padding: 6px 12px !important;
        width: auto !important;
        box-shadow: {c["shadow"]} !important;
        transition: all 0.2s ease !important;
    }}

    /* ---- PANELS ---- */
    .glass-panel {{
        background: {c["surface"]} !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {c["border"]} !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: {c["shadow"]};
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }}
    .glass-panel:hover {{
        box-shadow: {c["shadow_hover"]};
        border-color: {c["accent_glow"]};
    }}
    .glass-panel {{
        color: {c["text"]} !important;
    }}
    .glass-panel * {{
        color: inherit;
    }}
    .panel-header {{
        font-size: 12px;
        font-weight: 700;
        color: {c["accent"]} !important;
        margin-bottom: 14px;
        border-left: 3px solid {c["accent"]};
        padding-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    .section-title {{
        font-size: 14px;
        font-weight: 700;
        color: {c["accent"]} !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin: 16px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid {c["border_subtle"]};
    }}

    .kpi-card {{
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: {c["shadow_hover"]} !important;
        border-color: {c["accent"]} !important;
    }}

    .app-footer {{
        text-align: center;
        padding: 16px 0 8px 0;
        color: {c["text_muted"]};
        font-size: 11px;
        border-top: 1px solid {c["border_subtle"]};
        margin-top: 24px;
    }}

    /* ---- HR divider ---- */
    hr {{
        border-color: {c["border"]} !important;
        opacity: 0.6;
    }}

    /* Emergency Alert Ring & Glowing Effect */
    @keyframes alertGlow {{
        0%, 100% {{ border-color: rgba(239, 68, 68, 0.4); box-shadow: 0 0 10px rgba(239, 68, 68, 0.15); }}
        50% {{ border-color: rgba(239, 68, 68, 0.85); box-shadow: 0 0 20px rgba(239, 68, 68, 0.45); }}
    }}
    .alert-active-panel {{
        animation: alertGlow 1.8s infinite ease-in-out !important;
        border-left: 4px solid {c["danger"]} !important;
    }}

    @keyframes blink {{
        0%, 100% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .header-bar {{ animation: fadeIn 0.3s ease; }}
</style>
""", unsafe_allow_html=True)

inject_app_styles(st.session_state.theme)
C = get_theme_colors(st.session_state.theme)

# ----------------------------------------------------
# Top Navigation Header Bar
# ----------------------------------------------------
hdr_left, hdr_mid, hdr_right = st.columns([5.5, 3.5, 1])

with hdr_left:
    st.markdown(f"""
<div class="header-bar" style="display:flex;align-items:center;gap:14px;padding:14px 20px;
    background:{C["header_bg"]};backdrop-filter:blur(12px);border:1px solid {C["border"]};
    border-radius:14px;box-shadow:{C["shadow"]};">
    <div style="display:flex;align-items:center;justify-content:center;width:42px;height:42px;
        background:{C["border_subtle"]};border-radius:11px;border:1px solid {C["border"]};">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{C["accent"]}"
            stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        </svg>
    </div>
    <div>
        <div style="font-size:20px;font-weight:800;letter-spacing:0.5px; color: {C["text"]};">RoadGuard AI</div>
        <div style="font-size:11px;color:{C["text_muted"]};font-weight:500;margin-top:2px;
            letter-spacing:0.4px;">Real-Time Accident Detection Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

with hdr_mid:
    st.markdown(f"""
<div class="header-bar" style="display:flex;align-items:center;justify-content:center;gap:10px;
    padding:14px 20px;background:{C["header_bg"]};backdrop-filter:blur(12px);
    border:1px solid {C["border"]};border-radius:14px;box-shadow:{C["shadow"]};height:100%;">
    <span style="display:inline-block;width:8px;height:8px;background:{C["success"]};
        border-radius:50%;box-shadow:0 0 8px {C["success"]};animation:blink 1.5s infinite;"></span>
    <span style="font-size:12px;font-weight:700;color:{C["success"]};letter-spacing:0.6px;">
        SYSTEM ONLINE</span>
    <span style="color:{C["border"]};">|</span>
    <span style="font-size:12px;color:{C["text_muted"]};font-weight:500;">4 channels active</span>
</div>
""", unsafe_allow_html=True)

with hdr_right:
    st.markdown('<div class="theme-toggle-wrap"></div>', unsafe_allow_html=True)
    theme_label = "☀️" if st.session_state.theme == "dark" else "🌙"
    theme_help = "Switch to light mode" if st.session_state.theme == "dark" else "Switch to dark mode"
    if st.button(theme_label, key="theme_toggle", help=theme_help):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Helper Functions for UI Layout
# ----------------------------------------------------
def render_kpi_card(title, value, icon_svg, icon_bg, trend_text, trend_color, trend_icon):
    """Renders a glassmorphic KPI dashboard card."""
    c = get_theme_colors(st.session_state.theme)
    card_html = f"""
    <div class="kpi-card" style="background:{c["surface"]};backdrop-filter:blur(12px);
        border:1px solid {c["border"]};border-radius:12px;padding:16px;text-align:left;
        box-shadow:{c["shadow"]};">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:10px;color:{c["text_muted"]};font-weight:700;
                text-transform:uppercase;letter-spacing:0.8px;">{title}</span>
            <div style="background:{icon_bg};padding:6px;border-radius:8px;
                display:flex;align-items:center;justify-content:center;">{icon_svg}</div>
        </div>
        <div style="font-size:24px;font-weight:800;color:{c["text"]};
            font-family:'JetBrains Mono',monospace;line-height:1.1;">{value}</div>
        <div style="font-size:10px;margin-top:8px;color:{trend_color};font-weight:600;
            display:flex;align-items:center;gap:4px;">
            <span style="font-size:9px;">{trend_icon}</span> {trend_text}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def trigger_authority_dispatch(classification, severity, type_, confidence, camera_id, location, timestamp):
    """Triggers automated SMS, Email, and Webhook alerts based on the detection."""
    cfg = st.session_state["dispatch_config"]
    session_id = f"RA-{uuid.uuid4().hex[:5].upper()}"
    dispatch_notes = []
    
    # 1. SMS Dispatch to Police
    police_phone = cfg.get("police_phone", "+15550199")
    sms_body = f"[RoadGuard AI Alert] Incident: {severity} {type_} at {location} (Camera: {camera_id}). Detected at {timestamp} with {confidence:.1%} confidence. Requesting immediate response. Log ID: {session_id}"
    
    sms_status, sms_msg = EmergencyCommManager.send_sms(police_phone, sms_body, cfg)
    status_str = "SUCCESS" if sms_status else "SIMULATED"
    st.session_state["dispatch_logs"].append({
        "timestamp": timestamp,
        "camera_id": camera_id,
        "type": "SMS Dispatch",
        "recipient": police_phone,
        "status": status_str,
        "details": sms_msg if sms_status else "Simulated SMS alert logged (Twilio credentials not set)"
    })
    dispatch_notes.append("Police notified")

    # 1b. WhatsApp Emergency Alert
    lat, lon = resolve_camera_coords(camera_id)
    map_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}" if lat is not None and lon is not None else "GPS unavailable"
    whatsapp_body = (
        f"🚨 RoadGuard AI Emergency Alert 🚨\n"
        f"Incident ID: {session_id}\n"
        f"Classification: {type_}\n"
        f"Severity: {severity}\n"
        f"Camera: {camera_id}\n"
        f"Location: {location}\n"
        f"GPS Coordinates: {lat},{lon}\n"
        f"Map: {map_link}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Detected: {timestamp}"
    )
    whatsapp_to = cfg.get("whatsapp_to", "03141988998")
    whatsapp_status, whatsapp_msg = EmergencyCommManager.send_whatsapp(whatsapp_to, whatsapp_body, cfg)
    status_str = "SUCCESS" if whatsapp_status else "SIMULATED"
    details_text = whatsapp_msg if whatsapp_status else "Simulated WhatsApp alert logged (Twilio WhatsApp credentials not set)"
    st.session_state["dispatch_logs"].append({
        "timestamp": timestamp,
        "camera_id": camera_id,
        "type": "WhatsApp Alert",
        "recipient": whatsapp_to,
        "status": status_str,
        "details": details_text
    })
    if whatsapp_status and details_text.startswith("Manual WhatsApp alert ready:"):
        link = details_text.split(": ", 1)[1]
        st.session_state["last_whatsapp_link"] = link
    dispatch_notes.append("WhatsApp notified")

    # 2. Email Dispatch to EMS / Hospital
    hospital_email = cfg.get("hospital_email", "emergency@traumacenter.org")
    subject = f"🚨 RoadGuard AI ALERT: {severity} Road Accident at {location}"
    email_html = f"""
    <html>
    <body style="font-family: sans-serif; background-color: #F8FAFC; color: #0F172A; padding: 20px;">
        <div style="background-color: #EF4444; color: white; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 20px;">
            🚨 EMERGENCY INCIDENT TELEMETRY DISPATCH 🚨
        </div>
        <div style="background: white; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px;">
            <h3>Automatic Dispatch Logs</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold; width: 35%;">Log ID</td><td style="padding: 10px;">{session_id}</td></tr>
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold;">Camera ID</td><td style="padding: 10px;">{camera_id}</td></tr>
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold;">Location GPS</td><td style="padding: 10px;">{location}</td></tr>
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold;">Incident Classification</td><td style="padding: 10px;">{type_}</td></tr>
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold;">Severity Grade</td><td style="padding: 10px; color: red; font-weight: bold;">{severity.upper()}</td></tr>
                <tr style="border-bottom: 1px solid #F1F5F9;"><td style="padding: 10px; font-weight: bold;">Detection Time</td><td style="padding: 10px;">{timestamp}</td></tr>
                <tr><td style="padding: 10px; font-weight: bold;">Model Confidence</td><td style="padding: 10px;">{confidence:.2%}</td></tr>
            </table>
            <p style="margin-top: 20px; font-size: 12px; color: #64748B;">This is an automated dispatch alert triggered by the RoadGuard AI Computer Vision highway scanner.</p>
        </div>
    </body>
    </html>
    """
    
    email_status, email_msg = EmergencyCommManager.send_email(hospital_email, subject, email_html, cfg)
    status_str = "SUCCESS" if email_status else "SIMULATED"
    st.session_state["dispatch_logs"].append({
        "timestamp": timestamp,
        "camera_id": camera_id,
        "type": "Email Dispatch",
        "recipient": hospital_email,
        "status": status_str,
        "details": email_msg if email_status else "Simulated email alert logged (SMTP credentials not set)"
    })
    dispatch_notes.append("EMS dispatched")

    # 3. Webhook Dispatch
    webhook_url = cfg.get("webhook_url", "")
    payload = {
        "event_id": session_id,
        "timestamp": timestamp,
        "camera_id": camera_id,
        "location": location,
        "classification": classification,
        "incident_type": type_,
        "severity": severity,
        "confidence": float(confidence),
        "status": "DISPATCHED"
    }
    
    webhook_status, webhook_msg = EmergencyCommManager.trigger_webhook(webhook_url, payload)
    status_str = "SUCCESS" if webhook_status else "SIMULATED"
    st.session_state["dispatch_logs"].append({
        "timestamp": timestamp,
        "camera_id": camera_id,
        "type": "Webhook Payload",
        "recipient": webhook_url if webhook_url else "Municipal Central API",
        "status": status_str,
        "details": webhook_msg if webhook_status else "Simulated API webhook trigger logged (URL not set)"
    })
    dispatch_notes.append("Municipal CAD updated")
    save_dispatch_logs(st.session_state["dispatch_logs"])
    
    # Return formatted dispatch status string for record
    return session_id, f"Dispatched ({', '.join(dispatch_notes)})"


# Define Streamlit Tabs
tab_monitor, tab_analytics, tab_dispatch, tab_config = st.tabs([
    "🎛️ Live CCTV Monitor",
    "📊 Incident Analytics",
    "🚨 Emergency Dispatch Center",
    "⚙️ System Settings"
])

# ----------------------------------------------------
# 1. LIVE CCTV MONITOR TAB
# ----------------------------------------------------
with tab_monitor:
    df = st.session_state["history"]
    
    # Calculate operational metrics
    total_scans = len(df)
    active_alerts = len(df[(df["classification"] == "Accident") & (~df["dispatch_status"].str.contains("Resolved|False", case=False))])
    critical_alerts = len(df[(df["classification"] == "Accident") & (df["severity"] == "Critical")])
    resolved_alerts = len(df[df["dispatch_status"].str.contains("Resolved", case=False)])
    
    # KPI Grid Row
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    
    with col_kpi1:
        icon_scan = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>"""
        render_kpi_card("Total Monitored Logs", str(total_scans), icon_scan, "rgba(59, 130, 246, 0.1)", "+8.4% vs yesterday", C["success"], "▲")
        
    with col_kpi2:
        icon_alert = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""
        trend_alert_color = C["danger"] if active_alerts > 0 else C["success"]
        trend_alert_lbl = f"{active_alerts} needs attention" if active_alerts > 0 else "All zones clear"
        render_kpi_card("Active Alerts", str(active_alerts), icon_alert, "rgba(239, 68, 68, 0.1)", trend_alert_lbl, trend_alert_color, "🚨" if active_alerts > 0 else "✔")
        
    with col_kpi3:
        icon_crit = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>"""
        render_kpi_card("Critical Incidents", str(critical_alerts), icon_crit, "rgba(245, 158, 11, 0.1)", "Requiring Heavy Rescue", C["warning"], "⚠")
        
    with col_kpi4:
        icon_res = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>"""
        res_rate = f"{(resolved_alerts / len(df[df['classification'] == 'Accident']) * 100):.1f}%" if len(df[df['classification'] == 'Accident']) > 0 else "100%"
        render_kpi_card("Resolution Rate", res_rate, icon_res, "rgba(16, 185, 129, 0.1)", f"{resolved_alerts} dispatches closed", C["success"], "✔")
        
    with col_kpi5:
        icon_active_cams = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15.6 11.6L22 7v10l-6.4-4.6v-3.2zM2 5h12v14H2z"></path></svg>"""
        render_kpi_card("Operational Feeds", "4 / 4", icon_active_cams, "rgba(16, 185, 129, 0.1)", "100% CCTV uptime", C["success"], "✔")

    st.markdown("<br>", unsafe_allow_html=True)

    camera_states = st.session_state.get("camera_states", {})
    camera_df = pd.DataFrame([
        {
            "Camera ID": cam_id,
            "Location": cam["location"],
            "Status": cam.get("status", "UNKNOWN"),
            "Latitude": cam.get("lat"),
            "Longitude": cam.get("lon"),
            "Alert Active": "Yes" if cam.get("alert_active", False) else "No"
        }
        for cam_id, cam in camera_states.items()
    ])

    if not camera_df.empty:
        st.markdown('<div class="glass-panel"><div class="panel-header">CCTV GPS Zone Network</div>', unsafe_allow_html=True)
        st.markdown("<div style='min-height:300px;'>", unsafe_allow_html=True)
        st.map(camera_df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})[["latitude", "longitude"]])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("**Station GPS Telemetry**", unsafe_allow_html=True)
        st.dataframe(camera_df.set_index("Camera ID")[['Location', 'Status', 'Alert Active']], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"❌ Failed to load model weights: {model_error}")
    else:
        # ---- INPUT MODE SELECTOR ----
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:12px;
            padding:6px 12px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:11px;font-weight:700;color:{C['text_muted']};text-transform:uppercase;
                letter-spacing:0.8px;">Input Mode:</span>
        </div>
        """, unsafe_allow_html=True)
        
        input_mode = st.radio(
            "Input Mode",
            options=["📹 Upload Image / Video", "📡 Simulated CCTV Station"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode_radio"
        )
        
        uploaded_file = None
        cctv_choice = None
        
        if input_mode == "📹 Upload Image / Video":
            # ---- PROMINENT UPLOAD PANEL ----
            col_upload_main, col_upload_ctrl = st.columns([7, 3])
            
            with col_upload_main:
                st.markdown(f"""
                <div style="background:{C['surface']};backdrop-filter:blur(12px);
                    border:2px dashed {C['accent']};border-radius:14px;padding:28px 24px;
                    box-shadow:0 0 20px {C['accent_glow']};margin-bottom:16px;">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                        <div style="background:{C['accent_glow']};padding:10px;border-radius:10px;">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                                stroke="{C['accent']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="16 16 12 12 8 16"></polyline>
                                <line x1="12" y1="12" x2="12" y2="21"></line>
                                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path>
                            </svg>
                        </div>
                        <div>
                            <div style="font-size:15px;font-weight:800;color:{C['text']};">Upload Traffic Media</div>
                            <div style="font-size:11px;color:{C['text_muted']};margin-top:2px;">Supports JPG, PNG images and MP4 videos</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Drag & Drop or Browse your accident image/video file",
                    type=["png", "jpg", "jpeg", "mp4"],
                    help="Upload a road/traffic image or video clip to run the accident detection pipeline on it.",
                    label_visibility="visible"
                )
                
            with col_upload_ctrl:
                st.markdown(f'<div class="glass-panel"><div class="panel-header">Detection Controls</div>', unsafe_allow_html=True)
                conf_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
                    help="Higher values reduce false alerts; lower values detect more subtle incidents."
                )
                btn_process = st.button("🔍 Run Detection", type="primary", key="btn_upload_scan")
                if uploaded_file is None:
                    st.caption("⬅️ Upload a file first to enable scanning.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            active_cam_id = "External Upload Feed"
            active_location = "Custom Media Terminal"
            
        else:
            # ---- SIMULATED CCTV PANEL ----
            col_cctv_select, col_cctv_opts = st.columns([7, 3])
            
            with col_cctv_select:
                st.markdown('<div class="glass-panel"><div class="panel-header">📡 CCTV Channel Selector</div>', unsafe_allow_html=True)
                cctv_choice = st.selectbox(
                    "Select Active CCTV Camera Station",
                    options=list(st.session_state["camera_states"].keys()),
                    help="Choose a pre-defined CCTV monitoring point to simulate a live stream."
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_cctv_opts:
                st.markdown('<div class="glass-panel"><div class="panel-header">Station Telemetry</div>', unsafe_allow_html=True)
                cam_info = st.session_state["camera_states"][cctv_choice]
                st.write(f"**GPS Zone:** `{cam_info['location']}`")
                st.write(f"**Status:** `ONLINE` 🟢")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Action layout for CCTV mode
            btn_col1, btn_col2 = st.columns([7, 3])
            with btn_col2:
                st.markdown('<div class="glass-panel"><div class="panel-header">Control Action Room</div>', unsafe_allow_html=True)
                conf_threshold = st.slider(
                    "Inference Sensitivity (Confidence Threshold)",
                    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
                    help="Higher values reduce false alerts; lower values catch more subtle incidents."
                )
                btn_process = st.button("Initiate Active Scan", type="primary", key="btn_cctv_scan")
                st.markdown('</div>', unsafe_allow_html=True)
                
            active_cam_id = cctv_choice
            active_location = st.session_state["camera_states"][cctv_choice]["location"]
            
        # Display streams area
        # Determine what input we're working with
        is_video = False
        video_path = None
        static_img = None
        
        if input_mode == "📹 Upload Image / Video":
            # Upload mode — only show inference area if a file was uploaded
            if uploaded_file is not None:
                st.markdown('<div class="section-title">Deep Inference Operational Feeds</div>', unsafe_allow_html=True)
                col_orig, col_det, col_stats = st.columns([3.5, 3.5, 3])
                file_extension = uploaded_file.name.split(".")[-1].lower()
                if file_extension in ["jpg", "jpeg", "png"]:
                    uploaded_file.seek(0)
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    static_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                elif file_extension == "mp4":
                    is_video = True
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    uploaded_file.seek(0)
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    video_path = tfile.name
        else:
            # CCTV simulation mode — always show the inference area
            st.markdown('<div class="section-title">Deep Inference Operational Feeds</div>', unsafe_allow_html=True)
            col_orig, col_det, col_stats = st.columns([3.5, 3.5, 3])
            is_video = True
            video_path = "https://assets.ultralytics.com/assets/stuttgart.mp4"
        
        # Only enter the rendering block when we have valid input
        if (input_mode == "📹 Upload Image / Video" and uploaded_file is None):
            pass  # Nothing to render yet — uploader shown above
        elif True:

            # --- RENDER STATIC IMAGE PIPELINE ---
            if not is_video and static_img is not None:
                with col_orig:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Raw CCTV Feed Frame</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(static_img, cv2.COLOR_BGR2RGB), width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_det:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Deep Inference Mapping</div>', unsafe_allow_html=True)
                    det_placeholder = st.empty()
                    btn_label = "🔍 Run Detection" if input_mode == "📹 Upload Image / Video" else "Initiate Active Scan"
                    det_placeholder.info(f"Click **{btn_label}** to compute telemetry.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_stats:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Incident Telemetry</div>', unsafe_allow_html=True)
                    stats_placeholder = st.empty()
                    stats_placeholder.write("Awaiting scanning cycle...")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                if btn_process:
                    with st.spinner("Processing feed telemetry..."):
                        annotated, heatmap, classification, confidence, severity, box = process_image(
                            static_img, model, severity_model=severity_model if severity_model_loaded else None, threshold=conf_threshold
                        )
                        
                        # Render outputs
                        with col_det:
                            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                            
                        with col_stats:
                            acc_type = f"{severity} Vehicle Collision" if classification == "Accident" else "None"
                            severity = severity if classification == "Accident" else "None"
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Trigger emergency integrations on detection
                            dispatch_msg = "None"
                            session_id = "None"
                            if classification == "Accident":
                                session_id, dispatch_msg = trigger_authority_dispatch(
                                    classification, severity, acc_type, confidence, active_cam_id, active_location, timestamp
                                )
                                # Activate camera alert state
                                if active_cam_id in st.session_state["camera_states"]:
                                    st.session_state["camera_states"][active_cam_id]["alert_active"] = True
                            
                            # Record to global history
                            lat, lon = resolve_camera_coords(active_cam_id)
                            new_row = {
                                "id": session_id if classification == "Accident" else f"RA-{uuid.uuid4().hex[:5].upper()}",
                                "timestamp": timestamp,
                                "classification": classification,
                                "type": acc_type,
                                "severity": severity,
                                "confidence": confidence,
                                "camera_id": active_cam_id,
                                "location": active_location,
                                "lat": lat,
                                "lon": lon,
                                "dispatch_status": dispatch_msg
                            }
                            st.session_state["history"] = pd.concat(
                                [pd.DataFrame([new_row]), st.session_state["history"]], 
                                ignore_index=True
                            )
                            save_history(st.session_state["history"])
                            
                            # Telemetry Card
                            class_color = C["danger"] if classification == "Accident" else C["success"]
                            alert_panel_class = "alert-active-panel" if classification == "Accident" else ""
                            
                            stats_html = f"""
                            <div class="glass-panel {alert_panel_class}" style="padding:10px; border-radius:8px;">
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">CLASSIFICATION</span>
                                    <span style="color:{class_color};font-weight:800;font-family:'JetBrains Mono',monospace;">{classification.upper()}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">ACCIDENT CONFIDENCE</span>
                                    <span style="color:{C["text"]};font-weight:800;font-family:'JetBrains Mono',monospace;">{confidence:.2%}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">ACCIDENT SEVERITY</span>
                                    <span style="color:{class_color};font-weight:800;font-family:'JetBrains Mono',monospace;">{severity.upper()}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">DISPATCH DETAILS</span>
                                    <span style="color:{C["accent"]};font-weight:700;font-size:11px;text-align:right;">{dispatch_msg}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;padding:8px 0 0 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">TELEMETRY REF</span>
                                    <span style="color:{C["text"]};font-family:'JetBrains Mono',monospace;font-size:11px;">{new_row['id']}</span>
                                </div>
                            </div>
                            """
                            stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                            
                            # Expandable diagnostic panel
                            with st.expander("🔍 Model Diagnostics & Inference Info", expanded=False):
                                st.write(f"**Task Domain:** `{model.task}`")
                                st.write(f"**Feed Resolution:** {static_img.shape if static_img is not None else 'None'}")
                                st.write(f"**Classification Model:** `YOLOv8-Classify`" if getattr(model, "task", "detect") == "classify" else f"**Object Detector:** `YOLOv8-Detect`")

            # --- RENDER VIDEO LIVE FEED SCANNER ---
            elif is_video and video_path is not None:
                # Load video Capture
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 100
                
                with col_orig:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Raw CCTV Feed Stream</div>', unsafe_allow_html=True)
                    if cctv_choice != "Custom Upload / External Media Feed":
                        # Explain that stream selection acts as simulated stream
                        st.info("Simulating Live CCTV Feed. Scan to analyze frames in real-time.")
                    else:
                        st.video(video_path, format='video/mp4')
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_det:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Deep Inference Live Stream</div>', unsafe_allow_html=True)
                    video_placeholder = st.empty()
                    video_placeholder.info("Click **Initiate Active Scan** to connect inference pipelines.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_stats:
                    st.markdown('<div class="glass-panel"><div class="panel-header">Operational Analytics</div>', unsafe_allow_html=True)
                    live_stats_placeholder = st.empty()
                    live_stats_placeholder.write("Awaiting active scan...")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                if btn_process:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Store intermediate classification metrics
                    all_scores = []
                    accident_frames_count = 0
                    current_severity = "None"
                    alert_triggered = False
                    dispatch_status_label = "None"
                    saved_session_id = "None"
                    
                    step = 4
                    current_frame_idx = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        if current_frame_idx % step == 0:
                            pct = min(current_frame_idx / frame_count, 1.0)
                            progress_bar.progress(pct)
                            status_text.text(f"Analyzing Live CCTV Feed Frame {current_frame_idx}/{frame_count}...")
                            
                            # Run inference
                            annotated, heatmap, classification, confidence, frame_severity, box = process_image(
                                frame, model, threshold=conf_threshold, severity_model=severity_model if severity_model_loaded else None
                            )
                            
                            # Render current frame
                            display_frame = heatmap if (classification == "Accident" and heatmap is not None) else annotated
                            video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width='stretch')
                            
                            all_scores.append(confidence)
                            if classification == "Accident":
                                accident_frames_count += 1
                                current_severity = frame_severity
                                
                                # Trigger automated emergency dispatch ONCE during the video stream
                                if not alert_triggered:
                                    alert_triggered = True
                                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    acc_type = f"{current_severity} Vehicle Collision"
                                    saved_session_id, dispatch_status_label = trigger_authority_dispatch(
                                        classification, current_severity, acc_type, confidence, active_cam_id, active_location, timestamp
                                    )
                                    if active_cam_id in st.session_state["camera_states"]:
                                        st.session_state["camera_states"][active_cam_id]["alert_active"] = True
                            
                            # Stats Panel Update
                            class_val = "Accident" if accident_frames_count > 1 else "Normal"
                            avg_conf = np.mean(all_scores) if len(all_scores) > 0 else 0.0
                            acc_type = f"{current_severity} Vehicle Collision" if class_val == "Accident" else "None"
                            severity = current_severity if class_val == "Accident" else "None"
                            
                            class_color = C["danger"] if class_val == "Accident" else C["success"]
                            alert_panel_class = "alert-active-panel" if class_val == "Accident" else ""
                            
                            live_stats_html = f"""
                            <div class="glass-panel {alert_panel_class}" style="padding:10px; border-radius:8px;">
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">STREAM STATE</span>
                                    <span style="color:{class_color};font-weight:800;font-family:'JetBrains Mono',monospace;">{class_val.upper()}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">AVG ACCIDENT CONFIDENCE</span>
                                    <span style="color:{C["text"]};font-weight:800;font-family:'JetBrains Mono',monospace;">{avg_conf:.2%}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">ACCIDENT FRAMES DETECTED</span>
                                    <span style="color:{C["warning"]};font-weight:800;font-family:'JetBrains Mono',monospace;">{accident_frames_count}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;border-bottom:1px solid {C["border_subtle"]};padding:8px 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">INCIDENT SEVERITY</span>
                                    <span style="color:{class_color};font-weight:800;font-family:'JetBrains Mono',monospace;">{severity.upper()}</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;padding:8px 0 0 0;">
                                    <span style="color:{C["text_muted"]};font-size:11px;font-weight:700;">DISPATCH COORDINATION</span>
                                    <span style="color:{C["accent"]};font-weight:700;font-size:11px;text-align:right;">{dispatch_status_label}</span>
                                </div>
                            </div>
                            """
                            live_stats_placeholder.markdown(live_stats_html, unsafe_allow_html=True)
                            
                        current_frame_idx += 1
                        
                    cap.release()
                    progress_bar.progress(1.0)
                    status_text.text("CCTV Live Feed Analysis Cycle Completed.")
                    
                    # Final database update if an accident occurred and was logged
                    final_class = "Accident" if accident_frames_count > 1 else "Normal"
                    final_conf = np.max(all_scores) if len(all_scores) > 0 else 0.0
                    final_type = f"{current_severity} Vehicle Collision" if final_class == "Accident" else "None"
                    final_sev = current_severity if final_class == "Accident" else "None"
                    
                    if final_class == "Accident" and not alert_triggered:
                        # Fallback alert trigger if not triggered during frame loop
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        saved_session_id, dispatch_status_label = trigger_authority_dispatch(
                            final_class, final_sev, final_type, final_conf, active_cam_id, active_location, timestamp
                        )
                        if active_cam_id in st.session_state["camera_states"]:
                            st.session_state["camera_states"][active_cam_id]["alert_active"] = True
                            
                    elif final_class == "Normal":
                        saved_session_id = f"RA-{uuid.uuid4().hex[:5].upper()}"
                        dispatch_status_label = "None"
                        
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    lat, lon = resolve_camera_coords(active_cam_id)
                    new_row = {
                        "id": saved_session_id,
                        "timestamp": timestamp,
                        "classification": final_class,
                        "type": final_type,
                        "severity": final_sev,
                        "confidence": final_conf,
                        "camera_id": active_cam_id,
                        "location": active_location,
                        "lat": lat,
                        "lon": lon,
                        "dispatch_status": dispatch_status_label
                    }
                    st.session_state["history"] = pd.concat(
                        [pd.DataFrame([new_row]), st.session_state["history"]], 
                        ignore_index=True
                    )
                    save_history(st.session_state["history"])
                    
                    # Clean up custom video temp file
                    if cctv_choice == "Custom Upload / External Media Feed" and os.path.exists(video_path):
                        os.unlink(video_path)

        # Dynamic verified active alerts control panel
        st.markdown('<div class="section-title">Verified Alerts Operational Queue</div>', unsafe_allow_html=True)
        alert_df = df[df["classification"] == "Accident"].copy()
        if len(alert_df) == 0:
            st.success("✔ Active Alert Queue is empty. No active road incidents logged.")
        else:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            for idx, r in alert_df.head(4).iterrows():
                # Display individual incidents with actions
                c1, c2, c3, c4 = st.columns([2.5, 4.5, 3, 2])
                with c1:
                    st.markdown(f"**ID:** `{r['id']}`  \n**Time:** *{r['timestamp']}*")
                with c2:
                    st.markdown(f"**Camera:** {r['camera_id']}  \n**Location:** {r['location']}")
                with c3:
                    sev_lbl = f"<span style='color:{C['danger']};font-weight:bold;'>🚨 {r['severity'].upper()}</span>"
                    st.markdown(f"**Severity:** {sev_lbl}  \n**State:** *{r['dispatch_status']}*", unsafe_allow_html=True)
                with c4:
                    # Operational buttons
                    if "Resolved" not in r["dispatch_status"]:
                        if st.button("Mark Resolved", key=f"res_{r['id']}"):
                            # Update df
                            st.session_state["history"].loc[st.session_state["history"]["id"] == r["id"], "dispatch_status"] = "Resolved"
                            save_history(st.session_state["history"])
                            # Turn off camera alerts
                            cam_id = r["camera_id"]
                            if cam_id in st.session_state["camera_states"]:
                                st.session_state["camera_states"][cam_id]["alert_active"] = False
                            st.rerun()
                    else:
                        st.markdown("<span style='color:green;font-weight:bold;'>CLOSED ✔</span>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:10px 0; border-color:rgba(128,128,128,0.1);'>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 2. INCIDENT ANALYTICS TAB
# ----------------------------------------------------
with tab_analytics:
    df_an = st.session_state["history"]
    
    st.markdown('<div class="section-title">Command Center Analytics</div>', unsafe_allow_html=True)

    camera_states = st.session_state.get("camera_states", {})
    coverage_df = pd.DataFrame([
        {
            "Camera ID": cam_id,
            "Location": cam["location"],
            "Status": cam.get("status", "UNKNOWN"),
            "latitude": cam.get("lat"),
            "longitude": cam.get("lon"),
            "Alert Active": "Yes" if cam.get("alert_active", False) else "No"
        }
        for cam_id, cam in camera_states.items()
    ])

    if not coverage_df.empty:
        st.markdown('<div class="glass-panel"><div class="panel-header">CCTV Coverage GPS Map</div>', unsafe_allow_html=True)
        st.map(coverage_df[["latitude", "longitude"]])
        st.markdown('</div>', unsafe_allow_html=True)

    if "lat" in df_an.columns and "lon" in df_an.columns and len(df_an[df_an["classification"] == "Accident"]):
        map_df = df_an[df_an["classification"] == "Accident"].copy()
        map_df = map_df.rename(columns={"lat": "latitude", "lon": "longitude"})
        map_df = map_df.dropna(subset=["latitude", "longitude"])
        if not map_df.empty:
            route_rows = []
            line_traces = []
            for _, row in map_df.iterrows():
                nearest = find_nearest_responder(row["latitude"], row["longitude"])
                if nearest is None:
                    continue
                eta = format_eta(nearest["distance_km"])
                route_rows.append({
                    "Incident ID": row["id"],
                    "Camera": row["camera_id"],
                    "Location": row["location"],
                    "Responder Hub": nearest["name"],
                    "Distance (km)": f"{nearest['distance_km']:.1f}",
                    "ETA": eta
                })
                line_traces.append(go.Scattermapbox(
                    lat=[row["latitude"], nearest["lat"]],
                    lon=[row["longitude"], nearest["lon"]],
                    mode="lines",
                    line=dict(width=2, color="#F59E0B"),
                    hoverinfo="text",
                    text=[f"Incident {row['id']}", f"Responder {nearest['name']}"],
                    showlegend=False
                ))

            if route_rows:
                route_fig = go.Figure()
                route_fig.add_trace(go.Scattermapbox(
                    lat=map_df["latitude"],
                    lon=map_df["longitude"],
                    mode="markers+text",
                    marker=dict(size=12, color="#EF4444"),
                    text=map_df["camera_id"],
                    textposition="top right",
                    name="Accident"
                ))
                route_fig.add_trace(go.Scattermapbox(
                    lat=[resp["lat"] for resp in RESPONDER_STATIONS],
                    lon=[resp["lon"] for resp in RESPONDER_STATIONS],
                    mode="markers+text",
                    marker=dict(size=14, color="#10B981"),
                    text=[resp["name"] for resp in RESPONDER_STATIONS],
                    textposition="bottom left",
                    name="Responder Hub"
                ))
                for trace in line_traces:
                    route_fig.add_trace(trace)
                center_lat = float(map_df["latitude"].mean())
                center_lon = float(map_df["longitude"].mean())
                route_fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=12),
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(font=dict(color=get_theme_colors(st.session_state.theme)["text"]))
                )

                st.markdown('<div class="glass-panel"><div class="panel-header">Predicted Response Routes</div>', unsafe_allow_html=True)
                st.plotly_chart(route_fig, use_container_width=True)
                st.dataframe(pd.DataFrame(route_rows), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        map_df = df_an[df_an["classification"] == "Accident"].copy()
        map_df = map_df.rename(columns={"lat": "latitude", "lon": "longitude"})
        map_df = map_df.dropna(subset=["latitude", "longitude"])
        st.markdown('<div class="glass-panel"><div class="panel-header">Incident Hotspot Map</div>', unsafe_allow_html=True)
        if not map_df.empty:
            st.map(map_df[["latitude", "longitude"]])
        else:
            st.info("Incident hotspot map will appear when accident records contain valid GPS coordinates.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Incident hotspot map will appear after the first detected accident with camera GPS coordinates.")
    col_chart_row1_1, col_chart_row1_2 = st.columns(2)
    
    with col_chart_row1_1:
        st.markdown('<div class="glass-panel"><div class="panel-header">Frequency of Incident Typologies</div>', unsafe_allow_html=True)
        # Bar Chart of Incident Types
        bar_df = df_an[df_an["type"] != "None"]["type"].value_counts().reset_index()
        bar_df.columns = ["Accident Type", "Frequency"]
        
        fig_bar = px.bar(
            bar_df,
            x="Accident Type",
            y="Frequency",
            color="Accident Type",
            color_discrete_sequence=['#3B82F6', '#EF4444', '#F59E0B', '#10B981'],
            text="Frequency"
        )
        fig_bar.update_layout(
            **plotly_theme_layout(st.session_state.theme, height=280, showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=C["border_subtle"]))
        )
        st.plotly_chart(fig_bar, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart_row1_2:
        st.markdown('<div class="glass-panel"><div class="panel-header">Historical Incident Vector (Dailies)</div>', unsafe_allow_html=True)
        # Line chart trend over days
        df_an["date"] = pd.to_datetime(df_an["timestamp"]).dt.date
        trend_df = df_an.groupby(["date", "classification"]).size().reset_index(name="count")
        
        fig_line = px.line(
            trend_df,
            x="date",
            y="count",
            color="classification",
            color_discrete_map={'Normal': '#10B981', 'Accident': '#EF4444'},
            markers=True
        )
        fig_line.update_layout(
            **plotly_theme_layout(st.session_state.theme, height=280,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(showgrid=False, tickmode='linear'),
                yaxis=dict(showgrid=True, gridcolor=C["border_subtle"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        )
        st.plotly_chart(fig_line, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
    col_chart_row2_1, col_chart_row2_2 = st.columns(2)
    
    with col_chart_row2_1:
        st.markdown('<div class="glass-panel"><div class="panel-header">Incident Severity Vector</div>', unsafe_allow_html=True)
        # Severity pie chart
        sev_df = df_an[df_an["severity"] != "None"]["severity"].value_counts().reset_index()
        sev_df.columns = ["Severity", "Count"]
        
        # Order severity
        order = {"Critical": 0, "High": 1, "Substantial": 2, "Minor": 3, "Medium": 4, "Low": 5}
        sev_df["order"] = sev_df["Severity"].map(order)
        sev_df = sev_df.sort_values("order")
        
        fig_sev = px.pie(
            sev_df,
            values="Count",
            names="Severity",
            color="Severity",
            color_discrete_map={
                'Critical': '#EF4444', 
                'High': '#F59E0B', 
                'Substantial': '#3B82F6', 
                'Minor': '#10B981',
                'Medium': '#6366F1'
            },
            hole=0.4
        )
        fig_sev.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color=C["chart_line"], width=2))
        )
        fig_sev.update_layout(
            **plotly_theme_layout(st.session_state.theme, height=280,
                margin=dict(l=20, r=20, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        )
        st.plotly_chart(fig_sev, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart_row2_2:
        st.markdown('<div class="glass-panel"><div class="panel-header">CCTV Incident Distribution by Station</div>', unsafe_allow_html=True)
        # Bar chart showing accidents per camera
        cam_df = df_an[df_an["classification"] == "Accident"]["camera_id"].value_counts().reset_index()
        cam_df.columns = ["CCTV Station", "Incidents"]
        
        fig_cam = px.bar(
            cam_df,
            y="CCTV Station",
            x="Incidents",
            orientation="h",
            color="CCTV Station",
            color_discrete_sequence=['#6366F1', '#4F46E5', '#3B82F6', '#60A5FA']
        )
        fig_cam.update_layout(
            **plotly_theme_layout(st.session_state.theme, height=280, showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(showgrid=True, gridcolor=C["border_subtle"]),
                yaxis=dict(showgrid=False))
        )
        st.plotly_chart(fig_cam, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 3. EMERGENCY DISPATCH CENTER TAB
# ----------------------------------------------------
with tab_dispatch:
    st.markdown('<div class="section-title">Emergency Communications Hub</div>', unsafe_allow_html=True)
    
    # Render dispatch logs
    st.markdown('<div class="glass-panel"><div class="panel-header">Live Authority Communications Log</div>', unsafe_allow_html=True)
    
    if len(st.session_state["dispatch_logs"]) == 0:
        st.info("No authority communication notifications have been triggered yet. System monitoring in stand-by.")
    else:
        disp_logs_df = pd.DataFrame(st.session_state["dispatch_logs"])
        # Format df columns
        disp_logs_df = disp_logs_df.rename(columns={
            "timestamp": "Dispatch Time",
            "camera_id": "CCTV Camera",
            "type": "Channel Type",
            "recipient": "Notified Destination",
            "status": "Transmission Status",
            "details": "Gateway Message/Details"
        })
        st.dataframe(disp_logs_df, width='stretch', hide_index=True)

        whatsapp_link = st.session_state.get("last_whatsapp_link")
        if whatsapp_link:
            st.markdown(f"<div style='margin-top:16px;padding:14px;border:1px solid #10B981;background:#ECFDF5;border-radius:12px;'>"
                        f"<strong>WhatsApp Alert Ready:</strong> "
                        f"<a href=\"{whatsapp_link}\" target=\"_blank\">Open WhatsApp Chat</a> "
                        f"(Auto-open attempted; if blocked, use the link manually.)</div>",
                        unsafe_allow_html=True)
            st.components.v1.html(
                f"<script>"
                f"if (!window.roadguardWhatsappAutoOpened) {{"
                f"  window.roadguardWhatsappAutoOpened = true;"
                f"  setTimeout(function() {{ window.open(\"{whatsapp_link}\", \"_blank\"); }}, 500);"
                f"}}"
                f"</script>",
                height=0,
                scrolling=False
            )

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search and Filter Records
    st.markdown('<div class="section-title">Historical Incident Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel"><div class="panel-header">Search & Filter Logs</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_class = st.selectbox(
            "Filter Classification",
            options=["All Statuses", "Accident", "Normal"],
            key="dispatch_filter_class"
        )
    with col_f2:
        f_sev = st.selectbox(
            "Filter Severity",
            options=["All Severities", "Critical", "High", "Substantial", "Minor", "None"],
            key="dispatch_filter_sev"
        )
    with col_f3:
        search_query = st.text_input("Search Logs by ID/Location", value="", key="dispatch_search")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter operations
    df_rep = df.copy()
    if f_class != "All Statuses":
        df_rep = df_rep[df_rep["classification"] == f_class]
    if f_sev != "All Severities":
        df_rep = df_rep[df_rep["severity"] == f_sev]
    if search_query != "":
        df_rep = df_rep[
            df_rep["id"].str.contains(search_query, case=False) |
            df_rep["location"].str.contains(search_query, case=False)
        ]
        
    # Render table
    st.markdown('<div class="glass-panel"><div class="panel-header">Operational Records Archive</div>', unsafe_allow_html=True)
    
    view_df = df_rep.rename(columns={
        "id": "Log Incident ID",
        "timestamp": "Timestamp",
        "classification": "Scan Result",
        "type": "Crash Type",
        "severity": "Severity Grade",
        "confidence": "Inference Conf",
        "camera_id": "Source CCTV",
        "location": "GPS Zone Location",
        "dispatch_status": "Response Status"
    })
    
    view_df["Inference Conf"] = view_df["Inference Conf"].map(lambda x: f"{x:.2%}")
    st.dataframe(view_df, width='stretch', hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_d1, col_d2 = st.columns([8, 2])
    with col_d2:
        csv_filtered = df_rep.to_csv(index=False)
        st.download_button(
            label="Download Log Archive (CSV)",
            data=csv_filtered,
            file_name="filtered_accident_report.csv",
            mime="text/csv",
            key="download_filtered_dispatch"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 4. SYSTEM SETTINGS TAB
# ----------------------------------------------------
with tab_config:
    st.markdown('<div class="section-title">Emergency Communications Gateway Settings Title</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel"><div class="panel-header">Twilio SMS Configuration</div>', unsafe_allow_html=True)
    
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.session_state["dispatch_config"]["twilio_sid"] = st.text_input(
            "Twilio Account SID", value=st.session_state["dispatch_config"]["twilio_sid"],
            type="password", help="Twilio account identifier for SMS dispatch."
        )
        st.session_state["dispatch_config"]["twilio_token"] = st.text_input(
            "Twilio Auth Token", value=st.session_state["dispatch_config"]["twilio_token"],
            type="password", help="Authorization Token for Twilio API."
        )
    with c_col2:
        st.session_state["dispatch_config"]["twilio_from"] = st.text_input(
            "Twilio Outbound Phone Number", value=st.session_state["dispatch_config"]["twilio_from"],
            help="Your registered Twilio phone number."
        )
        st.session_state["dispatch_config"]["whatsapp_from"] = st.text_input(
            "Twilio WhatsApp Sender Number", value=st.session_state["dispatch_config"]["whatsapp_from"],
            help="Your Twilio WhatsApp sender number, e.g. whatsapp:+14155238886. Leave empty to use free manual WhatsApp chat link."
        )
        st.session_state["dispatch_config"]["police_phone"] = st.text_input(
            "Authority Recipient Phone Number", value=st.session_state["dispatch_config"]["police_phone"],
            help="Phone number of Traffic Control / Police dispatch (recipient of SMS alerts)."
        )
        st.session_state["dispatch_config"]["whatsapp_to"] = st.text_input(
            "Emergency WhatsApp Recipient Number", value=st.session_state["dispatch_config"]["whatsapp_to"],
            help="Destination WhatsApp number, e.g. 03141988998 or +923141988998."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="panel-header">SMTP Email Gateway Configuration</div>', unsafe_allow_html=True)
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.session_state["dispatch_config"]["smtp_server"] = st.text_input(
            "SMTP Server Host", value=st.session_state["dispatch_config"]["smtp_server"],
            help="Outgoing mail server (e.g. smtp.gmail.com)."
        )
        st.session_state["dispatch_config"]["smtp_port"] = st.number_input(
            "SMTP Server Port", value=int(st.session_state["dispatch_config"]["smtp_port"]),
            help="SMTP Port (default TLS is 587)."
        )
        st.session_state["dispatch_config"]["hospital_email"] = st.text_input(
            "Hospital Emergency Dispatcher Email", value=st.session_state["dispatch_config"]["hospital_email"],
            help="Email address of First Responder/EMS/Hospital trauma care center."
        )
    with e_col2:
        st.session_state["dispatch_config"]["smtp_user"] = st.text_input(
            "SMTP Username / Email", value=st.session_state["dispatch_config"]["smtp_user"],
            help="Authentication email address."
        )
        st.session_state["dispatch_config"]["smtp_pass"] = st.text_input(
            "SMTP Password / App Password", value=st.session_state["dispatch_config"]["smtp_pass"],
            type="password", help="Gmail users should generate an 'App Password' for connection."
        )
        st.session_state["dispatch_config"]["smtp_from"] = st.text_input(
            "Sender Email (From)", value=st.session_state["dispatch_config"]["smtp_from"],
            help="Display sender email address (leave empty to use username)."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="panel-header">Enterprise CAD Webhook API Integration</div>', unsafe_allow_html=True)
    st.session_state["dispatch_config"]["webhook_url"] = st.text_input(
        "Incident CAD Webhook POST URL", value=st.session_state["dispatch_config"]["webhook_url"],
        help="Municipal dispatch server webhook endpoint to automatically register accidents."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel"><div class="panel-header">AI Pipeline Calibration Diagnostics</div>', unsafe_allow_html=True)
    st.write(f"**YOLOv8 Accident Model Status:** Loaded ✅ (`{MODEL_PATH}`)" if model_loaded else "Model Unloaded ❌")
    st.write(f"**YOLOv8 Severity Model Status:** Loaded ✅ (`{SEVERITY_MODEL_PATH}`)" if severity_model_loaded else "Severity Model Unloaded ❌")
    
    st.write("---")
    st.caption("Verify raw system variables or trigger database recalibration below.")
    if st.button("Reset In-Memory Incident Logs"):
        now = datetime.datetime.now()
        st.session_state["history"] = pd.DataFrame(get_demo_records(now))
        st.session_state["dispatch_logs"] = []
        save_history(st.session_state["history"])
        save_dispatch_logs(st.session_state["dispatch_logs"])
        st.success("In-memory logs successfully reset to default simulation logs.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown(f"""
<div class="app-footer">
    RoadGuard AI Command Station &mdash; YOLOv8-Powered Real-Time Traffic Accident Detection Dashboard &nbsp;|&nbsp;
    Machine Learning Fundamentals Project &nbsp;|&nbsp;
    {datetime.datetime.now().strftime("%Y")}
</div>
""", unsafe_allow_html=True)
