
import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Server Configuration
SERVER_IP = "127.0.0.1"  # Default to localhost for testing
SERVER_PORT = 8000
API_BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/api"
SERVER_DB_PATH = os.path.join(DATA_DIR, "server_attendance.db")

# Device Configuration
DEVICE_ID = "cm4_device_001"

# MySQL Configuration
# MySQL Configuration
MYSQL_HOST = "127.0.0.1" 
MYSQL_USER = "root"
MYSQL_PASSWORD = "Atharv@123"
MYSQL_DB = "bio_attendance"
MYSQL_PORT = 3306

DB_PATH = os.path.join(DATA_DIR, "attendance_buffer.db")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
PROCESSED_LOG_PATH = os.path.join(DATA_DIR, "processed_log.json")

# Models & Embeddings
YUNET_PATH = os.path.join(ASSETS_DIR, "face_detection_yunet_2023mar.onnx")
MOBILEFACENET_PATH = os.path.join(ASSETS_DIR, "MobileFaceNet.onnx")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
NAMES_FILE = os.path.join(DATA_DIR, "names.json")

# MQTT Configuration
# Using EMQX Public Broker for testing
MQTT_BROKER = "broker.emqx.io" 
MQTT_PORT = 1883
MQTT_TOPIC = "attendance/updates"

# Camera Configuration
CAMERA_INDEX = 0 
DETECTION_THRESHOLD = 0.6 
RECOGNITION_THRESHOLD = 0.70
VERIFICATION_FRAMES = 5 

# --- LOCAL CONFIG OVERRIDE ---
# This allows each device to have its own settings (IP, Port, etc)
# without changing the code. Create a file 'data/local_config.json'
# with the settings you want to override.
LOCAL_CONFIG_PATH = os.path.join(DATA_DIR, "local_config.json")

if os.path.exists(LOCAL_CONFIG_PATH):
    import json
    try:
        with open(LOCAL_CONFIG_PATH, 'r') as f:
            local_config = json.load(f)
            # Override globals if key exists
            if "MYSQL_HOST" in local_config: MYSQL_HOST = local_config["MYSQL_HOST"]
            if "MYSQL_USER" in local_config: MYSQL_USER = local_config["MYSQL_USER"]
            if "MYSQL_PASSWORD" in local_config: MYSQL_PASSWORD = local_config["MYSQL_PASSWORD"]
            if "MYSQL_DB" in local_config: MYSQL_DB = local_config["MYSQL_DB"]
            if "MQTT_BROKER" in local_config: MQTT_BROKER = local_config["MQTT_BROKER"]
            if "DEVICE_ID" in local_config: DEVICE_ID = local_config["DEVICE_ID"]
            
        print(f"Loaded local configuration from {LOCAL_CONFIG_PATH}")
    except Exception as e:
        print(f"Error loading local config: {e}") 
