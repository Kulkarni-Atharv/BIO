import cv2
import os
import json
import logging
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Recognizer")

from shared.config import (
    YUNET_PATH, MOBILEFACENET_PATH, 
    EMBEDDINGS_FILE, NAMES_FILE,
    DETECTION_THRESHOLD, RECOGNITION_THRESHOLD
)

# [NEW] Import Aligner
try:
    from core.alignment import StandardFaceAligner
    aligner = StandardFaceAligner()
except ImportError:
    aligner = None


class FaceRecognizer:
    def __init__(self):
        # Disable OpenCV multi-threading to prevent GPU/memory conflicts on RPi
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        
        self.yunet_path = YUNET_PATH
        self.mobilefacenet_path = MOBILEFACENET_PATH
        self.embeddings_file = EMBEDDINGS_FILE
        self.names_file = NAMES_FILE
        
        self.detector = None
        self.recognizer = None
        self.known_embeddings = []
        self.known_names = []
        
        self._load_models()
        self._load_database()

    def _load_models(self):
        if not os.path.exists(self.yunet_path) or not os.path.exists(self.mobilefacenet_path):
            logger.error("Models not found.")
            return

        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path, "", (320, 320), DETECTION_THRESHOLD, 0.3, 5000
        )
        self.recognizer = cv2.dnn.readNetFromONNX(self.mobilefacenet_path)
        logger.info("Models loaded successfully.")

    def _load_database(self):
        if os.path.exists(self.embeddings_file) and os.path.exists(self.names_file):
            try:
                self.known_embeddings = np.load(self.embeddings_file)
                with open(self.names_file, 'r') as f:
                    self.known_names = json.load(f)
                logger.info(f"Loaded {len(self.known_embeddings)} identities.")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.known_embeddings = []
                self.known_names = []
        else:
            logger.warning("No database found.")

    def reload(self):
        """Safe reload of the database"""
        try:
            if os.path.exists(self.embeddings_file) and os.path.exists(self.names_file):
                # Load to temp variables first
                new_embeddings = np.load(self.embeddings_file)
                with open(self.names_file, 'r') as f:
                    new_names = json.load(f)
                
                # Atomic swap
                self.known_embeddings = new_embeddings
                self.known_names = new_names
                logger.info(f"Reloaded database: {len(self.known_embeddings)} identities.")
                return True
        except Exception as e:
            logger.error(f"Reload failed: {e}")
        return False

    def recognize_faces(self, frame):
        """
        Recognize faces in the given frame.
        Returns: (face_locations, face_names) where locations are (x,y,w,h) tuples
        """
        try:
            # Validate input
            if frame is None or frame.size == 0:
                return [], []
            
            if self.detector is None or self.recognizer is None:
                return [], []

            h, w = frame.shape[:2]
            if h < 10 or w < 10:
                return [], []
            
            self.detector.setInputSize((w, h))
            
            _, faces = self.detector.detect(frame)
            
            face_locations = []
            face_names = []

            if faces is not None and len(faces) > 0:
                for face in faces:
                    try:
                        # Bounding Box
                        box = face[:4].astype(int)
                        x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                        
                        # Validate bounding box
                        if w_box <= 0 or h_box <= 0:
                            continue
                        
                        # Landmarks for alignment
                        landmarks = face[4:14].reshape((5, 2))
                        
                        face_locations.append((x, y, w_box, h_box))
                        
                        name = "Unknown"
                        
                        if aligner:
                            try:
                                face_img = aligner.align(frame, landmarks)
                                
                                if face_img is not None and face_img.size > 0:
                                    # Validate aligned face
                                    if face_img.shape[0] != 112 or face_img.shape[1] != 112:
                                        logger.warning(f"Invalid aligned face shape: {face_img.shape}")
                                        face_names.append(name)
                                        continue
                                    
                                    # Get embedding with error handling
                                    blob = cv2.dnn.blobFromImage(
                                        face_img, 
                                        1.0/128.0, 
                                        (112, 112), 
                                        (127.5, 127.5, 127.5), 
                                        swapRB=True
                                    )
                                    
                                    if blob is None or blob.size == 0:
                                        face_names.append(name)
                                        continue
                                    
                                    self.recognizer.setInput(blob)
                                    embedding = self.recognizer.forward()
                                    
                                    if embedding is None or embedding.size == 0:
                                        face_names.append(name)
                                        continue
                                    
                                    embedding_norm = cv2.normalize(
                                        embedding, 
                                        None, 
                                        alpha=1, 
                                        beta=0, 
                                        norm_type=cv2.NORM_L2
                                    )
                                    
                                    # Compare with known embeddings
                                    if len(self.known_embeddings) > 0:
                                        scores = np.dot(self.known_embeddings, embedding_norm.T).flatten()
                                        best_match_idx = np.argmax(scores)
                                        max_score = scores[best_match_idx]
                                        
                                        if max_score > RECOGNITION_THRESHOLD:
                                            name = self.known_names[best_match_idx]
                                            
                            except cv2.error as e:
                                logger.error(f"OpenCV error during recognition: {e}")
                            except Exception as e:
                                logger.error(f"Recognition error: {e}")
                        
                        face_names.append(name)
                        
                    except Exception as e:
                        logger.error(f"Error processing face: {e}")
                        continue

            return face_locations, face_names
            
        except Exception as e:
            logger.error(f"Critical error in recognize_faces: {e}")
            return [], []
