#!/usr/bin/env python3
"""
Diagnostic script to isolate the exact crash point in face recognition pipeline.
Run this to see where the segfault occurs.
"""

import cv2
import numpy as np
import sys
import os

# Disable OpenCL
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

print("=" * 60)
print("FACE RECOGNITION CRASH DIAGNOSTIC")
print("=" * 60)

# Test 1: Load Models
print("\n[TEST 1] Loading YuNet detector...")
try:
    detector = cv2.FaceDetectorYN.create(
        "models/face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )
    print("✅ YuNet loaded successfully")
except Exception as e:
    print(f"❌ YuNet failed: {e}")
    sys.exit(1)

print("\n[TEST 2] Loading MobileFaceNet...")
try:
    recognizer = cv2.dnn.readNetFromONNX("models/face_recognition_sface_2021dec.onnx")
    print("✅ MobileFaceNet loaded successfully")
except Exception as e:
    print(f"❌ MobileFaceNet failed: {e}")
    sys.exit(1)

# Test 2: Camera
print("\n[TEST 3] Opening camera...")
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=3
    )
    picam2.configure(config)
    picam2.start()
    print("✅ Camera started")
    
    import time
    time.sleep(2)
    
    frame = picam2.capture_array()
    print(f"✅ Captured frame: {frame.shape}")
except Exception as e:
    print(f"❌ Camera failed: {e}")
    sys.exit(1)

# Test 3: Face Detection
print("\n[TEST 4] Detecting faces...")
try:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    
    if faces is None or len(faces) == 0:
        print("⚠️  No faces detected. Please position your face in front of camera.")
        print("   Rerun this script when ready.")
        picam2.stop()
        sys.exit(0)
    
    print(f"✅ Detected {len(faces)} face(s)")
    face = faces[0]
    box = face[:4].astype(int)
    landmarks = face[4:14].reshape((5, 2))
    print(f"   Box: {box}")
    print(f"   Landmarks shape: {landmarks.shape}")
except Exception as e:
    print(f"❌ Detection failed: {e}")
    picam2.stop()
    sys.exit(1)

# Test 4: Face Alignment
print("\n[TEST 5] Testing face alignment...")
try:
    from core.alignment import StandardFaceAligner
    aligner = StandardFaceAligner()
    print("✅ Aligner created")
    
    print("   Attempting alignment...")
    aligned_face = aligner.align(frame, landmarks)
    
    if aligned_face is None:
        print("❌ Alignment returned None")
        picam2.stop()
        sys.exit(1)
    
    print(f"✅ Aligned face: {aligned_face.shape}")
except Exception as e:
    print(f"❌ Alignment CRASHED: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 5: Blob Creation
print("\n[TEST 6] Creating blob for inference...")
try:
    blob = cv2.dnn.blobFromImage(
        aligned_face,
        1.0/128.0,
        (112, 112),
        (127.5, 127.5, 127.5),
        swapRB=True
    )
    print(f"✅ Blob created: {blob.shape}")
except Exception as e:
    print(f"❌ Blob creation CRASHED: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 6: Inference
print("\n[TEST 7] Running inference...")
try:
    recognizer.setInput(blob)
    print("   Input set...")
    
    embedding = recognizer.forward()
    print(f"✅ Inference successful: {embedding.shape}")
except Exception as e:
    print(f"❌ Inference CRASHED: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 7: Normalization
print("\n[TEST 8] Normalizing embedding...")
try:
    embedding_norm = cv2.normalize(
        embedding,
        None,
        alpha=1,
        beta=0,
        norm_type=cv2.NORM_L2
    )
    print(f"✅ Normalized: {embedding_norm.shape}")
except Exception as e:
    print(f"❌ Normalization CRASHED: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - NO CRASH DETECTED")
print("=" * 60)
print("\nIf hmi.py still crashes, the issue is in:")
print("  1. Qt/PyQt5 integration")
print("  2. Threading issues")
print("  3. Memory corruption over time")
print("\nPlease run: python hmi.py 2>&1 | tee crash_log.txt")
print("And share the crash_log.txt file.")

picam2.stop()
