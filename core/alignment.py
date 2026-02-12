
import cv2
import numpy as np
import logging

logger = logging.getLogger("Alignment")

class StandardFaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=112, desiredFaceHeight=112):
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        
        # Standard landmarks for 112x112 arcface
        self.reference_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def align(self, image, landmarks):
        """
        Aligns the face image using the provided 5 landmarks.
        landmarks: numpy array of shape (5, 2)
        Returns: aligned face image or None if alignment fails
        """
        try:
            # Validation
            if image is None or landmarks is None:
                return None
            
            if len(landmarks) != 5:
                return None
            
            # Validate image
            if image.shape[0] < 10 or image.shape[1] < 10:
                return None
            
            # Convert landmarks to float32
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Check for NaN or Inf in landmarks
            if not np.all(np.isfinite(landmarks)):
                logger.warning("Invalid landmarks detected (NaN/Inf)")
                return None
            
            # Estimate transformation matrix
            tform, inliers = cv2.estimateAffinePartial2D(
                landmarks, 
                self.reference_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if tform is None:
                return None
            
            # Validate transformation matrix
            if not np.all(np.isfinite(tform)):
                logger.warning("Invalid transformation matrix")
                return None
            
            # Apply transformation with border handling
            output_size = (self.desiredFaceWidth, self.desiredFaceHeight)
            aligned_face = cv2.warpAffine(
                image, 
                tform, 
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Validate output
            if aligned_face is None or aligned_face.size == 0:
                return None
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Alignment error: {e}")
            return None

# Default instance
aligner = StandardFaceAligner()
