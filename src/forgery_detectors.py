import cv2
import numpy as np

class ForgeryDetectors:
    def __init__(self):
        # Initialize the ORB detector for Copy-Move sensing
        # ORB is fast and free to use (unlike SIFT in some OpenCV versions)
        #orb detect a sharp edge , texture pattrun , corner
        #after that it  genertae a numaricode for each keypoint
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_copy_move(self, image):
        """
        Detects parts of the image that have been copied and pasted elsewhere.
        Uses keypoint matching within the same image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) < 10:
            return 0.0 # Not enough detail to detect cloning

        # Match descriptors with themselves (excluding perfect matches at same location)
    
        # For our "Evidence-First" ML model, we'll calculate a 'Cloning Score'
        
        # Simple heuristic: how many keypoints are 'too similar' despite being in different locations
        matches = self.bf.match(des, des)
        
        # Filter matches:
        # 1. Distance should be small (high similarity)
        # 2. Geometric distance should be large (not the same point)
        cloning_points = 0
        for m in matches:
            idx1, idx2 = m.queryIdx, m.trainIdx
            if idx1 == idx2: continue
            
            p1 = kp[idx1].pt
            p2 = kp[idx2].pt
            
            # Calculate physical distance between points
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            
            # If descriptors are identical but locations are far apart (> 20 pixels)
            if m.distance < 10 and dist > 20: 
                cloning_points += 1
                
        # Normalize score
        score = (cloning_points / len(kp)) * 100 if len(kp) > 0 else 0
        return score

    def detect_noise_inconsistency(self, noise_map):
        """
        Analyzes the noise map for statistical anomalies.
        Spliced parts often have different noise variance than the background.
        """
        # Divide image into a grid (e.g., 8x8)
        h, w = noise_map.shape[:2]
        grid_h, grid_w = h // 8, w // 8
        
        variances = []
        for i in range(8):
            for j in range(8):
                section = noise_map[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                variances.append(np.var(section))
        
        # Calculate the variance of the variances (how inconsistent is the noise?)
        inconsistency_score = np.std(variances)
        return inconsistency_score

    def get_forgery_report(self, image, noise_map):
        """
        Aggregates forgery-specific scores.
        """
        copy_move_score = self.detect_copy_move(image)
        noise_inc_score = self.detect_noise_inconsistency(noise_map)
        
        return {
            "copy_move_score": float(copy_move_score),
            "noise_inconsistency": float(noise_inc_score)
        }

if __name__ == "__main__":
    print("Forgery Detectors initialized. Splicing and Copy-Move modules ready.")
