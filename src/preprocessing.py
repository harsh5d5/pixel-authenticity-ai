import cv2
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def resize_with_padding(self, image):
        """Resizes image maintaining aspect ratio and adds black padding to reach target_size."""
        h, w = image.shape[:2]
        sh, sw = self.target_size
        
        # Calculate aspect ratio
        aspect = w / h
        
        if aspect > 1: # wider
            new_w = sw
            new_h = int(sw / aspect)
        else: # taller or square
            new_h = sh
            new_w = int(sh * aspect)
            
        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center image
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        x_offset = (sw - new_w) // 2
        y_offset = (sh - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled_img
        return canvas

    def get_color_spaces(self, image):
        """Returns the image in different color spaces useful for forensics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"rgb": rgb, "gray": gray, "ycbcr": ycbcr}

    def extract_noise_map(self, image):
        """
        Extracts high-frequency noise using a median filter subtraction.
        In forensics, noise inconsistencies often reveal spliced parts.
        """
        # Convert to float for subtraction
        img_float = image.astype(np.float32)
        # Apply median filter to get a denoised version
        denoised = cv2.medianBlur(image, 3).astype(np.float32)
        # The 'noise' is the difference between original and denoised
        noise_map = cv2.absdiff(img_float, denoised)
        # Normalize for visualization/analysis
        noise_map = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return noise_map

    def normalize(self, image):
        """Scales pixel values to [0, 1] range."""
        return image.astype(np.float32) / 255.0

    def process(self, image_path):
        """Complete preprocessing pipeline for a single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # 1. Standardize Size
        raw_processed = self.resize_with_padding(img)
        
        # 2. Extract Forensic Noise Map before further modification
        noise = self.extract_noise_map(raw_processed)
        
        # 3. Get Color variations
        spaces = self.get_color_spaces(raw_processed)
        
        # 4. Normalize
        normalized_rgb = self.normalize(spaces['rgb'])
        
        return {
            "original_standardized": raw_processed,
            "noise_map": noise,
            "gray": spaces['gray'],
            "ycbcr": spaces['ycbcr'],
            "normalized_rgb": normalized_rgb
        }

if __name__ == "__main__":
    # Example usage / Test
    preprocessor = ImagePreprocessor()
    print("Preprocessor class initialized. Ready to handle image forensic standardization.")
