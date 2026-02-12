import cv2
import numpy as np
from PIL import Image
import io

class ForensicExtractors:
    def __init__(self):
        pass

    def run_ela(self, image, quality=90):
        """
        Error Level Analysis (ELA)
        Detects differences in compression levels within an image.
        Modified areas will typically show higher error levels.
        """
        # 1. Compress image in memory
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # 2. Re-decode the compressed image from buffer
        compressed_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        if compressed_img is None:
            return np.zeros_like(image)

        # 3. Calculate absolute difference
        ela_map = cv2.absdiff(image, compressed_img)
        
        # 4. Amplify the difference for better visualization/feature extraction
        max_diff = np.max(ela_map)
        if max_diff == 0: max_diff = 1 # Avoid division by zero
        scale = 255.0 / max_diff
        ela_map = (ela_map * scale).astype(np.uint8)
        
        return ela_map

    def run_fft(self, gray_image):
        """
        Frequency Domain Analysis (FFT)
        Identifies unnatural patterns or checkerboard artifacts typical of AI generators.
        """
        # 1. Perform 2D Fast Fourier Transform
        dft = np.fft.fft2(gray_image)
        # 2. Shift the zero-frequency component to the center of the spectrum
        dft_shift = np.fft.fftshift(dft)
        # 3. Calculate Magnitude Spectrum (on log scale for better features)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        
        return magnitude_spectrum.astype(np.uint8)

    def get_texture_features(self, gray_image):
        """
        Calculates texture consistency using Laplacian Variance.
        Low variance = blurry/smoothed (possibly fake)
        High variance = sharp/noisy
        """
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def extract_all_features(self, image_data):
        """
        Aggregates multiple forensic scores into a single feature vector.
        """
        # Calculate ELA
        ela = self.run_ela(image_data['original_standardized'])
        ela_mean = np.mean(ela)
        ela_std = np.std(ela)

        # Calculate FFT
        fft = self.run_fft(image_data['gray'])
        fft_mean = np.mean(fft)
        
        # Texture consistency
        texture_var = self.get_texture_features(image_data['gray'])

        # Noise statistics
        noise_mean = np.mean(image_data['noise_map'])
        
        # Return a dictionary of features
        return {
            "ela_mean": float(ela_mean),
            "ela_std": float(ela_std),
            "fft_mean": float(fft_mean),
            "texture_variance": float(texture_var),
            "noise_mean": float(noise_mean)
        }

if __name__ == "__main__":
    print("Forensic Extractors initialized. ELA and FFT engines are ready.")
