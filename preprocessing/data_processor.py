import cv2
import mediapipe as mp
import os
import pandas as pd
from glob import glob
from hand_detection import HandDetector

class DataProcessor:
    def __init__(self):
        self.detector = HandDetector(static_image_mode=True)

    def process_image_folder(self, folder_path, output_csv):
        """
        Process all images in a folder and save hand coordinates to CSV
        
        Args:
            folder_path (str): Path to folder containing images
            output_csv (str): Path to output CSV file
        """
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob(os.path.join(folder_path, ext)))
        
        all_coordinates = []
        
        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = self.detector.get_relative_coordinates(hand_landmarks, image.shape[1], image.shape[0])
                    for i, coord in enumerate(coords):
                        all_coordinates.append({
                            'image': os.path.basename(img_path),
                            'hand_point': i,
                            'x': coord[0],
                            'y': coord[1],
                            'z': coord[2]
                        })
        
        if all_coordinates:
            df = pd.DataFrame(all_coordinates)
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Coordinates saved to {output_csv}")
        else:
            print("No hands detected in any images")