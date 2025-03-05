# Instructions for setting up the data preprocessing notebook

"""
To set up the data preprocessing notebook:

1. Create a new Jupyter notebook called 'data_preprocessing.ipynb'
2. Copy and paste the following code blocks into separate cells:

# Cell 1 - Imports (Code cell)
import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from hand_detection import HandDetector
%matplotlib inline

# Cell 2 - Initialize Hand Detector (Code cell)
detector = HandDetector(static_image_mode=True)

# Cell 3 - Process Single Image Function (Code cell)
def process_single_image(image_path, label, visualize=False):
    \"\"\"Process a single image and return hand coordinates relative to wrist
    
    Args:
        image_path (str): Path to the image file
        label (str): Label for the gesture (derived from folder name)
        visualize (bool): Whether to display the image with detected landmarks
    
    Returns:
        list: List of dictionaries containing relative hand coordinates
    \"\"\"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.hands.process(image_rgb)
    
    coordinates = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get raw coordinates
            coords = detector.get_relative_coordinates(hand_landmarks, image.shape[1], image.shape[0])
            
            # Get wrist coordinates (point 0 is the wrist in MediaPipe)
            wrist_x, wrist_y, wrist_z = coords[0]
            
            # Calculate coordinates relative to wrist
            relative_coords = []
            for coord in coords:
                relative_coords.append([
                    coord[0] - wrist_x,  # x relative to wrist
                    coord[1] - wrist_y,  # y relative to wrist
                    coord[2] - wrist_z   # z relative to wrist
                ])
            
            # Store coordinates with additional metadata
            for i, coord in enumerate(relative_coords):
                coordinates.append({
                    'image': os.path.basename(image_path),
                    'label': label,
                    'hand_point': i,
                    'x_rel': coord[0],
                    'y_rel': coord[1],
                    'z_rel': coord[2],
                    'x_abs': coords[i][0],
                    'y_abs': coords[i][1],
                    'z_abs': coords[i][2],
                    'is_wrist': i == 0
                })
            
            if visualize:
                # Draw landmarks on the image
                mp.solutions.drawing_utils.draw_landmarks(
                    image_rgb,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )
                plt.figure(figsize=(10, 10))
                plt.imshow(image_rgb)
                plt.title(f'Gesture Label: {label}')
                plt.axis('off')
                plt.show()
    
    return coordinates

# Cell 4 - Process Folder Function (Code cell)
def process_folder(base_folder, output_csv):
    \"\"\"Process all images in subfolders and save hand coordinates to CSV
    
    Args:
        base_folder (str): Path to the base folder containing gesture subfolders
        output_csv (str): Path to output CSV file
    \"\"\"
    all_coordinates = []
    
    # Get all subfolders (each represents a gesture class)
    gesture_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
    for gesture_folder in gesture_folders:
        gesture_label = os.path.basename(gesture_folder)
        print(f"Processing gesture: {gesture_label}")
        
        # Process all images in the gesture folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob(os.path.join(gesture_folder, ext)))
        
        for img_path in image_files:
            coords = process_single_image(img_path, gesture_label)
            all_coordinates.extend(coords)
    
    if all_coordinates:
        df = pd.DataFrame(all_coordinates)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Coordinates saved to {output_csv}")
        print(f"Total gestures processed: {len(df['label'].unique())}")
        print(f"Total images processed: {len(df['image'].unique())}")
        return df
    else:
        print("No hands detected in any images")
        return None

# Cell 5 - Analysis Function (Code cell)
def analyze_coordinates(df):
    \"\"\"Analyze the extracted hand coordinates\"\"\"
    if df is None or len(df) == 0:
        print("No data to analyze")
        return
    
    print("Dataset Summary:")
    print(f"Total number of points: {len(df)}")
    print(f"Number of unique images: {df['image'].nunique()}")
    print(f"Number of gesture classes: {df['label'].nunique()}")
    print("\nGesture class distribution:")
    print(df.groupby('label')['image'].nunique())
    
    # Plot coordinate distributions by gesture
    for coord in ['x_rel', 'y_rel', 'z_rel']:
        plt.figure(figsize=(15, 5))
        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            plt.hist(label_data[coord], alpha=0.5, label=label, bins=30)
        plt.title(f'{coord} Distribution by Gesture')
        plt.xlabel(coord)
        plt.ylabel('Count')
        plt.legend()
        plt.show()
    
    # Basic statistics by gesture
    print("\nCoordinate Statistics by Gesture:")
    for label in df['label'].unique():
        print(f"\nGesture: {label}")
        label_data = df[df['label'] == label]
        print(label_data[['x_rel', 'y_rel', 'z_rel']].describe())

# Cell 6 - Example Usage (Code cell)
# Example usage
# base_folder = "path/to/gesture/folders"  # Should contain subfolders for each gesture
# output_csv = "path/to/output/coordinates.csv"
# 
# # Process images
# df = process_folder(base_folder, output_csv)
# 
# # Analyze results
# if df is not None:
#     analyze_coordinates(df)
#
# # Visualize a single image with landmarks
# # image_path = "path/to/image.jpg"
# # label = "gesture_label"
# # process_single_image(image_path, label, visualize=True)

"""

print("Instructions for setting up the Jupyter notebook have been created.")
print("Please create a new Jupyter notebook and copy the code blocks as described above.") 