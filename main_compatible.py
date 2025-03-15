import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp

# Disable eager execution to help with compatibility
tf.compat.v1.disable_eager_execution()

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model=None, class_names=None):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.model = model
        self.class_names = class_names

    def get_relative_coordinates(self, hand_landmarks):
        """Get coordinates relative to wrist, matching training data format"""
        coordinates = []
        wrist = hand_landmarks.landmark[0]
        
        # Get coordinates in same order as training
        for landmark in hand_landmarks.landmark:
            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z
            coordinates.append([x, y, z])
        
        return np.array(coordinates)

    def preprocess_coordinates(self, coords):
        """Preprocess coordinates to match model input requirements"""
        # Ensure coordinates are in the correct shape (21, 3)
        if coords.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {coords.shape}")
        
        # Reshape for LSTM model (samples, timesteps, features)
        coords_reshaped = coords.reshape(1, 21, 3)
        
        return coords_reshaped

    def detect_hands(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read video feed")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    try:
                        # Get coordinates in same format as training
                        coords = self.get_relative_coordinates(hand_landmarks)
                        
                        # Preprocess coordinates for model input
                        coords_processed = self.preprocess_coordinates(coords)
                        
                        # Make prediction
                        prediction = self.model.predict(coords_processed, verbose=0)
                        gesture_idx = np.argmax(prediction[0])
                        confidence = prediction[0][gesture_idx]
                        
                        # Get gesture label
                        if self.class_names is not None:
                            gesture = self.class_names[gesture_idx]
                        else:
                            gesture = f"Class {gesture_idx}"
                            
                        label = f"{gesture} ({confidence:.2f})"
                        
                        # Draw bounding box
                        h, w, _ = image.shape
                        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                        x1, x2 = int(min(x_coords)), int(max(x_coords))
                        y1, y2 = int(min(y_coords)), int(max(y_coords))
                        
                        # Add padding
                        padding = 20
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)
                        
                        # Draw box and label
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(image, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    except Exception as e:
                        print(f"Error processing hand: {str(e)}")
                        continue
            
            cv2.imshow('Hand Gesture Recognition', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

def load_model_with_compatibility(model_path):
    """Load model with compatibility fixes for different Keras versions"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try different approaches to load the model
    try:
        # Approach 1: Load with tf.keras.models.load_model with compile=False
        print("Trying to load model with compile=False...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e1:
        print(f"First approach failed: {str(e1)}")
        
        try:
            # Approach 2: Load with custom objects
            print("Trying to load model with custom objects...")
            custom_objects = {
                'InputLayer': tf.keras.layers.InputLayer
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print("Model loaded successfully with custom objects!")
            return model
        except Exception as e2:
            print(f"Second approach failed: {str(e2)}")
            
            try:
                # Approach 3: Convert to SavedModel format and reload
                print("Trying to convert to SavedModel format...")
                # Create a temporary directory for the SavedModel
                temp_dir = "temp_saved_model"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Use tf.saved_model.load_v2 to load the model
                model = tf.saved_model.load(model_path)
                print("Model loaded successfully with tf.saved_model.load!")
                return model
            except Exception as e3:
                print(f"Third approach failed: {str(e3)}")
                
                # Final approach: Use h5py to extract weights and rebuild model
                print("Trying to rebuild model from scratch...")
                try:
                    # Create a simple LSTM model with the same architecture
                    input_shape = (21, 3)  # 21 landmarks with 3 coordinates each
                    num_classes = 29  # Assuming 29 classes (A-Z, del, nothing, space)
                    
                    model = tf.keras.Sequential([
                        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(64),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(num_classes, activation='softmax')
                    ])
                    
                    # Compile the model
                    model.compile(
                        loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                    )
                    
                    # Try to load weights
                    try:
                        model.load_weights(model_path)
                        print("Weights loaded successfully!")
                    except Exception as e:
                        print(f"Failed to load weights: {str(e)}")
                        print("Using model with random weights. Performance will be poor.")
                    
                    return model
                except Exception as e4:
                    print(f"All approaches failed: {str(e4)}")
                    raise Exception("Could not load model with any method")

def load_class_names(class_names_path):
    """Load class names from numpy file"""
    print(f"Loading class names from: {class_names_path}")
    
    if not os.path.exists(class_names_path):
        print(f"Warning: Class names file not found: {class_names_path}")
        return None
    
    try:
        class_names = np.load(class_names_path, allow_pickle=True)
        print(f"Loaded {len(class_names)} classes: {class_names}")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return None

def main():
    try:
        # Set paths
        model_path = 'models/hand_gesture_model_lstm.h5'
        class_names_path = 'models/label_encoder_classes.npy'
        
        # Load model with compatibility fixes
        model = load_model_with_compatibility(model_path)
        
        # Load class names
        class_names = load_class_names(class_names_path)
        
        # Initialize hand detector
        detector = HandDetector(model=model, class_names=class_names)
        
        # Start detection
        print("Starting hand gesture recognition. Press 'q' to quit.")
        detector.detect_hands()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 