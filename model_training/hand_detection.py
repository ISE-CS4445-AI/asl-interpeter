import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model=None):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.model = model

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
                    
                    # Calculate bounding box
                    h, w, _ = image.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding to bounding box
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    # Get coordinates for prediction
                    coords = []
                    wrist = hand_landmarks.landmark[0]
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x - wrist.x
                        y = landmark.y - wrist.y
                        z = landmark.z - wrist.z
                        coords.extend([x, y, z])
                    
                    # Make prediction
                    coords = np.array(coords).reshape(1, -1)
                    prediction = self.model.predict(coords, verbose=0)[0][0]
                    gesture = "Closed Fist" if prediction < 0.5 else "Finger Circle"
                    confidence = prediction if prediction >= 0.5 else 1 - prediction
                    label = f"{gesture} ({confidence:.2f})"
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(image, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 255, 0), -1)
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.imshow('Hand Gesture Recognition', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

    def get_relative_coordinates(self, hand_landmarks, image_width, image_height):
        coordinates = []
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z
        
        for landmark in hand_landmarks.landmark:
            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z
            coordinates.append((x, y, z))
        return coordinates
 