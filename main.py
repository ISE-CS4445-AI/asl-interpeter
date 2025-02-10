from tensorflow.keras.models import load_model
from hand_detection import HandDetector
from data_processor import DataProcessor

def main():
    # Example usage of hand detection
    model = load_model('hand_gesture_model.h5')
    detector = HandDetector(model=model)
    detector.detect_hands()


if __name__ == "__main__":
    main()