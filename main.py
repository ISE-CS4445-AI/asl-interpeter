from tensorflow.keras.models import load_model
from model_training.hand_detection import HandDetector
from preprocessing.data_processor import DataProcessor

def main():
    model = load_model('models/hand_gesture_model_lstm.h5')
    detector = HandDetector(model=model)
    detector.detect_hands()

if __name__ == "__main__":
    main()