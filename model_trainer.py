import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class GestureModelTrainer:
    def __init__(self):
        self.model = None

    def prepare_data(self, finger_circle_csv, closed_fist_csv):
        finger_circle_data = pd.read_csv(finger_circle_csv)
        closed_fist_data = pd.read_csv(closed_fist_csv)

        finger_circle_data['label'] = 1
        closed_fist_data['label'] = 0

        all_data = pd.concat([finger_circle_data, closed_fist_data])
        grouped = all_data.groupby('image')
        
        X = []
        y = []
        
        for name, group in grouped:
            group = group.sort_values('hand_point')
            coords = group[['x', 'y', 'z']].values.flatten()
            X.append(coords)
            y.append(group['label'].iloc[0])

        return np.array(X), np.array(y)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(63,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

        history = self.model.fit(X_train, y_train,
                               epochs=50,
                               batch_size=32,
                               validation_split=0.2,
                               verbose=1)

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_accuracy:.4f}")

    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved as '{path}'") 