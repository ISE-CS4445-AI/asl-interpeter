import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py

def inspect_h5_file(file_path):
    """Inspect the contents of an H5 file without loading the model"""
    print(f"Inspecting H5 file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Print the keys at the root level
            print("Root level keys:", list(f.keys()))
            
            # Check if it's a Keras model
            if 'model_weights' in f:
                print("This appears to be a Keras model")
                
                # Print model config if available
                if 'model_config' in f.attrs:
                    config = f.attrs['model_config']
                    print("Model config available")
                    
                    # Try to decode the config
                    try:
                        if isinstance(config, bytes):
                            config_str = config.decode('utf-8')
                            print("Model config type: bytes (decoded to string)")
                        else:
                            config_str = str(config)
                            print("Model config type:", type(config))
                        
                        # Print a snippet of the config
                        print("Config snippet:", config_str[:200] + "...")
                    except Exception as e:
                        print(f"Error decoding config: {str(e)}")
                else:
                    print("No model_config attribute found")
                
                # Check keras version
                if 'keras_version' in f.attrs:
                    keras_version = f.attrs['keras_version']
                    if isinstance(keras_version, bytes):
                        keras_version = keras_version.decode('utf-8')
                    print(f"Keras version: {keras_version}")
                
                # Check backend
                if 'backend' in f.attrs:
                    backend = f.attrs['backend']
                    if isinstance(backend, bytes):
                        backend = backend.decode('utf-8')
                    print(f"Backend: {backend}")
            else:
                print("This does not appear to be a standard Keras model")
    
    except Exception as e:
        print(f"Error inspecting H5 file: {str(e)}")

def try_load_model_with_custom_objects(file_path):
    """Try to load the model with custom objects to handle compatibility issues"""
    print(f"\nAttempting to load model with custom objects: {file_path}")
    
    try:
        # Define custom objects to handle compatibility issues
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer
        }
        
        model = load_model(file_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully with custom objects!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {str(e)}")
        return None

def check_class_names(file_path):
    """Check if class names file exists and can be loaded"""
    print(f"\nChecking class names file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    try:
        class_names = np.load(file_path, allow_pickle=True)
        print(f"Class names loaded successfully: {class_names}")
        print(f"Number of classes: {len(class_names)}")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return None

if __name__ == "__main__":
    model_path = 'models/hand_gesture_model_lstm.h5'
    class_names_path = 'models/label_encoder_classes.npy'
    
    # Inspect the H5 file
    inspect_h5_file(model_path)
    
    # Try to load the model with custom objects
    model = try_load_model_with_custom_objects(model_path)
    
    # Check class names
    class_names = check_class_names(class_names_path)
    
    # Print TensorFlow and Keras versions
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}") 