import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
import signal
import sys
import pyttsx3

ROWS_PER_FRAME = 543  # number of landmarks per frame

# Load relevant data subset
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter("./model.tflite")
interpreter.allocate_tensors()
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

# Load training data and create dictionaries for label translation
train = pd.read_csv('./input/asl-signs/train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# Get prediction from the model
def get_prediction(prediction_fn, pq_file):
    try:
        xyz_np = load_relevant_data_subset(pq_file)
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        # Clean the output.parquet file
        try:
            if os.path.exists(pq_file):
                os.remove(pq_file)
        except Exception as e_remove:
            print(f"An error occurred while removing the file: {e_remove}")
        return
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    sign = ORD2SIGN[pred]
    pred_conf = prediction['outputs'][pred]
    print(f'Predicted Sign {sign} [{pred}] with confidence {pred_conf:0.4}')
    return sign

# Define a handler for keyboard interrupt to ensure clean exit
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def speak_sign(sign):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Convert the sign string to speech
    engine.say(sign)
    
    # Wait for the speech to complete
    engine.runAndWait()

# Main loop to keep checking for new data
def main_loop():
    pq_file = 'output.parquet'
    last_mod_time = None
    
    while True:
        try:
            if os.path.exists(pq_file):
                current_mod_time = os.path.getmtime(pq_file)
                if last_mod_time is None or current_mod_time != last_mod_time:
                    last_mod_time = current_mod_time
                    sign=get_prediction(prediction_fn, pq_file)
                    speak_sign(sign)
            time.sleep(1)  # Delay between checks to avoid excessive CPU usage
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main_loop()
