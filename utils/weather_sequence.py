import numpy as np
import joblib

# Load saved scaler
scaler = joblib.load("models/weather_scaler.pkl")


def prepare_weather_sequence(weather_data):

    data = np.array(weather_data)

    # Scale data
    scaled = scaler.transform(data)

    # Reshape for LSTM
    sequence = scaled.reshape(1, 6, 3)

    return sequence