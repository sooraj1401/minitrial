import pandas as pd
import joblib

# Load Model
model = joblib.load('models/traffic_model.pkl')

def predict_signal_duration(hour, vehicle_count, pedestrian_count):
    total_count = vehicle_count + pedestrian_count
    prediction = model.predict([[hour, total_count]])
    return int(prediction[0])

if __name__ == "__main__":
    # Example Data
    hour = 17
    vehicle_count = 200
    pedestrian_count = 25

    predicted_duration = predict_signal_duration(hour, vehicle_count, pedestrian_count)
    print(f"Predicted Signal Duration: {predicted_duration} seconds")
