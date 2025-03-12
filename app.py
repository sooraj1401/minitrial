from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load Model
try:
    model = joblib.load('models/traffic_model.pkl')
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Validate if data exists
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Extract parameters from JSON request
    hour = data.get('hour')
    vehicle_count = data.get('vehicle_count')
    pedestrian_count = data.get('pedestrian_count')

    # Validate input fields
    if hour is None or vehicle_count is None or pedestrian_count is None:
        return jsonify({'error': 'Missing required fields: hour, vehicle_count, pedestrian_count'}), 400

    if not isinstance(hour, int) or not isinstance(vehicle_count, int) or not isinstance(pedestrian_count, int):
        return jsonify({'error': 'All inputs must be integers'}), 400

    if model is None:
        return jsonify({'error': 'Model could not be loaded'}), 500

    # Compute total count
    total_count = vehicle_count + pedestrian_count
    
    # Make prediction
    try:
        prediction = model.predict([[hour, total_count]])
        return jsonify({'predicted_signal_duration': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

