from flask import Flask, request, jsonify
import joblib
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.simplefilter("ignore", ConvergenceWarning)

app = Flask(__name)

# Load the preprocessing objects and the trained model
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
min_max_scaler = joblib.load('min_max_scaler.pkl')
linear_regression_model = joblib.load('demand.pkl')

@app.route('/predict', methods=['POST'])
def predict_demand():
    try:
        data = request.json
        input_data = {
            'PotSize': data['PotSize'],
            'Temperature': data['Temperature'],
            'Humidity': data['Humidity'],
            'Rainfall': data['Rainfall']
        }

        input_df = pd.DataFrame([input_data])

        encoded_data = one_hot_encoder.transform(input_df[['PotSize']])
        encoded_feature_names = one_hot_encoder.get_feature_names_out(input_df[['PotSize']].columns)

        numeric_columns = input_df[['Temperature', 'Humidity', 'Rainfall']]
        scaled_data = min_max_scaler.transform(numeric_columns)

        preprocessed_data = pd.concat([pd.DataFrame(encoded_data, columns=encoded_feature_names),
                                      pd.DataFrame(scaled_data, columns=numeric_columns.columns)], axis=1)

        prediction = int(linear_regression_model.predict(preprocessed_data))

        return jsonify({"DemandPrediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main':
    app.run(debug=True)
