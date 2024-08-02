import pandas as pd
from pycaret.classification import *
import extract_features

# Load data and set up environment
data = pd.read_csv("extracted_features3.csv")
clf1 = setup(data, target='User_Label')
loaded_model = load_model("model94")

def predict_from_csv(csv_path):
    """
    This function takes the path of a CSV file as input, extracts features from it,
    and uses a pre-trained model to make predictions.
    
    """

    # Extract features from the new CSV file
    supposed_prediction = extract_features.extract_live(csv_path)
    new_data = pd.read_csv("live_features.csv")

    # Test model
    predictions_new_data = predict_model(loaded_model, data=new_data)

    print("Predicted label from new data: " + str(predictions_new_data['prediction_label'][0]))
    print("Supposed prediction: " + str(supposed_prediction))

    # Return True if the prediction matches the supposed prediction, otherwise False
    return predictions_new_data['prediction_label'][0] == supposed_prediction

