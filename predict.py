import numpy as np
import pickle
from lstm import LstmNetwork
from sklearn.preprocessing import MinMaxScaler

def predict_next(model, input_values):
    #model.reset_state()
    model.x_list_clear()
    model.x_list_add(input_values)  # Ensure it's a 1D list/array
    return model.get_output()

def load_model(model):
    """Loads an LSTM model from a pickle file."""
    with open(model, "rb") as f:
        saved_parameters = pickle.load(f)
    return LstmNetwork(saved_parameters)

def scale_features(min_value, max_value):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit(np.array([[min_value], [max_value]]))

def predict_using_observed_values(model, scaler, obs_values):
    observed_values = np.array(obs_values)

    # Normalize the observed values
    observed_values_scaled = scaler.transform(observed_values.reshape(-1, 1)).flatten()

    predictions = []

    for new_value in observed_values_scaled:
        predicted_value = predict_next(model, np.array([new_value]))
        predictions.append(predicted_value)
    
    # Convert predictions back to the original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predicted_original_scale = scaler.inverse_transform(predictions).flatten()
    
    return predicted_original_scale

def predict_by_rolling_forward(model, scaler, last_known_obs_values, future_time_steps):
    initial_values = np.array(last_known_obs_values)

    # Normalize the last known observed values
    initial_values_scaled = scaler.transform(initial_values.reshape(-1, 1)).flatten()

    predictions = []
    values_to_use = list(initial_values_scaled)
    idx = 1

    for i in range(future_time_steps):
        if i < len(values_to_use):
            current_value = values_to_use[i]
        else:
            current_value = predictions[-1]

        # Predict the next step
        predicted_value = predict_next(model, np.array([current_value]))  # Pass only a single value

        # Extract the value correctly
        new_value_scaled = predicted_value.flatten()[0]
        predictions.append(new_value_scaled)

    # Convert predictions back to the original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predicted_original_scale = scaler.inverse_transform(predictions).flatten()

    return predicted_original_scale


if __name__ == "__main__":
    lstm = load_model("lstm_model_003.pkl")

    min_value = 20.84
    max_value = 35.5

    scaler = scale_features(min_value, max_value)

    observed_values = [25.19, 30.29, 31.6, 30.09, 27.33, 26.8, 25.73, 24.43]
    initial_values = observed_values[:1]  

    predicted_values_via_obs = predict_using_observed_values(lstm, scaler, observed_values)
    predicted_values_via_rolling = predict_by_rolling_forward(lstm, scaler, initial_values, 8)

    print("                       Observed values: ", *observed_values)
    print("Predicted values using observed values: ", *np.round(predicted_values_via_obs, 2))
    print("    Predicted values by rolling foward: ", *np.round(predicted_values_via_rolling, 2))
    