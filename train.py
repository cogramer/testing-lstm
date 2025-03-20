import sqlite3
import numpy as np
import pickle
from lstm import LstmParameters, LstmNetwork
from sklearn.preprocessing import MinMaxScaler

class MSELoss:
    """ Standard Mean Squared Error (MSE) loss for single-value outputs. """
    @staticmethod
    def loss(pred, label):
        return (pred - label) ** 2  # Scalar loss
    
    @staticmethod
    def bottom_diff(pred, label):
        return 2 * (pred - label)  # Gradient for backpropagation


def load_weather_data(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
        SELECT temperature FROM {table_name}
        WHERE DATE(datetime) >= '2025-01-12' AND DATE(datetime) <= '2025-01-25';
    """

    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    
    return np.array(data, dtype=np.float32).flatten()  # Convert to 1D array

if __name__ == "__main__":
    # Load and preprocess data
    db_path = ''
    table_name = ''
    data = load_weather_data(db_path, table_name)

    min_value = data.min()
    max_value = data.max()
    print(f"Min value: {min_value}\nMax value: {max_value}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()  # Normalize

    lstm = LstmNetwork(LstmParameters(memory_cell_count=100, input_dimensions=1))
    loss_layer = MSELoss()

    # Training loop
    for epoch in range(300):
        total_loss = 0

        for i in range(len(scaled_data) - 1):  # Predict one step ahead
            x = scaled_data[i]   # Current value
            y = scaled_data[i+1] # Next time step

            lstm.x_list_clear()
            lstm.x_list_add([x])  # Single input

            total_loss += lstm.y_list_is([y], loss_layer)  # Single target value
        
        lstm.lstm_parameters.apply_derivatives(learning_rate=0.01)

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    input()

    with open("lstm_model_001.pkl", "wb") as f:
        pickle.dump(lstm.lstm_parameters, f)
    
    print("Model parameters saved")

        