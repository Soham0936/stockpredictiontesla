# Tesla Stock Price Prediction using LSTM

## 📌 Project Overview
This project aims to predict Tesla's stock prices using a **Long Short-Term Memory (LSTM) neural network**. We utilize historical stock data to train our model, enabling it to make accurate future predictions. The dataset includes daily stock prices with attributes like **Open, High, Low, Close, Adjusted Close, and Volume**.

## 🚀 Features
- **Data Preprocessing**: Cleans and normalizes historical Tesla stock prices.
- **Visualization**: Plots trends in stock prices.
- **Train-Test Split**: Uses an 80-20 split for training and evaluation.
- **LSTM Model**: Implements an advanced deep learning model for sequential data.
- **Evaluation**: Uses RMSE and loss visualization to assess performance.

## 📂 Dataset
The dataset is sourced from **Yahoo Finance** and contains Tesla stock prices from **2010 to 2020**. The key columns include:
- **Date**: Trading date.
- **Open**: Opening stock price.
- **High**: Highest stock price.
- **Low**: Lowest stock price.
- **Close**: Closing stock price.
- **Adj Close**: Adjusted closing price.
- **Volume**: Trading volume.

## 📊 Data Preprocessing
- Convert the **Date** column to `datetime` format.
- Extract only the **Date** and **Close** price for modeling.
- Normalize data using **MinMaxScaler**.
- Perform an **80-20 train-test split**.

## 📌 Installation & Dependencies
To run this project, install the required libraries:
```bash
pip install numpy pandas tensorflow matplotlib scikit-learn
```

## 🔧 Usage
1. **Load Data**
```python
import pandas as pd

data = pd.read_csv("TSLA.csv")
data.head()
```

2. **Preprocess Data**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data[['Close']])
```

3. **Train LSTM Model**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(100, 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

4. **Make Predictions**
```python
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
```

## 📉 Results & Visualization
- **Plot actual vs. predicted stock prices**
```python
import matplotlib.pyplot as plt

plt.plot(y_test, color='blue', label='Actual Tesla Stock Price')
plt.plot(y_pred, color='red', label='Predicted Tesla Stock Price')
plt.legend()
plt.show()
```

## 📌 Future Improvements
- Add **hyperparameter tuning**.
- Implement **GRU & Transformer models**.
- Integrate **real-time stock price prediction**.

## 📜 License
This project is open-source and available under the **MIT License**.

## 📬 Contact
If you have any questions or suggestions, feel free to connect with me on **[LinkedIn](https://linkedin.com/in/soham-das)** or via **email: mynameissoham1234@gmail.com**.

