import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

def train_and_predict(company, feature):
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[df['Name'] == company]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    data = df[[feature]].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    seq_length = 60

    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model_path = f'models/{company}_{feature}_model.h5'

    if not os.path.exists(model_path):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
        os.makedirs('models', exist_ok=True)
        model.save(model_path)
    else:
        model = load_model(model_path)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    dates = df['date'].values[-len(predictions):]

    # Additional Analysis
    df['MA50'] = df[feature].rolling(50).mean()
    df['Volatility'] = df[feature].rolling(50).std()

    return {
        'dates': [str(d) for d in dates],
        'actual': actual.flatten().tolist(),
        'predicted': predictions.flatten().tolist(),
        'ma50': df['MA50'].dropna().tolist(),
        'volatility': df['Volatility'].dropna().tolist(),
        'volume': df['volume'].tolist(),
        'full_dates': [str(d) for d in df['date'].values]
    }
