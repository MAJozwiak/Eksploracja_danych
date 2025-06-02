import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

def train(merged):
    X = merged.drop(columns=['arr_del15']).values
    y = merged['arr_del15'].values


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2
    )

    mse_test = model.evaluate(X_test, y_test, verbose=0)[0]
    rmse_test = np.sqrt(mse_test)
    print(f'Test RMSE: {rmse_test}')

    y_pred = model.predict(X_test).flatten()
    print(f'Test R2: {r2_score(y_test, y_pred)}')
