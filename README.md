# ✈️ Boston Flight Delay Prediction
Flight Delay Data dataset link - https://www.kaggle.com/datasets/sriharshaeedala/airline-delay?resource=download

Boston Weather 2013-2023 dataset link - https://www.kaggle.com/datasets/swaroopmeher/boston-weather-2013-2023

This project predicts the number of flights delayed by more than 15 minutes using a neural network. The dataset is created by merging flight statistics with weather data. All features are standardized to improve model training.

The neural network consists of dense layers with L2 regularization and dropout layers, trained with early stopping to prevent overfitting. The model is evaluated using RMSE and R² metrics, achieving approximately 0.90 R² on the test set.

Future improvements may include adding additional features, hyperparameter tuning and experimenting with different neural network architectures.