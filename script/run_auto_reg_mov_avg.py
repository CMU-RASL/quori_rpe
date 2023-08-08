import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import glob
import math
import os

def get_train_test_data(csv_dir):
    csv_files = glob.glob(csv_dir + "/*")
    csv_files.sort()

    df_list = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path).dropna()
        start_index = file_path.rfind("/")+1
        end_index = file_path.rfind(".csv")
        df_list.append((file_path[start_index:end_index], df))

    train_size = math.ceil(0.8 * len(df_list))
    train_data = pd.concat([df[1] for df in df_list[:train_size]], ignore_index=True)
    test_data = pd.concat([df[1] for df in df_list[train_size:]], ignore_index=True)
    
    return train_data, test_data, df_list[train_size:]


def plot_results(model, test_dfs, output_plot_dir):
    for test_df_info in test_dfs:
        test_df = test_df_info[1]
        test_data = test_df[features]

        actual = test_df[pred_cat].values

        pred = model.predict(start=len(y_train), end=len(y_train) + len(actual) - 1, exog=test_data).values

        timestamps = test_df['ds']

        plt.figure(figsize=(8, 6))
        plt.plot(timestamps, actual, label='actual', marker='o')
        plt.plot(timestamps, pred, label='predicted', marker='x')

        plt.xlabel('timestamps')
        plt.ylabel(pred_cat)

        plt.legend()
        plt.grid(True)

        plt.savefig(output_plot_dir + "/" + test_df_info[0] + ".png")

# Set parameters
pred_cat = "slope" # rpe or slope
use_time = False
time = 'time_' if use_time else "no_time_"
output_plot_dir = "/home/kevin/Documents/quori/src/openface2_ros/results/arma_" + time + pred_cat 

if not os.path.exists(output_plot_dir):
    os.mkdir(output_plot_dir)

# Get train and test data
if use_time:
    features = ['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']
else:
    features = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']

train_df, test_df, test_dfs = get_train_test_data("/home/kevin/Documents/quori/src/openface2_ros/features")
X_train = train_df[features]
y_train = train_df[pred_cat]

X_test = test_df[features]
y_test = test_df[pred_cat]

y_train = y_train
y_test = y_test


# Initialize the ARMA model
arma_model = SARIMAX(y_train, exog=X_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0))

# Fit the ARMA model on the training data
arma_model_fit = arma_model.fit()

# Make predictions on the test data
y_pred = arma_model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test).values

# Calculate the mean squared error to evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'{pred_cat} MSE: {mse}')
print(f'{pred_cat} MAE: {mae}')

# Plot predictions
plot_results(arma_model_fit, test_dfs, output_plot_dir)

# Plot rpes if predicting slopes
if pred_cat == 'slope':
    all_preds = []
    all_actuals = []

    for test_df_info in test_dfs:
        test_df = test_df_info[1]

        timestamps = test_df['ds']
        rpes = test_df['rpe']

        slopes = arma_model_fit.predict(start=len(y_train), end=len(y_train) + len(rpes) - 1, exog=test_df[features]).values

        predicted_rpes = [1.0]
        for i in range(1, len(slopes)):

            slope = slopes[i]

            diff = timestamps[i] - timestamps[i-1]

            if diff >= 30.0:
                slope = -0.041

            predicted_rpe = predicted_rpes[i-1] + diff * slope
            if predicted_rpe < 0:
                dt = predicted_rpe / 0.041
                timestamps.insert(i, timestamps[i] + dt)
                predicted_rpes.append(0)

            val = min(max(predicted_rpe, 1.0), 10)
            predicted_rpes.append(val)

        all_actuals.extend(rpes)
        all_preds.extend(predicted_rpes)

        plt.figure(figsize=(8, 6))

        plt.plot(timestamps, rpes, label='actual', marker='o')
        plt.plot(timestamps, predicted_rpes, label='predicted', marker='x')

        plt.xlabel('timestamps')
        plt.ylabel('rpes')

        plt.legend()

        plt.grid(True)
        plt.savefig(output_plot_dir + "/" + test_df_info[0] + "_rpe.png")

    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)

    print("RPE MSE:", mse)
    print("RPE MAE:", mae)

# rpe (time)
# rpe MSE: 5.3848385837159585
# rpe MAE: 2.0719416224415514

# rpe (no time)
# rpe MSE: 8.176590582630348
# rpe MAE: 2.4394899432503907

# slope (time)
# slope MSE: 0.0008075795995629589
# slope MAE: 0.023568323812002787
# RPE MSE: 2.9640450481336877
# RPE MAE: 1.3275612649182855

# slope (no time)
# slope MSE: 0.002695509081450842
# slope MAE: 0.04591200058232923
# RPE MSE: 4.318360045804456
# RPE MAE: 1.4558448477111745