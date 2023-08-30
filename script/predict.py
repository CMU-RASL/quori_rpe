import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from collections import defaultdict
import math
import glob
import matplotlib.pyplot as plt
import os

class AVG:
    def __init__(self, y_train):
        self.avg = np.average(y_train)

    def predict(self, X_test):
        return [self.avg for _ in range(X_test.shape[0])]

# def get_train_test_data(csv_dir):
#     csv_files = glob.glob(csv_dir + "/*")
#     csv_files.sort()

#     df_list = []
    
#     for file_path in csv_files:
#         df = pd.read_csv(file_path).dropna()
#         initial_value = df['ds'][0]
#         df['ds'] = df['ds'] - initial_value
#         start_index = file_path.rfind("/")+1
#         end_index = file_path.rfind(".csv")
#         df_list.append((file_path[start_index:end_index], df))

#     train_size = math.ceil(0.8 * len(df_list))
#     train_data = pd.concat([df[1] for df in df_list[:train_size]], ignore_index=True)
#     test_data = pd.concat([df[1] for df in df_list[train_size:]], ignore_index=True)
    
#     return train_data, test_data, df_list[train_size:]

def get_train_test_data(csv_dir, num_folds=5):
    csv_files = glob.glob(csv_dir + "/*")
    csv_files.sort()

    df_list = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path).dropna()
        initial_value = df['ds'][0]
        df['ds'] = df['ds'] - initial_value
        start_index = file_path.rfind("/") + 1
        end_index = file_path.rfind(".csv")
        df_list.append((file_path[start_index:end_index], df))

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_data = []

    for train_indices, test_indices in kf.split(df_list):
        train_data = pd.concat([df_list[i][1] for i in train_indices], ignore_index=True)
        test_data = pd.concat([df_list[i][1] for i in test_indices], ignore_index=True)
        fold_data.append((train_data, test_data, [df_list[i] for i in test_indices]))
    
    return fold_data

def add_prophet_regressors(model):
    regressors = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']
    for regressor in regressors:
        model.add_regressor(regressor)

total_feature_weights = defaultdict(float)

def predict(model_name, X_train, y_train, X_test):
    if model_name == "ARMA":
        model = SARIMAX(y_train, exog=X_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0))
        model = model.fit()
        y_pred = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test).values

    elif model_name == "LR":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_name == "LR":
            feature_weights = model.coef_
            print("Feature Weights:", feature_weights)

            features_and_weights = list(zip(features, feature_weights))
            features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Sorted Features and Weights:")
            for feature, weight in features_and_weights:
                print(f"Feature: {feature}, Weight: {weight}")
                total_feature_weights[feature] += abs(weight)

    elif model_name == "Prophet":
        model = Prophet(changepoint_prior_scale=1.0, seasonality_prior_scale=1.0)
        add_prophet_regressors(model)

        train_data = pd.concat([X_train, y_train], axis=1)
        prophet_train_data = train_data[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S', pred_cat]]
        prophet_train_data.rename(columns={'ds': 'ds', pred_cat: 'y', 'AU01': 'AU01', 'AU02': 'AU02', 'AU04': 'AU04', 'AU05': 'AU05', 'AU06': 'AU06', 'AU07': 'AU07', 'AU09': 'AU09', 'AU10': 'AU10', 'AU12': 'AU12', 'AU14': 'AU14', 'AU15': 'AU15', 'AU17': 'AU17', 'AU20': 'AU20', 'AU23': 'AU23', 'AU25': 'AU25', 'AU26': 'AU26', 'AU28': 'AU28', 'AU45': 'AU45', 'E0': 'E0', 'E1': 'E1', 'E2': 'E2', 'E3': 'E3', 'E4': 'E4', 'S': 'S'}, inplace=True)
        model.fit(prophet_train_data)

        y_pred = model.predict(X_test)['yhat'].values

    elif model_name == "RFR":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        

    elif model_name == "SVR":
        model = SVR(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_name == "AVG":
        model = AVG(y_train)
        y_pred = model.predict(X_test)
    
    else:
        exit("The model type is not supported.")

    return model, y_pred

def plot(timestamps, actual, pred, output_plot_path):

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, actual, label='actual', marker='o')
    plt.plot(timestamps, pred, label='predicted', marker='x')

    plt.xlabel('Timestamp', fontsize=18)
    plt.ylabel(pred_cat, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.legend()

    plt.legend()
    plt.grid(True)
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_plot_path)
    plt.close()
    
def plot_results(model, test_dfs, output_plot_dir):
    for test_df_info in test_dfs:
        test_df = test_df_info[1]
        test_data = test_df[features]

        pred = model.predict(test_data)
        actual = test_df[pred_cat].values

        timestamps = test_df['ds']

        plot(timestamps, actual, pred, output_plot_dir + "/" + test_df_info[0] + ".png")

def reconstruct_rpes(model, test_dfs, output_plot_dir):
    for test_df_info in test_dfs:
        test_df = test_df_info[1]

        timestamps = list(test_df['ds'].values)
        rpes = list(test_df['rpe'].values)

        slopes = model.predict(test_df[features])

        predicted_rpes = [1.0]
        for i in range(1, len(slopes)):

            slope = slopes[i]

            diff = timestamps[i] - timestamps[i-1]

            if diff >= 30.0:
                slope = -0.041

            predicted_rpe = predicted_rpes[i-1] + diff * slope

            val = min(max(predicted_rpe, 1.0), 10)
            predicted_rpes.append(val)

        plot(timestamps, rpes, predicted_rpes, output_plot_dir + "/" + test_df_info[0] + "_rpe.png")

    mse = mean_squared_error(rpes, predicted_rpes)
    mae = mean_absolute_error(rpes, predicted_rpes)

    return mse, mae
 
if __name__ == "__main__":
    ###############################
    # Set parameters
    pred_cat = "slope" # rpe or slope
    use_time = True
    model_name = "LR"
    output_plot_dir = "/home/kevin/Documents/quori/src/openface2_ros/results/"
    data_dir  = "/home/kevin/Documents/quori/src/openface2_ros/features"
    ###############################

    # Generate paths
    time = 'time_' if use_time else "no_time_"
    output_plot_dir = output_plot_dir + "lr_" + time + pred_cat 

    if not os.path.exists(output_plot_dir):
        os.mkdir(output_plot_dir)

    # Get train and test data
    K_fold_splits = get_train_test_data(data_dir)

    # Select features
    if use_time:
        features = ['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']
    else:
        features = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']

    # features = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'E0', 'E1', 'E2', 'E3', 'E4', 'S'] 

    # Feature: AU45, Weight: 0.01961892176620879
    # Feature: AU25, Weight: 0.018198767054278215
    # Feature: E1, Weight: 0.016155241881551773
    # Feature: AU02, Weight: 0.011682142611110908
    # Feature: AU09, Weight: 0.010704460677879004
    # Feature: AU10, Weight: 0.010437691577937211
    # Feature: AU17, Weight: 0.01009585527271453
    # Feature: AU12, Weight: 0.009939500168477245
    # Feature: AU23, Weight: 0.009902186518827422
    # Feature: E2, Weight: 0.009172700170605189
    # Feature: AU14, Weight: 0.008556542537990158
    # Feature: AU15, Weight: 0.007513140263428921
    # Feature: AU26, Weight: 0.0070212852811371235
    # Feature: S, Weight: 0.00654103937848754
    features = ["AU45", "AU25", "E1", "AU02", "AU09", "AU10"] # 1.49 -> 1.53 mse

    mse_scores = []
    mae_scores = []

    if pred_cat == "slope":
        recon_mse_scores = []
        recon_mae_scores = []

    for train_df, test_df, test_dfs in K_fold_splits:
        X_train = train_df[features]
        y_train = train_df[pred_cat]

        X_test = test_df[features]
        y_test = test_df[pred_cat]

        # Make predictions on the test data
        model, y_pred = predict(model_name, X_train, y_train, X_test)

        # Calculate the mean squared error to evaluate the model performance
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'{pred_cat} MSE: {mse}')
        print(f'{pred_cat} MAE: {mae}')

        mse_scores.append(mse)
        mae_scores.append(mae)

        # Plot predictions
        plot_results(model, test_dfs, output_plot_dir)

        # Plot rpes if predicting slopes
        if pred_cat == 'slope':
            recon_mse, recon_mae = reconstruct_rpes(model, test_dfs, output_plot_dir)
            
            recon_mse_scores.append(recon_mse)
            recon_mae_scores.append(recon_mae)

            print("RPE MSE:", recon_mse)
            print("RPE MAE:", recon_mae)

    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    print(f'Average {pred_cat} MSE: {avg_mse}')
    print(f'Average {pred_cat} MAE: {avg_mae}')

    if pred_cat == "slope":
        avg_mse = sum(recon_mse_scores) / len(recon_mse_scores)
        avg_mae = sum(recon_mae_scores) / len(recon_mae_scores)
        print(f'Average rpe MSE: {avg_mse}')
        print(f'Average rpe MAE: {avg_mae}')

    sorted_items = sorted(total_feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)

    # Print the sorted key-value pairs
    for key, value in sorted_items:
        print(f"Feature: {key}, Weight: {value}")