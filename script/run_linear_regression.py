import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import glob
import matplotlib.pyplot as plt
import os

def get_train_test_data(csv_dir):
    csv_files = glob.glob(csv_dir + "/*")
    csv_files.sort()

    df_list = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path).dropna()
        initial_value = df['ds'][0]
        df['ds'] = df['ds'] - initial_value
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

        pred = model.predict(test_data)
        actual = test_df[pred_cat].values

        timestamps = test_df['ds']

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
        

        plt.savefig(output_plot_dir + "/" + test_df_info[0] + ".png")
        plt.close()

# Set parameters
pred_cat = "slope" # rpe or slope
use_time = False
time = 'time_' if use_time else "no_time_"
output_plot_dir = "/home/kevin/Documents/quori/src/openface2_ros/results/lr_" + time + pred_cat 

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

# # Split the data into training and testing sets
# # Since it's time series data, we should not shuffle the data.
# # You can use a TimeSeriesSplit to perform cross-validation if needed.
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)


# data = pd.read_csv("/home/kevin/Documents/quori/src/openface2_ros/features/8.csv")
# features = data[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']]
# target = data['y']
# X_test = features
# y_test = target

# Initialize the Linear Regression model
lr_regressor = LinearRegression()

# Train the model on the training data
lr_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr_regressor.predict(X_test)

# Calculate the mean squared error to evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'{pred_cat} MSE: {mse}')
print(f'{pred_cat} MAE: {mae}')

# Plot predictions
plot_results(lr_regressor, test_dfs, output_plot_dir)

# Plot rpes if predicting slopes
if pred_cat == 'slope':
    # all_preds = []
    # all_actuals = []

    for test_df_info in test_dfs:
        test_df = test_df_info[1]

        timestamps = list(test_df['ds'].values)
        # timestamps_pred = []
        # timestamps_actual = []
        rpes = list(test_df['rpe'].values)

        slopes = lr_regressor.predict(test_df[features])

        predicted_rpes = [1.0]
        # actual_rpes = [1.0]
        for i in range(1, len(slopes)):

            slope = slopes[i]

            diff = timestamps[i] - timestamps[i-1]

            if diff >= 30.0:
                slope = -0.041

            predicted_rpe = predicted_rpes[i-1] + diff * slope
            # if predicted_rpe < 0:
            #     dt = (predicted_rpe-1) / 0.041
            #     timestamps_pred.append(timestamps[i-1] + dt)
            #     predicted_rpes.append(0)

                # if rpes[i] == 1:
                #     dt = (rpes[i-1]-1) / 0.041
                #     timestamps_actual.append(timestamps_actual[i-1] + dt)
                #     actual_rpes.append(0)
                # else:
                #     timestamps_actual.append(timestamps_actual[i-1])
                #     actual_rpes.append(rpes[i-1])

            val = min(max(predicted_rpe, 1.0), 10)
            predicted_rpes.append(val)
            # timestamps.append(timestamps[i])

            # actual_rpes.append(rpes[i])
            # timestamps_actual.append(timestamps[i])



        # # all_actuals.extend(actual_rpes)
        # all_preds.extend(predicted_rpes)

        plt.figure(figsize=(10, 6))

        plt.plot(timestamps, rpes, label='actual', marker='o')
        plt.plot(timestamps, predicted_rpes, label='predicted', marker='x')

        plt.xlabel('Timestamp', fontsize=18)
        plt.ylabel('RPE', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.legend()
        plt.tight_layout()

        plt.grid(True)
        plt.savefig(output_plot_dir + "/" + test_df_info[0] + "_rpe.png")

    mse = mean_squared_error(rpes, predicted_rpes)
    mae = mean_absolute_error(rpes, predicted_rpes)

    print("RPE MSE:", mse)
    print("RPE MAE:", mae)


# rpe (time)
# rpe MSE: 5.400135421325342
# rpe MAE: 2.027074724691093

# rpe (no time)
# rpe MSE: 5.680237918674875
# rpe MAE: 2.114377320670582

# slope (time)
# slope MSE: 0.0007429399228828578
# slope MAE: 0.02263100633954379
# RPE MSE: 1.8246464855819358
# RPE MAE: 1.1915604434018505

# slope (no time)
# slope MSE: 0.0007154635395936926
# slope MAE: 0.022123698249797978
# RPE MSE: 1.2732544485535995
# RPE MAE: 0.9967434894606727


