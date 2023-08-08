import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

        plt.savefig(output_plot_dir + "/" + test_df_info[0] + ".png")

# Set parameters
pred_cat = "slope" # rpe or slope
use_time = False
time = 'time_' if use_time else "no_time_"
output_plot_dir = "/home/kevin/Documents/quori/src/openface2_ros/results/svr_" + time + pred_cat 

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

# Initialize the Support Vector Machine Regressor
svm_regressor = SVR(kernel='linear')  # You can try other kernels as well, such as 'rbf'

# Train the model on the training data
svm_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_regressor.predict(X_test)

# Calculate the mean squared error to evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'{pred_cat} MSE: {mse}')
print(f'{pred_cat} MAE: {mae}')

# Plot predictions
plot_results(svm_regressor, test_dfs, output_plot_dir)

# Plot rpes if predicting slopes
if pred_cat == 'slope':
    all_preds = []
    all_actuals = []

    for test_df_info in test_dfs:
        test_df = test_df_info[1]

        timestamps = test_df['ds']
        rpes = test_df['rpe']

        slopes = svm_regressor.predict(test_df[features])

        predicted_rpes = [1.0]
        for i in range(1, len(slopes)):

            slope = slopes[i]

            diff = timestamps[i] - timestamps[i-1]

            if diff >= 30.0:
                slope = -0.041

            val = min(max(predicted_rpes[i-1] + diff * slope, 1.0), 10)
            predicted_rpes.append(val)

        all_actuals.extend(rpes)
        all_preds.extend(predicted_rpes)

        plt.figure(figsize=(10, 6))

        plt.plot(timestamps, rpes, label='actual', marker='o')
        plt.plot(timestamps, predicted_rpes, label='predicted', marker='x')

        plt.xlabel('Timestamp', fontsize=18)
        plt.ylabel("RPE", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.legend()
        plt.tight_layout()
        
        plt.grid(True)
        plt.savefig(output_plot_dir + "/" + test_df_info[0] + "_rpe.png")

    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)

    print("RPE MSE:", mse)
    print("RPE MAE:", mae)

# rpe (time)
# N/A

# rpe (no time)
# rpe MSE: 5.200404816957004
# rpe MAE: 1.9571810106238299

# slope (time)
# N/A

# slope (no time)
# slope MSE: 0.0007090072272201494
# slope MAE: 0.021594081144677757
# RPE MSE: 3.1885927195068713
# RPE MAE: 1.4863301787775285