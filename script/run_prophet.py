import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
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

    # csv_file = "/home/kevin/Documents/quori/src/openface2_ros/features/8.csv"
    # df = pd.read_csv(csv_file)
    # test_data = df
    
    return train_data, test_data, df_list[train_size:]

def add_regressors(model):
    regressors = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']
    for regressor in regressors:
        model.add_regressor(regressor)

def plot_results(model, test_dfs, output_plot_dir):
    for test_df_info in test_dfs:
        test_df = test_df_info[1]
        test_data = test_df[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']]
        forecast = model.predict(test_data)
        pred = forecast['yhat'].values
        actual = test_df[pred_cat].values

        timestamps = test_df['ds']

        plt.figure(figsize=(8, 6))
        plt.plot(timestamps, actual, label='actual', marker='o')
        plt.plot(timestamps, pred, label='predicted', marker='x')

        plt.xlabel('timestamps')
        plt.ylabel(pred_cat)

        plt.legend()
        plt.grid(True)

        plt.savefig(output_plot_dir + "/" + test_df_info[0] + ".png")

# Define parameters
pred_cat = "slope" # "rpe" or "slope"
csv_file_dir = "/home/kevin/Documents/quori/src/openface2_ros/features"
output_plot_dir = "/home/kevin/Documents/quori/src/openface2_ros/results/prophet_" + pred_cat 
if not os.path.exists(output_plot_dir):
    os.mkdir(output_plot_dir)


# Get train and test data
train_data, test_data, test_dfs = get_train_test_data(csv_file_dir)

# Prepare the data for Prophet with additional regressors (same as before)
prophet_train_data = train_data[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S', pred_cat]]
prophet_train_data.rename(columns={'ds': 'ds', pred_cat: 'y', 'AU01': 'AU01', 'AU02': 'AU02', 'AU04': 'AU04', 'AU05': 'AU05', 'AU06': 'AU06', 'AU07': 'AU07', 'AU09': 'AU09', 'AU10': 'AU10', 'AU12': 'AU12', 'AU14': 'AU14', 'AU15': 'AU15', 'AU17': 'AU17', 'AU20': 'AU20', 'AU23': 'AU23', 'AU25': 'AU25', 'AU26': 'AU26', 'AU28': 'AU28', 'AU45': 'AU45', 'E0': 'E0', 'E1': 'E1', 'E2': 'E2', 'E3': 'E3', 'E4': 'E4', 'S': 'S'}, inplace=True)

# Define a grid of hyperparameters to search
params = {
    'changepoint_prior_scale': 1.0,
    'seasonality_prior_scale': 1.0
}

# Train using the parameters
model = Prophet(changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'])
add_regressors(model)
model.fit(prophet_train_data)

# Make predictions on the test set with additional regressors
prophet_test_data = test_data[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']]
# prophet_test_data.rename(columns={'ds': 'ds', 'AU01': 'AU01', 'AU02': 'AU02', 'AU04': 'AU04', 'AU05': 'AU05', 'AU06': 'AU06', 'AU07': 'AU07', 'AU09': 'AU09', 'AU10': 'AU10', 'AU12': 'AU12', 'AU14': 'AU14', 'AU15': 'AU15', 'AU17': 'AU17', 'AU20': 'AU20', 'AU23': 'AU23', 'AU25': 'AU25', 'AU26': 'AU26', 'AU28': 'AU28', 'AU45': 'AU45', 'E0': 'E0', 'E1': 'E1', 'E2': 'E2', 'E3': 'E3', 'E4': 'E4', 'S': 'S'}, inplace=True)

forecast = model.predict(prophet_test_data)
predicted_values = forecast['yhat'].values
actual_values = test_data[pred_cat].values

# Calculate MSE for evaluation
mse = mean_squared_error(actual_values, predicted_values)
mae = mean_absolute_error(actual_values, predicted_values)

print(pred_cat + " MSE:", mse)
print(pred_cat + " MAE:", mae)

# Plot predictions
plot_results(model, test_dfs, output_plot_dir)

# Plot rpes if predicting slopes
if pred_cat == 'slope':
    all_preds = []
    all_actuals = []

    for test_df_info in test_dfs:
        test_df = test_df_info[1]

        timestamps = test_df['ds']
        rpes = test_df['rpe']

        test_data = test_df[['ds', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45', 'E0', 'E1', 'E2', 'E3', 'E4', 'S']]
        forecast = model.predict(test_data)
        slopes = forecast['yhat'].values

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

        # Plot
        plt.figure(figsize=(8, 6))

        # Plot y1s and y2s against xs
        plt.plot(timestamps, rpes, label='actual', marker='o')
        plt.plot(timestamps, predicted_rpes, label='predicted', marker='x')

        # Add labels and title
        plt.xlabel('timestamps')
        plt.ylabel('rpes')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.savefig(output_plot_dir + "/" + test_df_info[0] + "_rpe.png")

    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)

    print("RPE MSE:", mse)
    print("RPE MAE:", mae)

# rpe
# rpe MSE: 8.304602379476139
# rpe MAE: 2.6761112759786827

# slope
# slope MSE: 0.010184522062987783
# slope MAE: 0.0714298648399075
# RPE MSE: 24.13504591726835
# RPE MAE: 4.357856606659436