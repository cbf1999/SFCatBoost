import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import rasterio

#Read the training and testing sets from the Excel file
file_path = r'excel\TSS_best_train_test_data.xlsx'
train_sheet_name = 'Training Set'
test_sheet_name = 'Test Set'

tif_path =        r"20221221_S3B_EFR_bandlayerTSS.tif"
output_tif_path = r"20221221_S3B_EFR_TSS.tif"

train_data = pd.read_excel(file_path, sheet_name=train_sheet_name)
test_data = pd.read_excel(file_path, sheet_name=test_sheet_name)

spectrum_start = 7
spectrum_end = 21
TSS = 5

#Specify the column names for other feature columns, where b16 is b13
other_feature_columns = ['Cluster', 'b16/b5', '(b16-b4)/(b16+b4)', '(b12+b16)/b5']

#Extract spectral data column and chlorophyll concentration column
X_spectral_train = train_data.iloc[:, spectrum_start - 1:spectrum_end].values
y_train = train_data.iloc[:, TSS - 1].values

X_spectral_test = test_data.iloc[:, spectrum_start - 1:spectrum_end].values
y_test = test_data.iloc[:, TSS - 1].values

#Extract other feature columns
X_other_train = train_data[other_feature_columns].values
X_other_test = test_data[other_feature_columns].values

def extract_features(spectral_data):
    features = []
    for spectrum in spectral_data:
        max_val = np.max(spectrum)
        min_val = np.min(spectrum)
        peak_indices, _ = find_peaks(spectrum)
        trough_indices, _ = find_peaks(-spectrum)

        if peak_indices.size > 0:
            peak_val = spectrum[peak_indices[0]]
        else:
            peak_val = np.nan

        if trough_indices.size > 0:
            trough_val = spectrum[trough_indices[0]]
        else:
            trough_val = np.nan

        curvature = np.gradient(np.gradient(spectrum))
        convexity = np.mean(curvature)

        slope = (spectrum[-1] - spectrum[0]) / (len(spectrum) - 1)

        rise_amplitude = np.max(np.diff(spectrum))
        fall_amplitude = np.min(np.diff(spectrum))

        features.append([max_val, min_val, peak_val, trough_val, convexity, slope, rise_amplitude, fall_amplitude])

    return np.array(features)

#Extract spectral features from the training and testing sets
X_spectral_train_features = extract_features(X_spectral_train)
X_spectral_test_features = extract_features(X_spectral_test)

#Combine spectral features with other features
X_train_features = np.hstack((X_spectral_train_features, X_other_train))
X_test_features = np.hstack((X_spectral_test_features, X_other_test))

#Building a CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=3,
    random_seed=42,
    loss_function='RMSE',
    l2_leaf_reg=1,
    verbose=0
)
model.fit(X_train_features, y_train)

#Predicting training and testing sets
y_train_pred = model.predict(X_train_features)
y_test_pred = model.predict(X_test_features)

#Evaluate the performance of the model on the training and testing sets
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training Set Mean Squared Error: {train_mse}')
print(f'Training Set R^2 Score: {train_r2}')
print(f'Test Set Mean Squared Error: {test_mse}')
print(f'Test Set R^2 Score: {test_r2}')

#Visualize prediction results
plt.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Measured vs Predicted Chlorophyll Concentration')
plt.show()

#Read tif image data and perform suspended solids concentration inversion

with rasterio.open(tif_path) as src:
    image_data = src.read()
    transform = src.transform
    crs = src.crs

#Extract spectral data and other features
spectral_data = image_data[:15, :, :]
feature_b16cb5 = image_data[15, :, :]
feature_b16_b4gui = image_data[16, :, :]
feature_b12jb16cb5 = image_data[17, :, :]
feature_cluster = image_data[18, :, :]

#Preprocess data
rows, cols = spectral_data.shape[1], spectral_data.shape[2]
spectral_data_reshaped = spectral_data.reshape(15, -1).T
feature_b16cb5_reshaped = feature_b16cb5.reshape(-1, 1)
feature_b16_b4gui_reshaped = feature_b16_b4gui.reshape(-1, 1)
feature_b12jb16cb5_reshaped = feature_b12jb16cb5.reshape(-1, 1)
feature_cluster_reshaped = feature_cluster.reshape(-1, 1)

#Identify background values (pixels with the same spectral data)
background_mask = np.all(spectral_data_reshaped == spectral_data_reshaped[:, [0]], axis=1)

#Extract spectral features
spectral_features = extract_features(spectral_data_reshaped)

#Splicing all features together
all_features = np.hstack((spectral_features, feature_b16cb5_reshaped, feature_b16_b4gui_reshaped, feature_b12jb16cb5_reshaped, feature_cluster_reshaped))

#Make predictions
predictions = np.full((rows * cols), np.nan)
predictions[~background_mask] = model.predict(all_features[~background_mask])

#Reshape to image size
predictions_image = predictions.reshape(rows, cols)

#Save the result as tif image

with rasterio.open(
    output_tif_path, 'w',
    driver='GTiff',
    height=rows, width=cols,
    count=1, dtype=predictions_image.dtype,
    crs=crs,
    transform=transform
) as dst:
    dst.write(predictions_image, 1)

print(f"Chlorophyll concentration map saved to {output_tif_path}")
