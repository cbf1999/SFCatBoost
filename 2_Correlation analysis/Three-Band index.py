import pandas as pd
import numpy as np
from tqdm import tqdm

# Read Excel file
df = pd.read_excel('WQ_data.xlsx',sheet_name='sheet_name')

# Extract measured data columns and band data columns
cod_data = df['aCDOM440'].values

band_data = df.iloc[:, 6:21].values

# Initialization result DataFrame
result_df = pd.DataFrame(columns=['Band1', 'Band2', 'Band3', 'Correlation'])

total_loops = int(band_data.shape[1] * (band_data.shape[1] - 1) * band_data.shape[1] )

# Traverse all band ratios and calculate the Pearson correlation coefficient with the measured data
with tqdm(total=total_loops) as pbar:
    for i in range(band_data.shape[1]):
        for j in range(band_data.shape[1]):
            for k in range(band_data.shape[1]):
                if i !=j and j != k and i != k:
                    ratio_data = (band_data[:, i] + band_data[:, j]) / np.where(band_data[:, k] == 0, 1e-6, band_data[:, k])
                    pearson_corr = np.corrcoef(ratio_data, cod_data)[0, 1]
                    result_df.loc[len(result_df)] = [i+1, j+1, k+1, pearson_corr]
                    pbar.update(1)
# Save the results to an Excel file
result_df.to_excel('Three-Band Index.xlsx', index=False)

print('Successfully saved')
