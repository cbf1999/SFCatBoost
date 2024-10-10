import pandas as pd
import numpy as np

# Read Excel file
df = pd.read_excel('WQ_data.xlsx',sheet_name='sheet_name')

# Extract measured data columns and band data columns
cod_data = df['aCDOM440'].values

band_data = df.iloc[:, 6:21].values


# Initialization result DataFrame
result_df = pd.DataFrame(columns=['Band1', 'Band2', 'Correlation'])

# Traverse all band ratios and calculate the Pearson correlation coefficient with the measured data
for i in range(band_data.shape[1]):
    for j in range(band_data.shape[1]):
        if i != j:
            ratio_data = (band_data[:, i] - band_data[:, j]) / (band_data[:, i] + band_data[:, j])
            pearson_corr = np.corrcoef(ratio_data, cod_data)[0, 1]
            result_df = result_df._append({'Band1': i+1, 'Band2': j+1, 'Correlation': pearson_corr}, ignore_index=True)
    
# Save the results to an Excel file
result_df.to_excel('Normalized Difference Index.xlsx', index=False)

print('Successfully saved')
