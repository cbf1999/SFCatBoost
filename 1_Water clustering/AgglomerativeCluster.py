import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

# Read Excel file and specify worksheet name
file_path = 'WQ_data.xlsx'
sheet_name = 'sheet_name'  # Replace with your worksheet name
data = pd.read_excel(file_path, sheet_name=sheet_name)

spectral_col_start = 8  # Which column does the spectrum start in
spectral_col_end = 23  # Which column does the spectrum end in
chlorophyll_col = 5  # Which column is the chlorophyll concentration in
suspended_sediment_col = 6  # Which column is the concentration of suspended solids in
cdom_col = 7  # Which column is the CDOM concentration in
spectral_data = data.iloc[:, spectral_col_start - 1:spectral_col_end - 1]
chlorophyll_concentration = data.iloc[:, chlorophyll_col - 1]
suspended_sediment_concentration = data.iloc[:, suspended_sediment_col - 1]
cdom_concentration = data.iloc[:, cdom_col - 1]

# Calculate cosine distance matrix
cosine_distances = squareform(pdist(spectral_data, metric='cosine'))

# Cluster using Agglomerative Clustering
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
labels = clustering.fit_predict(cosine_distances)

# Calculate the average spectral curve,
# average chlorophyll concentration, average suspended solids concentration,
# and average CDOM concentration for each category
average_spectra = []
average_chlorophyll = []
average_suspended_sediment = []
average_cdom = []

for cluster in range(n_clusters):
    cluster_data = spectral_data[labels == cluster]
    mean_spectrum = cluster_data.mean(axis=0)
    average_spectra.append(mean_spectrum)

    cluster_chlorophyll = chlorophyll_concentration[labels == cluster]
    mean_chlorophyll = cluster_chlorophyll.mean()
    average_chlorophyll.append(mean_chlorophyll)

    cluster_suspended_sediment = suspended_sediment_concentration[labels == cluster]
    mean_suspended_sediment = cluster_suspended_sediment.mean()
    average_suspended_sediment.append(mean_suspended_sediment)

    cluster_cdom = cdom_concentration[labels == cluster]
    mean_cdom = cluster_cdom.mean()
    average_cdom.append(mean_cdom)

# Sort by average chlorophyll concentration
sorted_indices = np.argsort(average_chlorophyll)
sorted_spectra = [average_spectra[i] for i in sorted_indices]
sorted_chlorophyll = [average_chlorophyll[i] for i in sorted_indices]
sorted_suspended_sediment = [average_suspended_sediment[i] for i in sorted_indices]
sorted_cdom = [average_cdom[i] for i in sorted_indices]

# Print the average chlorophyll concentration, suspended solids concentration, and CDOM concentration for each category (to four decimal places)
print("Average Concentrations for each cluster (sorted by chlorophyll):")
for i in range(n_clusters):
    print(
        f"Cluster {i}: Chlorophyll = {sorted_chlorophyll[i]:.4f}, Suspended Sediment = {sorted_suspended_sediment[i]:.4f}, CDOM = {sorted_cdom[i]:.4f}")

# Reassign category number
new_labels = np.zeros_like(labels)
for new_label, old_label in enumerate(sorted_indices):
    new_labels[labels == old_label] = new_label

# Save clustering results and average spectral curves
np.save('Cluster_model/average_spectra.npy', sorted_spectra)

# Visualize the average spectral curve of each category
colors = cm.get_cmap('Greens', n_clusters)

for cluster in range(n_clusters):
    plt.plot(sorted_spectra[cluster], label=f'Cluster {cluster}')

plt.xlabel('Wavelength')
plt.ylabel('Reflectance')
plt.legend()
plt.title('Average Spectrum of Each Cluster')
plt.savefig('image/clustered_spectra.png')
plt.show()

# Add the new clustering results back to the original data box
data['Cluster'] = new_labels

# Save Excel file with new clustering results
output_file_path = 'result_excel/sorted_clustered_spectral_data.xlsx'
data.to_excel(output_file_path, index=False)

# Create a DataFrame to store average spectral data
average_spectra_df = pd.DataFrame(sorted_spectra, columns=spectral_data.columns)
average_spectra_df['Cluster'] = range(n_clusters)
average_spectra_df['Chlorophyll'] = sorted_chlorophyll
average_spectra_df['Suspended Sediment'] = sorted_suspended_sediment
average_spectra_df['CDOM'] = sorted_cdom

# Save the average spectral data to an Excel file
average_spectra_output_path = 'result_excel/average_spectra.xlsx'
average_spectra_df.to_excel(average_spectra_output_path, index=False)

print("Clustering completed and results saved.")
