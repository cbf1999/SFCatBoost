import rasterio
import numpy as np
from scipy.spatial.distance import cosine

# 读取tif影像，并保留原始的空值（NoData）
tif_file_path = r"F:\4_Sentinel-3 OLCI_ENVI处理结果\20231225_S3A\20231225_S3A_mask.tif"
output_tif_path = r"F:\4_Sentinel-3 OLCI_ENVI处理结果\20231225_S3A\20231225_S3A_cluster.tif"

with rasterio.open(tif_file_path) as src:
    spectral_data = src.read(list(range(1, 16)), masked=True)  # 使用masked=True读取数据，这样会自动识别NoData
    profile = src.profile  # 获取原始tif的元数据
    nodata = src.nodata  # 获取NoData值

# 转换为(num_pixels, num_bands)形状的数组
num_bands, height, width = spectral_data.shape
spectral_data_reshaped = spectral_data.reshape((num_bands, height * width)).T

# 加载之前计算得到的四个聚类的平均光谱曲线
average_spectra = np.load('聚类结果_模型/average_spectra.npy')

# 计算每个像元与四个聚类的平均光谱曲线的形状相似度（余弦相似度）
similarities = np.zeros((spectral_data_reshaped.shape[0], len(average_spectra)))
for i, spectrum in enumerate(average_spectra):
    similarities[:, i] = 1 - np.array([cosine(pixel, spectrum) for pixel in spectral_data_reshaped])

# 为每个像元分配形状相似度最高的类别
assigned_clusters = np.argmax(similarities, axis=1)

# 转换为原始tif影像的形状
assigned_clusters = assigned_clusters.reshape((height, width))

# 维持原先空值（NoData）位置
assigned_clusters = np.where(spectral_data.mask[0], np.nan, assigned_clusters)

# 更新tif文件的元数据，设置NoData值并使用合适的压缩格式
profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, compress='lzw')

# 导出结果为tif文件
with rasterio.open(output_tif_path, 'w', **profile) as dst:
    dst.write(assigned_clusters.astype(rasterio.float32), 1)

print("Pixel cluster assignment with NoData handling completed and results saved to tif.")
