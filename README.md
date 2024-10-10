# Water-quality-inversion-based-on-spectral-features-and-CatBoost
This project is used for the inversion study of Sentinel-3 OLCI chlorophyll-a concentration and suspended solids concentration


本代码适用于欧空局公开的Sentinel-3 OLCI S3A_OL_1_EFR遥感数据，在使用代码前，请先使用欧空局官方的图像处理软件SNAP进行大气校正与地理校正处理。
![image](https://github.com/user-attachments/assets/925ec722-1925-466e-8fb5-a929638268be)
![image](https://github.com/user-attachments/assets/a32b8bd5-1fc3-49fb-92dd-ee5965d6830c)

处理后将前缀为rrs的img影像进行合成，即可得到代码中输入的.tif影像。

本项目的代码分为以下三个部分：

1、Water clustering：用于输入图像并进行水体的分类。

2、Correlation analysis：用于对四种光谱特征组合进行相关性分析

3、model_train：用于进行Chla和TSS浓度的模型训练与预测。
