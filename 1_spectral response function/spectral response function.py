import pandas as pd
import numpy as np

# >>>>>>>>>>>>>>>>>The following three items need to be manually entered<<<<<<<<<<<<<<<<<
input_rrs = "GLORIA measured reflectance.xlsx"  # Excel file name for measured reflectance
input_srf = "S3B_SRF.xlsx"  # Satellite Spectral Response Function Excel File Name (NaN Processed)
output = "GLORIA equivalent Sentinl-3 reflectance.xlsx"  # Set output equivalent remote sensing reflectance file name

#Read data
try:
    Rrs = pd.read_excel(input_rrs, header=None).values  # Read remote sensing reflectance Rrs
    SRF = pd.read_excel(input_srf, header=None).values  # Read the spectral response function SRF of the sensor
except:
    print('>>>>>>>Reading failed, please check if the Excel file names of Rrs and SRF are incorrect<<<<<<<')

sn = Rrs[0, 1:]  #Station name

#Convert the band names (column headings) in SRF to strings
bn = SRF[0, 1:]  #Band name
bn = [str(b) for b in bn]  #Keep as string

Rrs = Rrs[1:, :].astype(float)
SRF = SRF[1:, :].astype(float)

c1 = Rrs.shape[1]  #Size of hyperspectral reflectance matrix
c2 = SRF.shape[1]  #SRF matrix size

#Change the possible NaN values and values less than 0 in the spectral response function to 0
SRF[np.isnan(SRF)] = 0
SRF[SRF < 0] = 0

#Calculate equivalent Rrs
er = np.zeros((c2 - 1, c1 - 1))  #Create a matrix for storing equivalent Rrs

for i in range(1, c2):
    for j in range(1, c1):
        l1 = np.where(SRF[:, i] != 0)[0][0]  #Lower limit of points
        l2 = np.where(SRF[:, i] != 0)[0][-1]  #Points limit
        #Align the wavelength range of Rrs and SRF
        rrs_range = np.where((Rrs[:, 0] >= SRF[l1, 0]) & (Rrs[:, 0] <= SRF[l2, 0]))[0]
        if rrs_range.size > 0:
            #Ensure that the lengths of both are the same
            common_wavelengths = np.intersect1d(Rrs[rrs_range, 0], SRF[l1:l2 + 1, 0])
            if common_wavelengths.size > 0:
                rrs_interp = np.interp(common_wavelengths, Rrs[rrs_range, 0], Rrs[rrs_range, j])
                srf_interp = np.interp(common_wavelengths, SRF[l1:l2 + 1, 0], SRF[l1:l2 + 1, i])
                er[i - 1, j - 1] = (np.trapz(rrs_interp * srf_interp, common_wavelengths)) / (np.trapz(srf_interp, common_wavelengths))

#Data saving
#Create an Excel file with labeled site and band names
temp = pd.DataFrame(np.zeros((c2, c1), dtype=object))
temp.iloc[1:, 0] = bn
temp.iloc[0, 1:] = sn
temp.iloc[1:, 1:] = er

#Output
temp.to_excel(output, index=False, header=False)