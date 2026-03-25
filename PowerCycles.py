import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.grid(True, linestyle='--', alpha=0.6)

########################################################   CONSTANTS   ########################################################
SunPower = 1361                                     # Raw power of sunlight when it reaches the satellite in W/m2
BOLSolarCellEfficiency = 0.3                        # Beginning Of Life efficiency of the Solar Cells
EOLSolarCellEfficiency = 0.209                      # End Of Life efficiency of the Solar Cells
PanelSurface = 3.8                                  # Effective surface of the Solar Panels (just the surface of the cells)
BOLBatteryCapacity = 900                            # Initial capacity of the battery in Wh
BatteryDegradationFittingConstant = 0.0000348       # Curve fitting constant in s^-1/2
PowerDraw = 700                                     # Power drawn by the system 

#######################################################   MAIN PROGRAM   ######################################################
# Import data from csv files
SolarIntensity = pd.read_csv('Solar_intensity.csv')['Intensity']
SunVector = pd.read_csv('sun_vector.csv')[['x (km)', 'y (km)', 'z (km)']]

# Check if the files match dates and times
Times = pd.read_csv('Solar_intensity.csv')[['Time (UTCG)']]
Times = Times.rename(columns={'Time (UTCG)': 'Solar Intensity'})
Times['Sun Vector'] = pd.read_csv('sun_vector.csv')['Time (UTCG)']
Times['Solar Intensity'] = pd.to_datetime(Times['Solar Intensity'])
Times['Sun Vector'] = pd.to_datetime(Times['Sun Vector'])
if (Times['Sun Vector'] != Times['Solar Intensity']).any():
    if len(Times['Sun Vector']) != len(Times['Solar Intensity']):
        raise ValueError('>>> ERROR: Files have different number of rows. Files may be from different simulations')
    else:
        raise ValueError('>>> ERROR: Timestamps are not the same between files. Files may be from different simulations')
    
# Create a timestep for the calculations
Times['Timestep'] = Times['Solar Intensity'].diff().dt.total_seconds().fillna(0)
Times['CumulativeTime'] = Times['Timestep'].cumsum()

# Calculate angles of incidence from each plane
SunVector['X Face Angle'] = np.arcsin(SunVector['x (km)']/(np.sqrt((SunVector['x (km)']**2) + (SunVector['y (km)']**2) + (SunVector["z (km)"]**2))))
SunVector['Y Face Angle'] = np.arcsin(SunVector['y (km)']/(np.sqrt((SunVector['x (km)']**2) + (SunVector['y (km)']**2) + (SunVector["z (km)"]**2))))
SunVector['Z Face Angle'] = np.arcsin(SunVector['z (km)']/(np.sqrt((SunVector['x (km)']**2) + (SunVector['y (km)']**2) + (SunVector["z (km)"]**2))))

# Calculate efficiency of the Solar Cells for the duration of the mission
PanelEfficiency = pd.Series(np.linspace(BOLSolarCellEfficiency, EOLSolarCellEfficiency, num = len(Times['Sun Vector'])))

# Calculate the power generation from each satellite face
Generation = pd.DataFrame()
Generation['W generated in face X'] = SunPower * np.sin(SunVector['X Face Angle']) * PanelEfficiency * PanelSurface * SolarIntensity
Generation['W generated in face Y'] = SunPower * np.sin(SunVector['Y Face Angle']) * PanelEfficiency * PanelSurface * SolarIntensity
Generation['W generated in face Z'] = SunPower * np.sin(SunVector['Z Face Angle']) * PanelEfficiency * PanelSurface * SolarIntensity

# Calculate the degradation of the battery in each point in time
Battery = pd.DataFrame()
Battery['TotalCapacity'] = BOLBatteryCapacity * (1 - (BatteryDegradationFittingConstant * np.sqrt(Times['CumulativeTime'])))

# Calculate state of charge of the battery in each point in time
Battery['Charge'] = np.nan
Battery.loc[0, 'Charge'] = BOLBatteryCapacity
OutOfBattery = 0
for i in range(len(Battery['Charge'])-1):
    NextValue = Battery['Charge'].iloc[i] + (((Generation['W generated in face Y'][i+1] - PowerDraw) * Times['Timestep'][1])/3600)
    if NextValue > Battery['TotalCapacity'].iloc[i+1]:
        Battery.loc[i+1, 'Charge'] = Battery['TotalCapacity'].iloc[i+1]
    elif NextValue < 0:
        Battery.loc[i+1, 'Charge'] = 0
        OutOfBattery = 1
    else:
        Battery.loc[i+1, 'Charge'] = NextValue