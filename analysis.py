import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_loc = r'C:\Users\rsjon_000\Desktop\VDTRC_hw\JonesSky_analysis_data.xlsx'  # absolute path to the data
graph_out = r'C:\Users\rsjon_000\Desktop\VDTRC_hw\'blood_cell_velocities.png'  # absolute path to plot output

#####

sheets = ['Vessel 1', 'Vessel 2']
vessel_data = {s: pd.read_excel(data_loc, s) for s in sheets}  #  this is a dictionary comprehension

for vessel, df in vessel_data.items():
    data = df[['Time [sec]', 'Cell number', 'Cell velocity [µm/sec]']].copy()
    data = data.sort_values('Time [sec]')
    # first thing to do is get a list of the unique cell numbers in the df and populate 'Cell number (unique)' with it
    unique_nums = data['Cell number'].unique()

    mean_vees = []
    for i in unique_nums:
        # now we can aggregate based on cell number
        sub = data.loc[df['Cell number'] == i]
        if len(sub) > 1:
            # calculate the intermeasurement time
            t_inter = sub['Time [sec]'].diff(1) # each row is the time since the last measurement
            # calculate mean veolocity of every 2 rows
            v_rolling = sub['Cell velocity [µm/sec]'].rolling(window=2).mean()
            # we can use t_inter and mean_vel to calculate the time-weighted average
            v_mean = np.average(v_rolling[1:], weights=t_inter[1:])
        else:
            v_mean = sub['Cell velocity [µm/sec]'].iloc[0]
        mean_vees.append(v_mean)