# ## Lab 2: Thermometry
# Andrew Loeppky
# ATSC 303
#
# Group: Eli Simcoe, Yang Zhou, Weihan Syu, Xinpeng Huang
#
#
# The objective of this lab is to analyze a thermocouple data and determine the type of thermocouple used. Data was taken from 12:39pm - 13:07pm on Jan 17th according to ATSC 303 Lab 2 found here:
#
# https://www.eoas.ubc.ca/courses/atsc303/Labs/2020/thermometry_lab/thermometry_2020.pdf 
#
# using a CR1000 Datalogger (AlpCAN 7071), and Laptop 3.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import stats

# Import the data saved to hard drive as a .csv

data_raw=pd.read_csv('C:/Users/Owner/UBC2020/ATSC303/thermometry_rawdata.csv',skiprows=0,sep=',',names=("TIMESTAMP","RECORD","Batt_Volt_Min","temp_1_Avg","temp_2_Avg","temp_3_Avg","temp_4_Avg","temp_5_Avg","temp_6_Avg","volt_1_Avg","volt_2_Avg","volt_3_Avg","volt_4_Avg","volt_5_Avg","volt_6_Avg"))

# Clean up the raw data, getting rid of all the nasty NAN columns and truncating the data set to within the times the measurement took place as noted in lab notebook (take the 2nd to 2nd last minute of time range)

# +
data = data_raw.drop(["temp_3_Avg","RECORD","temp_4_Avg","temp_5_Avg","temp_6_Avg","volt_3_Avg","volt_4_Avg","volt_5_Avg","volt_6_Avg"], axis = 1)

#make sure all data types that need to be floats are floats
data = data[(data['TIMESTAMP'] > '2020-01-17 12:40:00') & (data['TIMESTAMP'] < '2020-01-17 13:16:00')]
numkeys={"Batt_Volt_Min","temp_1_Avg","temp_2_Avg","volt_1_Avg","volt_2_Avg"}
for key in numkeys:
    data.dropna(subset=[key], axis = 0 , inplace= True)
    data[key] = data[key].astype(float)

data['temp_1_Avg'].dtype
# -

data

#plot the time series of the whole data set, just for fun
data.plot(figsize=(20,5),kind='line',x='TIMESTAMP',y='temp_1_Avg',color='g');

#Perform linear regressions on each set (voltage, temp)
b_slope, b_intercept, b_r_value, b_p_value, b_std_err  = scipy.stats.linregress(data['temp_1_Avg'],data['volt_1_Avg'])
p_slope, p_intercept, p_r_value, p_p_value, p_std_err  = scipy.stats.linregress(data['temp_2_Avg'],data['volt_2_Avg'])

#now plot output voltage against temperature as recorded by the datalogger
fig= plt.figure(figsize=(10,10))
plt.scatter(data['temp_1_Avg'],data['volt_1_Avg'],s=30,color='b')
plt.plot(data['temp_1_Avg'],(b_slope*data['temp_1_Avg'])+b_intercept,color='lightblue')
plt.ylabel('Voltage ($\mu V$)', fontsize=12)
plt.xlabel('CR1000 Temp ($\degree C$)', fontsize=12);
plt.title('Blue Thermocouple Voltage vs CR1000 Datalogger Temperature', fontdict={'fontsize': 20, 'fontweight': 'medium'});
print(f'Blue thermocouple regression: T = {round(b_slope,4)} V + {round(b_intercept,4)}')

fig= plt.figure(figsize=(10,10))
plt.scatter(data['temp_2_Avg'],data['volt_2_Avg'],s=30,color='purple')
plt.plot(data['temp_2_Avg'],(p_slope*data['temp_2_Avg'])+p_intercept,color='violet')
plt.ylabel('Voltage ($\mu V$)', fontsize=12)
plt.xlabel('CR1000 Temp ($\degree C$)', fontsize=12);
plt.title('Purple Thermocouple Voltage vs CR1000 Datalogger Temperature', fontdict={'fontsize': 20, 'fontweight': 'medium'});
print(f'Purple thermocouple regression: T = {round(p_slope,4)} V + {round(p_intercept,4)}')

# # Results
# From the slopes of the two regressions, we conclude that the blue thermocouple has a sensitivity of 40.5$\mu V/K$ (type T) and the purple thermocouple has a sensitivity of 60.5$\mu V/K$ (type E). 
#
# #### Sources of error include:
#
# -Imperfections in thermocouple construction (wires were twisted together over a finite length)
#
# -Some voltage is lost over the length of wire leading from the thermocouple to the datalogger, as well as the impedance of the datalogger itself
#
# #### Assumptions regarding thermocouples
#
# -Voltage response to temperature is linear over the range of temperatures tested
#
# -We are trusting the datalogger to have an accurate baseline temperature reading, as thermocouples only measure temperature $\textit{changes}$
#

# # Questions
#
# #### 1) Discuss how the model on slide 14 of the overview lecture applies to the measurements done in the lab. Label what happened in each applicable step and say what device it was associated with. 
#
# Sensor $\rightarrow$ thermocouple we constructed (blue and purple
#
# ASC $\rightarrow$ response time is governed by the length of time required to for the sensor to reach thermal equilibrium. For example, our blue sensor had much more electrical tape insulating it than the purple, and thus had a much longer response time
#
# ADC $\rightarrow$ datalogger converted DC voltage signal to a digital number to be stored in the computer
#
# DSC $\rightarrow$ Results were rounded to be stored digitally, but retained more precision than required for the purpose of this lab (not a large source of error)
#
# Transmit $\rightarrow$ Step 11 in procedure. Data is transmitted from CR1000 to the computer hard drive.
#
# Storage $\rightarrow$ Data is stored in the computer hard drive
#
# Display $\rightarrow$ After processing (in this notebook), data is displayed on a laptop screen using python/pandas/jupyter
#

# #### 2) What are some advantages of using thermocouples over other temperature sensors?
#
# Thermocouples are easy to construct and cheap to acquire. Because of their simplicity, they are also reliable, given that other components (eg datalogger) continue to function as well

# #### 3) What are some disadvantages?
#
# Thermocouples only measure differential temperature, so you need an additional thermometer to measure absolute temperature. Over a large temperature range, the response is non-linear, so one must be careful in selecting a thermocouple that is fit for purpose.

# #### 4) Why does soldering the two wires not affect the temperature reading?
#
# Thermocouple law number 3: "If a third metal 3 is inserted in one of the junctions, no net voltage is generated so long as the two new junctions are at the same temperature." Solder is a 3rd metal, but it will not generate a net voltage, but it is best to use one small, neat blob of solder so that no tempature gradient develops at the sensor (as would be the case with a very large thermocouple)

# #### 5) Why will your temperature/voltage reading be baised if the thermocouple wire is too short?
#
# Metal wires are generally good conductors of heat. If the thermocouple wire is too short, heat will be conducted along the wire from the sample you are trying to measure to the other end of the thermocouple with "known" temperature, contaminating the measurement.

# #### 6) What is the Seebeck effect?
#
# Also known as the $\textit{thermoelectric effect}$, the Seebeck effect states that two or more conductors span a temperature gradient, an emf (electromotive force) will be generated as long as the junctions are maintained at different temperatures. 

# #### 7) Assume internal datalogger metals are not the same as the thermocouples. Why does the presence of a third dissimilar metal not affect our results? What do we need to make sure in regards to the terminals of the datalogger?
#
# Thermocouple law number 2: "If a third metal is inserted in either wire and the two new junctions are at the same temperature, no effective voltage is generated by third metal." As long as the terminals of the datalogger are at the same temperature (as each other), law #2 holds and the measurement will be unaffected.

# #### 8) What does the sensitivity of a sensor mean? What are the units of sensitivity for a thermocouple?
#
# Sensitivity is the ratio of the sensor response to a given input. In the case of a thermocouple, sensitivity can be written as:
#
# $$
# \frac{\Delta V}{\Delta T}
# $$
#
# Where the voltage is the response to a change in temperature.

# #### 9) Is what we performed a calibration? If so, explain. if not, how could we have performed one?
#
# We used a sensor, but did not compare it to an independent standard, so we did not perform a calibration. Had we measured the temperature of the air/cup with another thermometer (that we trust to be accurate) and compared the thermocouple measurement to the base, then we could calibrate the thermocouple.

# #### 10) How much will a bimetallic strip deflect for a temperature change of  $10 \degree C$ if it is 5cm long, 1mm thick, and has a deflection constant of $5\cdot 10^{-5}\degree C^{-1}$?
#
# $$
# Y = \frac{K \cdot\Delta T\cdot L^2}{th}
# $$
#
# $$
# Y = \frac{5\cdot 10^{-5}\degree C^{-1} \cdot 10\degree C\cdot 0.05m^2}{0.001m}
# $$
#
# $$
# \boxed{Y = 1.25mm}
# $$

# #### 11) Given a mercury-in-glass thermometer with $200mm^3$ of mercury in the bulb, a capillary diameter of 0.015mm, and a mercury cubic thermal expansivity of $1.6\cdot 10^{-4}$, calculate the sensitivity.
#
# $$
# \frac{\Delta l}{\Delta T} = \frac{\beta V_0}{\pi r^2}
# $$
#
# $$
# = \frac{1.6e-4 \cdot 200mm^3}{\pi\cdot 0.015mm}
# $$
#
# $$
# \boxed{\frac{\Delta l}{\Delta T} = 0.679 mm/\degree C}
# $$

# #### 12) Define the following terms:
#
# Thermistor: A device used to measure temperature as a function of resistance of a semi-conductor. A notable difference from a conducting ERT is that resistance typically $\textit{decreases}$ with increasing T
#
# Centigrade Scale: A temperature measurement scale where 0 is the freezing point (not the triple point) of water and 100 is the boiling point. 
#
# Metal Resistance Thermometer: A device which measures temperature as a (increasing) function of resistance measured in a metal.
#
# Self Heating: A hot-temperature bias possible with resistance thermometers, where the current dissipated through the thermometer (heating it up) causes a higher that true temperature reading
#
# Bimetallic Strip: An arrangement of two dissimilar metals with differing linear expansion coefficients used to measure temperature. Heating a bimetallic strip causes one side to expand more than the other, deflecting the strip a predictable distance.

# #### 13) An ideal radiation shield has what characteristics?
#
# Harrison: pg 92
#
# Shield 100% of incoming radiation (at all wavelengths) from the sun and/or buildings, trees, anything
#
# Not heat up on its own and bias the sensor
#
# Allow free passage of air to the sensor

# #### 14) What happens if you pass an excessive current through a metal resistance thermometer?
#
# It will self-heat! The current dissipating in the wire will bias the reading

# #### 15) A thermocouple has a transfer equation $\Delta V = (a + b\Delta T)\Delta T$, where $a=38.6 \mu V/K, b=0.0413\mu V/K^2.
#
# a) When \Delta T = 10 K, what is the output voltage? What are the units?
#
# $$
# \Delta V = (38.6\mu V + 0.0413\mu V/K\cdot 10K)\cdot10K
# $$
#
# $$
# \Delta V = \boxed{390\mu V}
# $$
#
# b) When $\Delta T = -10K$, what is the sensitivity?
#
# $$
# \frac{\Delta V}{\Delta T} = a + b\Delta T
# $$
#
# $$
# \frac{\Delta V}{-10K} = (38.6\mu V + 0.0413\mu V/K\cdot -10K)
# $$
#
# $$
# \frac{\Delta V}{\Delta T} = \frac{-382\mu V}{-10K} = \boxed{38.2 \mu V/K}
# $$
#
# c) When $\Delta T = 40K$, what is the sentitivity?
#
# $$
# \frac{\Delta V}{\Delta T} = a + b\Delta T
# $$
#
# $$
# \frac{\Delta V}{40K} = (38.6\mu V + 0.0413\mu V/K\cdot 40K)
# $$
#
# $$
# \frac{\Delta V}{\Delta T} = \frac{1610\mu V}{40K} = \boxed{40.3 \mu V/K}
# $$
#
# d) Show that $\frac{d\Delta V}{d\Delta T}= \frac{dV}{dT}$
#
# $$
# d\Delta V = dV - dV_0
# $$
# $$
# d\Delta T = dT -dT_0
# $$
#
# The total derivative of a constant $V_0$ or $T_0$ is 0.
# $$
# d\Delta V = dV
# $$
# $$
# d\Delta T = dT
# $$
#
# $$
# \therefore \frac{d\Delta V}{d\Delta T} = \frac{dV}{dT}
# $$


