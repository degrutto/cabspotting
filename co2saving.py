
# ### Question 1 - CO2 emission reduction

# We first need to find distance via longitude and latitude in Miles, only occupancy set to zero. Because in the question asked that by the taxi cabs roaming without passengers. 
# As we have 1 month,  we multiply the result by 12 and get the yearly distance of interest for the CO2 emissions
# Then, we assume that the taxi cab fleet is changing at the rate of 15% per month toward electric mobility. 
# 
# CO2 reduction is obtained by multiplying the total distance with the number of CO2 grams per mile and comparing the difference in terms of percentage.
 
 
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )
from tqdm import tqdm


import random
import datetime
from os import listdir

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

blue, = sns.color_palette("muted", 1)
sns.despine(left=True)


# first get the data

path = 'cabspottingdata'

all_files = [file_name for file_name in listdir(path) if file_name.endswith('.txt')]
print(f'All files are {len(all_files)}')

PercFiles = 1.0
random.Random(1923).shuffle(all_files)
SelectedFiles = all_files[:int(PercFiles * len(all_files))]

print(f'\n{len(SelectedFiles)} randomly selected files, if not all')


from utils import ConverttoDF, PreviousCoordinates, EstimatedDistance, DistanceCalculation

Data = pd.DataFrame()
with tqdm(desc="Process files", total=len(SelectedFiles), mininterval=10) as pbar:
    for f in SelectedFiles:
        pbar.update(1)
        df = ConverttoDF(file_name=f)
        df = PreviousCoordinates(df=df)
        df = DistanceCalculation(df=df)
        Data = pd.concat([Data, df])
       


print(Data.Taxi)




pd.set_option('display.float_format', lambda x: '%.4f' % x)

print(Data.describe())

#When I checked the count field below, I can say that there is no nan value there are no outliers.
# 
# 
# Maximum time is 06/10/2008 @ 9:09am (UTC) and minimum time is 05/17/2008 @ 10:00am (UTC)
# 
# Occupancy and vacancy rate seems to be equally distributed. 


# scatter plots
ax =sns.scatterplot(x="Miles", y="Occupancy", data=Data,  s=100 )
ax.tick_params(rotation=90)

totocc0 = Data.Miles[Data.Occupancy == 0].sum()
totocc1 = Data.Miles[Data.Occupancy == 1].sum()

print('max potential C02 reduction ', totocc0 / (totocc0 + totocc1))
# so we have up to 42% co2 saving to do....



df.Timestamp = pd.to_datetime(Data['Timestamp'], unit='s')



df.set_index(df.Timestamp, inplace=True)


# Average vacant distance for each taxi driver that I select

NoPassanger = Data[Data['Occupancy'] == 0]
#Distance Without Passanger
DistanceWOPassanger = NoPassanger.groupby(by=['Taxi'])['Miles'].sum()
print(DistanceWOPassanger)




DistanceWOPassangerPerMonth = DistanceWOPassanger.sum()
#We can assume that multiplying monthly distance by 12, we can find the yearly distance.
DistanceWOPassangerPerYear = DistanceWOPassangerPerMonth * 12

print(f'The distance for CO2 in one year (combustion engine-powered vehicles) '
      f'is approx {round(DistanceWOPassangerPerYear)} Miles\n')





#electric vehicles
DistanceWOPassangerPerYear = 0.0


#assume that the taxi cab fleet is changing at the rate of 15% per month
for month in range(12):
    if month == 0:
        DistanceWOPassangerPerYear = DistanceWOPassangerPerMonth
    else:
        DistanceWOPassangerPerMonth = DistanceWOPassangerPerMonth * 0.85
        DistanceWOPassangerPerYear += DistanceWOPassangerPerMonth
    print(f'The distance for CO2 after {month} month(s) of'
          f' Combustion Vehicles is approx {round(DistanceWOPassangerPerMonth, 3)} Miles')
    
print(f'\nThe distance for CO2 after one year of Combustion Vehicles '
      f' is approx {round(DistanceWOPassangerPerYear)} Miles')




CO2GramsPerMiles = 404
CO2EmissionWP = DistanceWOPassangerPerYear / PercFiles * CO2GramsPerMiles
print('Co2 emission without passanger', CO2EmissionWP)



DistanceTotal = Data.groupby(by=['Taxi'])['Miles'].sum()
print('DistanceTotal ', DistanceTotal.sum() * 12)
DistanceTotalYear = DistanceTotal.sum() * 12
CO2EmissionTotal = DistanceTotalYear / PercFiles * CO2GramsPerMiles
print('total C02 emission', CO2EmissionTotal)



print('potential C02 emission save [%]', round(CO2EmissionWP / CO2EmissionTotal , 3) * 100 )


