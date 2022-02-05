
# For this data science challenge, you are provided with a dataset containing mobility traces of ~500 taxi cabs in San Francisco collected over ~30 days. The format of each mobility trace file is the following - each line contains [latitude, longitude, occupancy, time], e.g.: [37.75134 -122.39488 0 1213084687], where latitude and longitude are in decimal degrees, occupancy shows if a cab has a fare (1 = occupied, 0 = free) and time is in UNIX epoch format.
# 
#  
# 
# The goal of this data science challenge is twofold:
# 
# 1. To calculate the potential for a yearly reduction in CO2 emissions, caused by the taxi cabs roaming without passengers. In your calculation please assume that the taxicab fleet is changing at the rate of 15% per month (from combustion engine-powered vehicles to electric vehicles). Assume also that the average passenger vehicle emits about 404 grams of CO2 per mile.
# 
# 2. To build a predictor for taxi drivers, predicting the next place a passenger will hail a cab.
# 
# 3. (Bonus question) Identify clusters of taxi cabs that you find being relevant from the taxi cab company point of view.


print("lets's start")
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

blue, = sns.color_palette("muted", 1)
sns.despine(left=True)



import pandas as pd
import numpy as np
import glob
glob.glob("cabspottingdata/*.txt")
l = [pd.read_csv(filename, sep = " ", names = ['lat', 'long', 'occupancy', 'time']) for filename in glob.glob("cabspottingdata/*.txt")[:2]]
df = pd.concat(l, axis=0, ignore_index=False)



def f(i):
    return pd.read_csv(i,  sep = " ", names = ['lat', 'long', 'occupancy', 'date']  )
    
import os

filepaths = [f for f in os.listdir("cabspottingdata") if f.endswith('.txt')]

headers = []
for i in filepaths:
    headers.append(i.replace('.txt', ''))

df = pd.concat(map(f, [ "cabspottingdata/" + f for f in filepaths]), keys=headers, axis=0 )


df.date = pd.to_datetime(df['date'], unit='s')


print(df.describe(include='all'))
# so we still have duplicated date....
# so occupancy min is 
#df.plot.line(subplots=True, x = "date", figsize=(12,4))

print(df.head())



print(df.isnull().sum(axis=0))
# very good there are no null entries
#df.occupancy.resample('M')

df  = df.reset_index(col_fill='ciao').set_index('date', drop=True)
df = df.rename(columns={'level_0': 'taxi_name', 'level_1': 'index_in_taxi'})



print(df.describe(include = 'all'))



#plt occupancy value vs date
plt.plot(df.occupancy.resample('D').mean())#['index'])
plt.title('occupancy over time')
plt.ylabel('')
plt.ylim(0)
plt.xticks(rotation=45)
plt.show()
plt.savefig('occupancy_vs_date')

