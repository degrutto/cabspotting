# 3. (Bonus question) Identify clusters of taxi cabs that you find being relevant from the taxi cab company point of view.


### includes #################

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

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from statistics import mean




from utils import ConverttoDF, PreviousCoordinates, EstimatedDistance, DistanceCalculation, NextPassangerCoordinates, eval_metrics, splitarray
from sklearn.cluster import DBSCAN


#### end include ##########


# first get the data

path = 'cabspottingdata'

all_files = [file_name for file_name in listdir(path) if file_name.endswith('.txt')]
print(f'All files are {len(all_files)}')

PercFiles = 0.03
random.Random(1923).shuffle(all_files)
SelectedFiles = all_files[:int(PercFiles * len(all_files))]

import datetime

def DateExtraction(row: Series):
    dt = datetime.datetime.fromtimestamp(row['Timestamp'])
    row['day'] = dt.weekday()
    row['hour'] = dt.hour
    row['minute'] = dt.minute
    row['monthday'] = dt.day
    return row
def ApplyDateExtraction(df):
    df['day'] = 0
    df['hour'] = 0
    df['minute'] = 0
    df['monthday'] = 0
    df = df.apply(lambda row: DateExtraction(row), axis=1)
    return df

print(f'\n{len(SelectedFiles)} randomly selected files, if not all')

Data = pd.DataFrame()
with tqdm(desc="Process files", total=len(SelectedFiles), mininterval=10) as pbar:
    for f in SelectedFiles:
        pbar.update(1)
        df = ConverttoDF(file_name=f)
        df = PreviousCoordinates(df=df)
        df = DistanceCalculation(df=df)
        df = ApplyDateExtraction(df =df)

        Data = pd.concat([Data, df])
       


# First is using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) that is a popular unsupervised learning method utilized in model building and machine learning algorithms. I subset 10% of the total data and divided into two in terms of occupancy flag. I looked other cabs where they are driving in the past and where they will go after it. By doing so, I clustered its cabs together based on similarity of behavior

def visualize_dbscan(db: DBSCAN, X: DataFrame):

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            continue

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask].values
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask].values
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)


    plt.title('Estimated number of clusters: %d' % n_clusters_)

    
    
sampled_data = Data[Data['Occupancy'] == 1][['Latitude', 'Longitude']]#.sample(frac=0.3, random_state=0)

db = DBSCAN(eps=0.0005, min_samples=50)
db.fit(sampled_data)
visualize_dbscan(db=db, X=sampled_data)
#plt.show()
plt.savefig('cluster_occ1.png')
plt.close()
plt.cla()
plt.clf()


sampled_data = Data[Data['Occupancy'] == 0][['Latitude', 'Longitude']]#.sample(frac=0.3, random_state=0)

db = DBSCAN(eps=0.0005, min_samples=50)
db.fit(sampled_data)
visualize_dbscan(db=db, X=sampled_data)
#plt.show()

plt.savefig('cluster_occ0.png')
plt.close()
plt.cla()
plt.clf()


MilesCovered=pd.DataFrame()

MilesCovered = Data.groupby(by=['Taxi'])['Miles'].sum()

print(MilesCovered)
WPassangerDistance = Data[Data['Occupancy'] == 1]
#Distance Without Passanger
DistanceWPassanger=pd.DataFrame()
DistanceWPassanger = WPassangerDistance.groupby(by=['Taxi'])['Miles'].sum()
print(DistanceWPassanger)

RFMData = pd.merge(MilesCovered,DistanceWPassanger, on='Taxi', how='left')
print(RFMData)

output  = pd.DataFrame()
output = Data[['Taxi','monthday']]
output = output.drop_duplicates()
ActDay = pd.DataFrame()
ActDay = output.groupby(by=['Taxi'])['monthday'].count()
print(ActDay)

#Active Minutes
ActMin = pd.DataFrame()
ActMin = Data.groupby(by=['Taxi'])['Timestamp'].count()
#Active Days
output  = pd.DataFrame()
output = Data[['Taxi','monthday']]
output = output.drop_duplicates()
ActDay = pd.DataFrame()
ActDay = output.groupby(by=['Taxi'])['monthday'].count()

Active = pd.DataFrame()
Active = pd.merge(ActMin, ActDay, on='Taxi', how='left')
#Finding Active Minutes per Day
Active['ActMinPerDay'] = Active['Timestamp'] / Active['monthday']

print(Active)

RFMData = pd.merge(RFMData,Active, on='Taxi', how='left')
RFMDataF = RFMData[['Miles_x', 'Miles_y', 'ActMinPerDay']]
RFMDataF.rename(columns={'Miles_x': 'MileCoverage', 'Miles_y': 'MileCoverageWPassanger'}, inplace=True)
print(RFMDataF)
quantiles = RFMDataF.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
print(quantiles)


#So we difine a quantile function that assign higher value to high miles/covoragewpaasenger and active days

def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    


RFMDataF['r_quartile'] = RFMDataF['MileCoverage'].apply(RScore, args=('MileCoverage',quantiles,))
RFMDataF['f_quartile'] = RFMDataF['MileCoverageWPassanger'].apply(RScore, args=('MileCoverageWPassanger',quantiles,))
RFMDataF['m_quartile'] = RFMDataF['ActMinPerDay'].apply(RScore, args=('ActMinPerDay',quantiles,))

print(RFMDataF.head())


print(RFMDataF.tail())

RFMDataF['RFMScore'] = RFMDataF.r_quartile.map(str) + RFMDataF.f_quartile.map(str) + RFMDataF.m_quartile.map(str)
RFMDataF.head()



print(RFMDataF.groupby(by=['RFMScore'])['RFMScore'].count())

# So we can give clusters a name accordingly. Here is some example of it. 
# ---
# Best Taxi Drivers: Actively driving cab and working much more than other per day
# 
#      (if mile coverage, mile coverage with passanger and active minutes per day are in the top %25)
# ---               
# Voyager : Actively driving cab
# 
#      (if mile coverage is in the top %25)
# ---               
# Most Occupied Taxi : Actively driving cab with passanger
# 
#      (if mile coverage with passanger is in the top %25)
# ---               
# Most Active Taxi: Working much more than other per day
# 
#      (if active minutes per day is in the top %25)
# ---               
# Fast and Vacant: Actively driving more miles but less minutes per day. That means he/she drives fast with more vacant 
# 
#      (if mile coverage is in the top %25 and mile coverage with passanger and active minutes per day are in the bottom %25)
# ---               
# Lucky/Eco Freindly Stand-by: Driver is not driving without passanger but he/she drive more than other in terms of mile coverage with passanger and active minutes per day.
# 
#      (if mile coverage is in the bottom %25 and mile coverage with passanger and active minutes per day are in the top %25)
# ---                
# Not Active Taxi Drivers: Not actively driving cab and working much more than other per day
# 
#      (if mile coverage, mile coverage with passanger and active minutes per day are in the bottom %25)
# --- 



print("Best Taxi Drivers: ",len(RFMDataF[RFMDataF['RFMScore']=='444']))
print('Voyagers : ',len(RFMDataF[RFMDataF['r_quartile']==4]))
print('Most Occupied Taxi : ',len(RFMDataF[RFMDataF['f_quartile']==4]))
print("Most Active Taxi: ",len(RFMDataF[RFMDataF['m_quartile']==4]))
print('Fast and Vacant: ', len(RFMDataF[RFMDataF['RFMScore']=='411']))
print('Lucky/Eco friendly Stand-by: ',len(RFMDataF[RFMDataF['RFMScore']=='144']))
print('Not Active Taxi Drivers: ',len(RFMDataF[RFMDataF['RFMScore']=='111']))

