# Question 2. Build a predictor for taxi drivers, predicting the next place a passenger will hail a cab.

# we treat the problem as a time series one

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



# first get the data

path = 'cabspottingdata'

all_files = [file_name for file_name in listdir(path) if file_name.endswith('.txt')]
print(f'All files are {len(all_files)}')

PercFiles = 0.1
random.Random(1923).shuffle(all_files)
SelectedFiles = all_files[:int(PercFiles * len(all_files))]

print(f'\n{len(SelectedFiles)} randomly selected files, if not all')


from utils import ConverttoDF, PreviousCoordinates, EstimatedDistance, DistanceCalculation, NextPassangerCoordinates, eval_metrics, splitarray

import datetime
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

#is there a hourly or daily trend?

# what if there is a hourly or weekly thrend?
def DateExtraction(row: Series):
    dt = datetime.datetime.fromtimestamp(row['Timestamp'])
    row['day'] = dt.weekday()
    row['dayname'] = days[dt.weekday()]
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


# In order to create a new dataset as a timeseries mindset, I follow below steps
# fo each taxi separately X is the (lat,long) when occupancy is 0 while Y/target is the (lat, long) when occupancy is 1 
#print(SelectedFiles[0])

Data = pd.DataFrame()
with tqdm(desc="Process files", total=len(SelectedFiles), mininterval=10) as pbar:
    for f in SelectedFiles:
        pbar.update(1)
        df = ConverttoDF(file_name=f)
        df = ApplyDateExtraction(df)
        Data = pd.concat([Data, df])
        

print(Data.head)

 




Data.pivot_table(
    index=Data.hour.values, 
    columns=Data.dayname.values , 
    values='Occupancy',
    aggfunc='mean'
).plot(figsize=(15,4), title='Occupancy - Daily Trends', ylabel='Occupany', xlabel='Hours')

#plt.show()
plt.savefig('dailyTrend.png')
plt.close()
plt.cla()
plt.clf()

# Derive ciclic trends in hs and day

def cyclic_encoding(df, col):
    
    max_val = df[col].max()
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    
    return df.drop([col], axis=1)


#X['DayOfWeek'] = X.index.dayofweek
#X = cyclic_encoding(X, 'DayOfWeek')

# lets's also normalize our data
lat_offset = 37.0
long_offset = -122.0


def PredictNextOnboardingCoord(df):
          
        df['PredNextOnboardLatitude'] = -1.0
        df['PredNextOnboardLongitude'] = -1.0
        
        df.Latitude = df.Latitude - lat_offset
        df.Longitude = df.Longitude - long_offset
        
        df.nextOnboardLatitude = df.nextOnboardLatitude - lat_offset
        df.nextOnboardLongitude = df.nextOnboardLongitude - long_offset

        df = cyclic_encoding(df, 'hour')
        df = cyclic_encoding(df, 'day')

        
        x  = df[['Latitude', 'Longitude', 'hour_cos', 'hour_sin', 'day_cos', 'day_sin']][df['Occupancy'] == 0] 
        y  = df[['nextOnboardLatitude', 'nextOnboardLongitude']][df['Occupancy'] == 0] 

            
        #This is a timeseries approach so splitting train and test data should be first n% as train last (1-n) % as test. I select 80-20 split rule.
        TrainingSize = round(len(x) * 0.8)
        x_train = np.array(x[:TrainingSize])
        x_test = np.array(x[TrainingSize:])
        y_train = np.array(y[:TrainingSize])
        y_test = np.array(y[TrainingSize:])
        
        # use xgb herey_pred_all_lat,y_pred_all_long
        # we use XGBoost
        y_train_lat, y_train_long = splitarray(y_train)
        y_test_lat, y_test_long = splitarray(y_test)
          
        ### TRAIN GRADIENT BOOSTING ###
        boosting = GridSearchCV(estimator=GradientBoostingRegressor(random_state=33), 
                     param_grid={'max_depth': [10, 20, 30], 'n_estimators': [100, 300]}, 
                     scoring='neg_mean_squared_error', cv=3, refit=True, n_jobs=-1)
        pred_boost_lat = boosting.fit(x_train, y_train_lat).predict(x_test)
        eval_metrics(y_test_lat , pred_boost_lat )
        
        
        pred_boost_long = boosting.fit(x_train, y_train_long).predict(x_test)
        
        bestmod = boosting.best_estimator_
        print(bestmod.feature_importances_)
        print(x.columns)
        # plot                                                                                                                                                                                  
        plt.subplot()
        #plt.bar(range(len(bestmod.feature_importances_)), bestmod.feature_importances_)
        plt.bar(x.columns, bestmod.feature_importances_)
        plt.xticks(x.columns, rotation='vertical')
        plt.title("feats importance for predicting longitude")
        #plt.show()

        plt.savefig("feats_importance_for_long_model.png")
        plt.close()
        plt.cla()
        plt.clf()
        eval_metrics(y_test_long , pred_boost_long )
        
        x_all =  df[['Latitude', 'Longitude', 'hour_cos', 'hour_sin', 'day_cos', 'day_sin']]
        y_all = df[['nextOnboardLatitude', 'nextOnboardLongitude']]
        
        y_pred_all_lat  = boosting.fit(x_train, y_train_lat).predict(x_all) 
        y_pred_all_long  = boosting.fit(x_train, y_train_long).predict(x_all) 
        #print('boosting y_pred_all_lat,y_pred_all_long', y_pred_all_lat,y_pred_all_long )
        with tqdm(desc="Process rows", total=df.shape[0], mininterval=10) as pbar:
            for (enum, row) in enumerate(df.iterrows()):
                #print(row[1])
                clat =  row[1].Latitude
                clong = row[1].Longitude
                #print(clat, x_test)
                #print(clat in x_test[:,0])
                #print(clong in x_test[:,1])
                
                if clat in x_test[:,0] and clong in x_test[:,1] and row[1].Occupancy==0 :
                    #print('found something to fill', y_pred_all_lat[enum])
                    df.at[enum, 'PredNextOnboardLatitude'] = y_pred_all_lat[enum] 
                    df.at[enum, 'PredNextOnboardLongitude'] = y_pred_all_long[enum]  
 
                pbar.update(1)
        print(df.tail(5))
        return df
          
        

Data = pd.DataFrame()
with tqdm(desc="Process files", total=len(SelectedFiles), mininterval=10) as pbar:
    for f in SelectedFiles:
        pbar.update(1)
        df = ConverttoDF(file_name=f)
        print(df.columns)
        df = ApplyDateExtraction(df =df)
        print(df.columns)

        df = NextPassangerCoordinates(df)
        df = PredictNextOnboardingCoord(df)
        Data = pd.concat([Data, df])

# evaluate results in distance

from math import radians, degrees, sin, cos, asin, acos, sqrt

def EstimatedDistance(row: Series):
    # finding distance via longitude and latitude in Miles from
    Longitude = row['nextOnboardLongitude']
    Latitude = row['nextOnboardLatitude']
    PrevLongitude = row['PredNextOnboardLongitude']
    PrevLatitude = row['PredNextOnboardLatitude']
    
    if PrevLongitude == PrevLatitude == -1:
                return 0.0
        
    Longitude, Latitude, PrevLongitude, PrevLatitude = map(radians, 
                                                                    [Longitude, Latitude, 
                                                                    PrevLongitude, PrevLatitude])
    #3959 # radius of the great circle in miles...some algorithms use 3956

    

    try:
        return 3959 * (acos(sin(Latitude) * sin(PrevLatitude) + cos(Latitude) * cos(PrevLatitude) * 
                                cos(Longitude - PrevLongitude)))
    except:
        return 0.0

def DistanceCalculation(df: DataFrame):
    df['DistancePred2GTOnboardingMiles'] = 0.0
    df['DistancePred2GTOnboardingMiles'] = df.apply(lambda row: EstimatedDistance(row), axis=1)
    return df

Data = DistanceCalculation(Data)

dist = Data.DistancePred2GTOnboardingMiles.values
print(dist.size, dist.mean())
dist = dist[np.where(dist>0)] 
print(dist.size, dist.mean())

dist = dist[np.where(dist<3)]
print(dist.size, dist.mean())

plt.hist(dist, bins=100)
plt.title("distance[mile] of pred next onboarding to actual onboarding")
plt.xlabel("distance[mile]")
plt.ylabel("#")

#plt.show()
plt.savefig("DistancePredVsRealOnboardLocation.png")
plt.close()
plt.cla()
plt.clf()
