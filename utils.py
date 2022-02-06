import pandas as pd
from pandas import DataFrame
from pandas import Series
from os.path import join as jp
from math import radians, degrees, sin, cos, asin, acos, sqrt

def ConverttoDF(file_name: str, path = 'cabspottingdata'):
    df = pd.read_csv(jp(path, file_name), sep=' ', header=None)
    df.index = file_name.split('.')[0] + "_" + df.index.map(str)
    df.columns = ['Latitude', 'Longitude', 'Occupancy', 'Timestamp']
    df['Taxi'] = file_name.split('.')[0]
    return df

def PreviousCoordinates(df: DataFrame):
    df['PrevLatitude'] = df.shift(1)['Latitude']
    df['PrevLongitude'] = df.shift(1)['Longitude']
    df = df.dropna()
    return df

def EstimatedDistance(row: Series):
    # finding distance via longitude and latitude in Miles from
    Longitude = row['Longitude']
    Latitude = row['Latitude']
    PrevLongitude = row['PrevLongitude']
    PrevLatitude = row['PrevLatitude']
    
    if PrevLongitude == Longitude and PrevLatitude == Latitude:
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
    df['Miles'] = 0.0
    df['Miles'] = df.apply(lambda row: EstimatedDistance(row), axis=1)
    return df


def NextPassangerCoordinates(dfx: DataFrame):
    print(dfx.shape[0])
    
    dfx = dfx.reset_index()
    dfx['nextOnboardLatitude'] = dfx.Latitude
    dfx['nextOnboardLongitude'] = dfx.Longitude
     
    for (enum, row) in enumerate(dfx.iterrows()):
       
       if enum==dfx.shape[0]-1:
         break 
       #print(enum, row) 
       others =dfx.iloc[enum :]
       #print(others) 
       try:
            c1 = others.loc[others.Occupancy==1].index[0]
       except:
        c1 = enum 
        pass
       #print(c1) 
       #print(dfx.iloc[c1]) 
       dfx.at[enum,'nextOnboardLatitude']=dfx.iloc[c1]['Latitude']
       dfx.at[enum,'nextOnboardLongitude']=dfx.iloc[c1]['Longitude']
    
       
        #    break
    #print(dfx.columns) 
    return dfx 

def appendfunc(BigA, smalla):
    for a in smalla:
        BigA.append(a)
    return np.concatenate(BigA).ravel().tolist()

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from statistics import mean


def eval_metrics(gt, pr):
    print('\n', 'TEST MAE ERROR:', mae(gt, pr))
    err = [ (p - t) for (p, t) in zip(pr , gt)]
    print('\n', 'TEST ERROR:' , mean(err))

# split now target in lat and long
def splitarray(BigA):
    arr1 = []
    arr2 = []
    for a in BigA:
        arr1.append(a[0])
        arr2.append(a[1])
    return arr1, arr2 
    
