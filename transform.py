from __future__ import division
import pandas as pd
import datetime
import numpy as np
import pytz

print("Reading pings file")
pings = pd.read_csv("pings.csv")
print("Read the pings file. Transforming pings file")
pings['dayoftheweek'] = pings['ping_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Jakarta')).weekday())
pings['date'] = pings['ping_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=pytz.timezone('Asia/Jakarta')).strftime('%m/%d/%Y'))
print("Transformation on pings file done")
hoursperday = pings.groupby(['driver_id','dayoftheweek','date'])['ping_timestamp'].count()

print("Reading drivers data")
driverdf = pd.read_csv('drivers.csv')

def genderTransform(a):
    if a == 'MALE':
        return 1
    else:
        return 0
'''
From pandas cut, use the same for test transformations
Categories (5, interval[float64]): [(17.943, 29.4] < (29.4, 40.8] < (40.8, 52.2] < (52.2, 63.6] <
                                    (63.6, 75.0]]
'''
def ageTransform(a):
    if a > 17.943 and a <= 29.4:
        return 0
    if a > 29.4 and a <= 40.8:
        return 1
    if a > 40.8 and a <= 52.2:
        return 2
    if a > 52.2 and a <= 63.6:
        return 3
    else:
        return 4

print("Transforming drivers data")
driverdf['gender'] = driverdf['gender'].apply(lambda x: genderTransform(x))

print("Joining drivers data with pings data")
normpings = pd.merge(pings,driverdf,on='driver_id')[['driver_id','ping_timestamp','dayoftheweek','date','age','gender','number_of_kids']]
#print("writing joint dataset to file")
#normpings.to_csv("joined_dataset.csv")

print("Calculating time logged in by each driver")
traindf = normpings.groupby(['gender','age','number_of_kids','date','driver_id','dayoftheweek'])['ping_timestamp'].size().to_frame(name='time').reset_index()
traindf.to_csv("traindata_allcols.csv", index=False)
finaltrain = traindf[['gender','age','number_of_kids','dayoftheweek','time']]
finaltrain['timeinhrs'] = finaltrain['time'].apply(lambda x: np.around((x*15)/(3600)))
print("Bucketing age groups")
finaltrain['agebins'] = pd.cut(finaltrain['age'],5,labels=False)
finaltrain2 = finaltrain[['agebins','gender','number_of_kids','dayoftheweek','timeinhrs']]
print("Saving final training data into file")
finaltrain2.to_csv("traindata.csv",index=False)

