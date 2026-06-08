#Parallelized mirror mode detection algorithm

#-----------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#----------------------------------------------------------

import datetime 
import numpy as np
import mpmath as mp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
from sklearn.preprocessing import StandardScaler
#%matplotlib qt

import os 
from os import listdir
from os.path import isfile, join
#------------------------------------------------------------------------------------
#This function receives a list with magnetic fields data and returns a list with the values of deltaB/B

def newdeltas(magneticfieldcolumn):
    rolling4=magneticfieldcolumn.rolling(600,center=True).mean() #moving average
    element=[] 
    for index1 in np.arange(0,len(rolling4)): #for every element in the rolling average
        if rolling4[index1]==np.nan:
            element.append(np.nan)
        else:    
            delta=((2*(magneticfieldcolumn[index1]-rolling4[index1]))/rolling4[index1])
            element.append(abs(delta))
            
    return element

#-----------------------------------------------------------------------------------------
#PCAnew returns the sorted eigenvalues and eigenvectors of the covariance matrix calculated with the X data 

def PCAnew(X):
    
    #The covariance matrix does not work with NaN values
    #------------------------------------------------
    check_for_nan_1 = X['Bx'].isnull()
    data_copy = X.copy()
    for i in np.arange(0,len(X)):
        if check_for_nan_1[i]==True:
            data_copy.drop(i, axis=0, inplace=True)
    #After this we droped all the nan values in the specific interval
    #-----------------------------------------------------------------------------
    if len(data_copy)!=1 and len(data_copy)!=2 and len(data_copy)!=3 and X.dropna().empty==False: #If the interval does not contain nan numbers
    
        x_std = StandardScaler().fit_transform(data_copy)
        #------------------------------------------------
    
        # features are columns from x_std
        features = x_std.T 
        covariance_matrix = np.cov(features)
        #print(covariance_matrix)
        #Eigen Vectors and Eigen Values from Covariance Matrix
        eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    
        #In order to print in descending order the vectors...
        vec=eig_vecs.T
        sortedd=sorted(eig_vals, reverse=True)
        neww=[]
    
        for index1 in np.arange(0,3):  
             for index2 in np.arange(0,3):
                if eig_vals[index2]==sortedd[index1]:
                    neww.append(vec[index2])
    else:
        sortedd=[np.nan, np.nan, np.nan]
        neww=[[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan]]

    return sortedd, neww  #In order to compute easier it is used the transpose of the eig_vecs



#-------------------------------------------------------------------------------------------
#Angles between the vectors

import numpy as np
import numpy.linalg as LA

def angles(a,b): #Returns the angle in degrees and radians
    inner = np.inner(a, b)
    norms = LA.norm(a) * LA.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    
    return deg, rad



#-----------------------------------------------------------------------------------------
#bgmgvector is a list with the background magnetic fields vector
  
#The method consist of an iterating decomposing of the data matrix second by second
#For example if the time used is 3 minutes-->180 seconds. The method will use 90 seconds to the left and 90 seconds to the right (similar to the moving average)
#It should be noted that in this example the first 90 and last 90 points will not have any value 


#This function returns a list of lists that contains the eigenvalues ratios of the PCA method, 
#the angles phi and theta and the biggest deltaB/B value for each iteration

def function1(X,n, newdeltaB, bgmgvector): #n must be an even integer
    element=[]
    nan=[np.nan, np.nan, np.nan, np.nan, np.nan]
    L=X.shape[0] #Size of the data
    for i in np.arange(0,L): #For each data set 
        #The first two ifs are going to tell us if the interval exists and allows the iteration to work, based on the quantity of points 
        if i<(n/2):
            element.append(nan)
        elif (L-i)<=(n/2):
            element.append(nan) 
        else: #If the interval exists
            newdata=X.iloc[np.arange(i-n/2,i+1+n/2),:] #Select the interval
            newdataf=newdata.reset_index(drop=True) 
            newdata2=list(newdeltaB[i] for i in np.arange(int(i-n/2),int(i+1+n/2)))
            method=PCA2(newdataf) #PCA using the data of the interval (this fuction returns the eigenvalues and eigenvectors)
            
            if np.isnan(method[0][0])==False and np.isnan(method[1][0][0])==False: #If the PCA exist
                eigenvector1=method[1][0] #eigenvector associated with the biggest eigenvalue 
                eigenvector2=method[1][2] #eigenvector associated with the smallest eigenvalue
                ratio1=method[0][0]/method[0][1] #Goodness betwen the biggest and the middle eigenvalue
                ratio2=method[0][1]/method[0][2] #Goodness betwen the middle and smallest eigenvalue
                mgvector=bgmgvector[i] #each background magnetic field vector
                theta=angles(eigenvector2,mgvector)[0] #angle in degrees associated with the smallest eigenvalue
                phi=angles(eigenvector1,mgvector)[0] #angle in degrees associated with the biggest eigenvalue
                #If the angle is needed in radians change angles[0] to angles [1]
                
                array=[ratio1, ratio2, theta,phi, max(newdata2)]  ##### MAYBE THIS IS NOT WORKING
                #array=[ratio1, ratio2, theta,phi, newdeltaB[i]]
                element.append(array)
            else:
                element.append(nan)
    return element


#------------------------------------------------------------------------------------------------------
def getting_day(data_plasma, year, month, day):
    
    time_plasma=pd.to_datetime(data_plasma['t_utc'])
    #year=2015
    mask = time_plasma.dt.year == int(year)
    include = data_plasma[mask]
    time_plasma2=pd.to_datetime(include['t_utc'])
    #month='06'
    mask2=time_plasma2.dt.month == int(month)
    include2 = include[mask2]
    time_plasma3=pd.to_datetime(include2['t_utc'])
    #day='07'
    mask3=time_plasma3.dt.day == int(day)
    include3=include2[mask3]

    time_plasma4=pd.to_datetime(include3['t_utc'])
    
  
    return time_plasma4, include3['npl']
#------------------------------------------------------------------------------------------------------
def get_directory(folder):
    files= [a for a in listdir(folder) if isfile(join(folder,a))]
    
    return files


#------------------------------------------------------------------------------------------------------
#This function receives a Mirror Mode Waves Candidates DataFrame and returns its time intervals and how many MM are in that day

def intervals(element, n, array1):  
    #modifying the MM table
    element['tvalue'] = element.index
    element['delta'] = (element['tvalue']-element['tvalue'].shift()).fillna(0)
    zx=element.copy()
    zx2=zx.reset_index(drop=True)
    deltas=zx2['delta']
    deltas=deltas.values.tolist()
    indexA=zx2['Index']
    
    #Empty Dataframes
    my_df=pd.DataFrame()
    my_df2=pd.DataFrame()
    
    if len(element['tvalue'])!=0: #If we have MM waves
    
        #LIMIT CONDITIONS
        limits=[]
        limits2=[]
        for i in np.arange(0,len(zx2)):
            if deltas[i]>(n/2) or deltas[i]==0: #Same event
                limits.append(indexA[i])
                if i!=0:
                    limits2.append(indexA[i-1])
        limits2.append(indexA[len(deltas)-1])   
        
    
        if len(limits)!=1: #If there is not only one MM
            for i in np.arange(0,len(limits)): 
                index1=limits[i]
                index2=limits2[i]
                #my_df=my_df.append({'Beginning' : array1[index1-(n/2)], 'End' : array1[index2+(n/2)], 'Index1' : index1, 'Index2': index2}, ignore_index=True) 
                
                
                df22=pd.DataFrame({'Beginning' : [array1[index1-(n/2)]], 'End' : [array1[index2+(n/2)]], 'Index1' : [index1], 'Index2': [index2]})
                my_df=pd.concat([my_df, df22],axis=0, ignore_index=True)
             
    
        else: #If there is only one MM
            index1=limits[0]
            index2=limits2[0]
            large=len(deltas)
            #my_df=my_df.append({'Beginning' : array1[index1-(n/2)], 'End' : array1[index2+(n/2)], 'Index1' : index1, 'Index2': index2}, ignore_index=True) 
            
            df22=pd.DataFrame({'Beginning' : [array1[index1-(n/2)]], 'End' : [array1[index2+(n/2)]], 'Index1' : [index1], 'Index2': [index2]})
            my_df=pd.concat([my_df, df22],axis=0,ignore_index=True)

    else: #If we have not MM
        my_df=my_df
        limits=[]
        limits2=[]


    return my_df, len(my_df), limits,limits2



def PCA2(X):
    
    #The covariance matrix does not work with NaN values
    #------------------------------------------------
    #print(X)
    check_for_nan_1 = X['Bx'].isnull()
    check_for_nan_2 = X['By'].isnull()
    check_for_nan_3 = X['Bz'].isnull()
    data_copy = X.copy()
    for i in np.arange(0,len(X)):
        if check_for_nan_1[i]==True or check_for_nan_2[i]==True or check_for_nan_3[i]==True :
            data_copy.drop(i, axis=0, inplace=True)
    #After this we droped all the nan values in the specific interval
    #print(data_copy)
    #-----------------------------------------------------------------------------
    if len(data_copy)!=1 and len(data_copy)!=2 and len(data_copy)!=3 and X.dropna().empty==False: #If the interval does not contain nan numbers
    
        covariance_matrix=data_copy.cov()
        #print(covariance_matrix)
        #Eigen Vectors and Eigen Values from Covariance Matrix
        eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    
        #In order to print in descending order the vectors...
        vec=eig_vecs.T
        sortedd=sorted(eig_vals, reverse=True)
        neww=[]

        for index1 in np.arange(0,3):  
             for index2 in np.arange(0,3):
                if eig_vals[index2]==sortedd[index1]:
                    neww.append(vec[index2])
    else:
        sortedd=[np.nan, np.nan, np.nan]
        neww=[[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan]]

    return sortedd, neww  #In order to compute easier it is used the transpose of the eig_vecs 

#---------------------------------------------------------------------------------------------------------------


titles2=['time_epoch', 't_obt', 'usc','qv' ,'qf' ,'t_utc' ,'data source' ,'macroId' ,'npl','t']

#, sep='\s+'
#Import the plasma density dataframe
data_density=pd.read_table("C:/Users/atell/OneDrive/Escritorio/ESA/usc_v09+npl.txt", header=None, names=titles2,parse_dates=['t_utc'],low_memory=False)
data_density=data_density.iloc[np.arange(1,len(data_density)),:] #delete the first row
data_density=data_density.reset_index(drop=True)
#print(data_density)
data_plasma=data_density[[ 't_utc', 'npl']]

#print(data_plasma)

plasma_folder="C:/Users/atell/OneDrive/Escritorio/ESA/Data_LAP"

#---------------------------------------------------------

list_of_files_plasma=get_directory(plasma_folder)

#List with the paths
newlist=[]
for item in list_of_files_plasma:
    newlist.append(plasma_folder+'/'+str(item))
print(newlist)  

year_plasma=[]
month_plasma=[]
day_plasma=[]
list_of_plasma=[] #List of arrays with plasma densities

#for i in np.arange(0, 2):
for i in np.arange(0, len(newlist)):
        title2=['t_utc','?','npl','??','???','????']
        path= str(newlist[i])
        data1= pd.read_table(path, header=None, names=title2, sep=',', engine='python', parse_dates=['t_utc'])
        data2=data1.copy()
        data2=data2.iloc[np.arange(1,len(data2)),:] #delete the first row
        data2=data2.reset_index(drop=True)
        data2=data2[['t_utc','npl']]
        
        #--------------------------------------------------------------------------
        #Saving the dates
        path_time= pd.to_datetime(data2['t_utc']) #data2['t_utc']
        #It is needed to obtain the year, month and day of an specific path
        q=path_time.dt.year
        qq=path_time.dt.month
        qqq=path_time.dt.day
        year=q[0]
        month=qq[0]
        day=qqq[0]
        
        year_plasma.append(year)
        month_plasma.append(month)
        day_plasma.append(day)
        #------------------------------------------------------------------------------
        #Resample
        data2['index'] = pd.to_datetime(data2['t_utc'])
        data2.set_index('index', inplace=True)
        data2=data2.resample('1s').mean(numeric_only=True)

        data2['t_utc'] = pd.to_datetime(data2.index.values)
        data2= data2.reset_index()
        #time_lap=pd.to_datetime(data2['t_utc'])
        #---------------------------------------------------------------------------------

        #Filling the data gaps
        data2.t_utc = data2.t_utc.dt.round('1s') #round to one second for simplicity
        if data2.shape[0] < 86400: # if the number of datapoints is lower than one day:
            #print('Data gaps detected, padding array....')
            data3 = data2.rename(index=(data2['t_utc']-data2.iat[0,0].round('1D')).dt.seconds) # we will index the file new, according to the number of seconds of the data point since the start of the day
            data3 = data3.reindex(range(0,86400)) # now we just fill in the missing values
            newt = pd.date_range(start = data2.iat[0,0].round('1D'), periods = 86400, freq = '1s').values # new time array
            data3['t_utc'] = newt # now fill in the times so there is no NaT
            data2 = data3 
            del(data3)
        elif data2.shape[0] > 86400:
            error('Data file is too long, probably need to debug the code again....')
            print('Done\n')
        list_of_plasma.append(data2['npl'])


folder="C:/Users/atell/OneDrive/Escritorio/ESA/DATA_MAG"
directory= os.scandir(folder)

#This function receives a path and returns the table of the MM Waves of that day and the magnetic fields/plasma density plots
def reading_table(path, data_plasma, G1, G2, THETA, PHI, B, n):
    
    #Importing data
    titles=['Dates_and_Hours', '?', 'x', 'y', 'z', 'Bx', 'By', 'Bz', 'flag']
    data= pd.read_table(str(path), header=None, names=titles, sep='\s+', parse_dates=['Dates_and_Hours'])
    #----------------------------------------
    
    path_time=pd.to_datetime(data['Dates_and_Hours'])                             
    #It is needed to obtain the year, month and day of an specific path
    q=path_time.dt.year
    qq=path_time.dt.month
    qqq=path_time.dt.day
    year=q[0]
    month=qq[0]
    day=qqq[0]
    
    #Filling the data gaps
    data.Dates_and_Hours = data.Dates_and_Hours.dt.round('1s') #round to one second for simplicity
    if data.shape[0] < 86400: # if the number of datapoints is lower than one day:
        #print('Data gaps detected, padding array....')
        data2 = data.rename(index=(data['Dates_and_Hours']-data.iat[0,0].round('1D')).dt.seconds) # we will index the file new, according to the number of seconds of the data point since the start of the day
        data2 = data2.reindex(range(0,86400)) # now we just fill in the missing values
        newt = pd.date_range(start = data.iat[0,0].round('1D'), periods = 86400, freq = '1s').values # new time array
        data2['Dates_and_Hours'] = newt # now fill in the times so there is no NaT
        data = data2 
        del(data2)
    elif data.shape[0] > 86400:
        error('Data file is too long, probably need to debug the code again....')
    #print('Done\n')

    #Index of data will be helpful later 
    data['Index'] = data.index
    path_time=pd.to_datetime(data['Dates_and_Hours']) 

    #-------------------------------------
    
    bmodulus=[]
    bx=data['Bx']
    by=data['By']
    bz=data['Bz']

    #Calculates the magnetic field modulus for each point
    for i in np.arange(0,len(bx)):
        Bpoint=(bx[i]**2+by[i]**2+bz[i]**2)**(1/2)
        bmodulus.append(Bpoint)
        
    #Transform a list into a data column
    df = pd.DataFrame({'col':bmodulus})
    magneticfieldcolumn=df['col']
    newdeltaB=newdeltas(magneticfieldcolumn)
    
    #print(newdeltaB)
    #----------------------------------------
    
    
    #MAGNETIC FIELDS ROLLING AVERAGES
    rolling1=bx.rolling(600,center=True).mean() #Rolling average of the x-field  
    rolling2=by.rolling(600,center=True).mean() #Rolling average of the y-field  
    rolling3=bz.rolling(600,center=True).mean() #Rolling average of the z-field
    rolling4=magneticfieldcolumn.rolling(600,center=True).mean() #Rolling average of the mfield modulus 
    
    #---------------------------------------------------
    XX = data[['Bx', 'By', 'Bz']]
    
    #Creating a list with the background magnetic field vectors
    bgmgvector=[]

    for i in np.arange(0,len(rolling1)):
        bgmgvector.append((rolling1[i],rolling2[i],rolling3[i]))
        
    #PCA
    final=function1(XX,n, newdeltaB, bgmgvector) 
    
    print('final')
    
    listofratio1=[]
    listofratio2=[]
    listofthetas=[] #These lists are used for the angles plotting
    listofphis=[]
    for i in np.arange(0,len(final)):
        listofratio1.append(final[i][0])
        listofratio2.append(final[i][1])
        listofthetas.append(final[i][2])
        listofphis.append(final[i][3])
    
    #---------------------------------------------------------------

    #Selection criteria
    
    finallistthetas=[] #These lists are used for the plot section
    finallistphis=[]
    finallistdeltas=[]
    for i in np.arange(0,len(final)):
        if final[i][0]==np.nan and final[i][1]==np.nan:
            finallistthetas.append(np.nan)
            finallistphis.append(np.nan)
            finallistdeltas.append(np.nan)
            
        elif final[i][0]>=G1 and final[i][1]<=G2 and final[i][2]>=THETA and final[i][3]<=PHI and final[i][4]>=B: #Goodness 1 and 2, theta, phi, deltaB criteria
            finallistthetas.append(final[i][2])
            finallistphis.append(final[i][3])
            finallistdeltas.append(final[i][4])
        else:
            finallistthetas.append(np.nan)
            finallistphis.append(np.nan)
            finallistdeltas.append(np.nan)
    
    #------------------------------------------------------------------
    #BUILDING THE TABLE
    
    listofmgflds=[]
    #check_for_nan = finallistthetas.isnull()
    for i in np.arange(0,len(bmodulus)):
        if np.isnan(finallistthetas[i])==True:
            listofmgflds.append(np.nan)
        else:
            listofmgflds.append(bmodulus[i])
    
    listofindex=[]
    #check_for_nan = finallistthetas.isnull()
    for i in np.arange(0,len(finallistthetas)):
        if np.isnan(finallistthetas[i])!=True:
            listofindex.append(i)

    newdata=data.iloc[listofindex,:]
    newdataf=newdata[['Dates_and_Hours', 'Bx','By','Bz','Index']]
    #print(newdataf)
    tablegoodness1=[]
    tablegoodness2=[]
    tablethetas=[]
    tablephis=[]
    tablemaxdeltab=[]
    for i in listofindex:
        tablegoodness1.append(final[i][0])
        tablegoodness2.append(final[i][1])
        tablethetas.append(final[i][2])
        tablephis.append(final[i][3])
        tablemaxdeltab.append(final[i][4])
    #-------------------------------------------------------------------
    #TABLE
    d = {'Dates_and_Hours': newdataf['Dates_and_Hours'], 'Bx': newdataf['Bx'], 'By': newdataf['By'], 'Bz': newdataf['Bz'], 'Goodness1':tablegoodness1, 'Goodness2':tablegoodness2 ,'Thetas':tablethetas, 'Phis':tablephis, 'MaxDeltaB/B':tablemaxdeltab, 'Index': newdataf['Index'] }
    MMTABLE = pd.DataFrame(data=d)
    #-------------------------------------------------------------------
 
    #PLASMA DENSITY SECTION
    
    plasma_density_of_that_day=getting_day(data_plasma,year, month, day)
    
    if len(plasma_density_of_that_day[0])==0:
        #New dataframe for the plasma density 
        a1={'Dates_and_Hours':  [], 'npl': []}
        data_plasma_of_that_day= pd.DataFrame (a1, columns = ['Dates_and_Hours','npl'])
    else:
        
        #New dataframe for the plasma density 
        a1={'Dates_and_Hours':  plasma_density_of_that_day[0], 'npl': plasma_density_of_that_day[1]}
        data_plasma_of_that_day= pd.DataFrame (a1, columns = ['Dates_and_Hours','npl'])
    
        #Filling the data gaps
        data_plasma_of_that_day.Dates_and_Hours = data_plasma_of_that_day.Dates_and_Hours.dt.round('1s') #round to one second for simplicity
        if data_plasma_of_that_day.shape[0] < 86400: # if the number of datapoints is lower than one day:
            #print('Data gaps detected, padding array....')
            data2 = data_plasma_of_that_day.rename(index=(data_plasma_of_that_day['Dates_and_Hours']-data_plasma_of_that_day.iat[0,0].round('1D')).dt.seconds) # we will index the file new, according to the number of seconds of the data point since the start of the day
            data2 = data2.reindex(range(0,86400)) # now we just fill in the missing values
            newt = pd.date_range(start = data_plasma_of_that_day.iat[0,0].round('1D'), periods = 86400, freq = '1s').values # new time array
            data2['Dates_and_Hours'] = newt # now fill in the times so there is no NaT
            data_plasma_of_that_day = data2 
            del(data2)
        elif data_plasma_of_that_day.shape[0] > 86400:
            error('Data file is too long, probably need to debug the code again....')
    #----------------------------------------------------------------------------------------------------
    #New dataframe for the correlation method
    a2={'Dates_and_Hours': data['Dates_and_Hours'], 'bmodulus': bmodulus, 'rolling4': rolling4}
    data_correlation= pd.DataFrame(a2,columns = ['Dates_and_Hours', 'bmodulus', 'rolling4'])
    
    #print(MMTABLE)
    return MMTABLE, year, month, day, path_time, bmodulus,rolling1, rolling2, rolling3, rolling4, data_plasma_of_that_day['Dates_and_Hours'], data_plasma_of_that_day['npl'], listofmgflds, listofthetas, listofphis, data['Bx'], data['By'], data['Bz'], newdeltaB, data_correlation,     XX, newdeltaB, bgmgvector
   









list_of_files=get_directory(folder)

newlist=[]
for item in list_of_files:
    newlist.append(folder+'/'+str(item))
print(newlist)  


#used in our study
G1=3
G2=6
THETA=70
PHI=20
B=1
n=30







# SPLIT WORK
my_list = newlist[rank::size]

# LOCAL STORAGE
MM_candidates = []
MM_final_table = []
count = []

years = []
months = []
days = []

array1=[]  # path_time
array2=[]  # bmodulus
array3=[]  # rolling1
array4=[]  # rolling2
array5=[]  # rolling3
array6=[]  # rolling4
array7=[]  # plasma density
array8=[]
array9=[]  # magnetic fields
array10=[]
array11=[]
array12=[]  # bx
array13=[]  # by
array14=[]  # bz
array15=[]  # deltaB
array16=[]  # correlation

mm1 = []


# PARALLEL LOOP

for item in my_list:

    print('lel')
    print(item)
    element = reading_table(
        item, data_plasma, G1, G2, THETA, PHI, B, n
    )

    MM_candidates.append(element[0])
    years.append(element[1])
    months.append(element[2])
    days.append(element[3])

    array1.append(element[4])
    array2.append(element[5])
    array3.append(element[6])
    array4.append(element[7])
    array5.append(element[8])
    array6.append(element[9])
    array7.append(element[10])
    array8.append(element[11])
    array9.append(element[12])
    array10.append(element[13])
    array11.append(element[14])
    array12.append(element[15])
    array13.append(element[16])
    array14.append(element[17])
    array15.append(element[18])
    array16.append(element[19])

    mm1.append(element[0])

    table = intervals(element[0], n, element[4])

    MM_final_table.append(table[0])
    count.append(table[1])

    print(f"[Rank {rank}] {item} -> {table[1]} MM waves")


# GATHER RESULTS
all_MM_candidates = comm.gather(MM_candidates, root=0)
all_MM_final_table = comm.gather(MM_final_table, root=0)
all_count = comm.gather(count, root=0)

all_years = comm.gather(years, root=0)
all_months = comm.gather(months, root=0)
all_days = comm.gather(days, root=0)


# MERGE
if rank == 0:

    MM_candidates = [x for sub in all_MM_candidates for x in sub]
    MM_final_table = [x for sub in all_MM_final_table for x in sub]
    count = [x for sub in all_count for x in sub]

    years = [x for sub in all_years for x in sub]
    months = [x for sub in all_months for x in sub]
    days = [x for sub in all_days for x in sub]

    print("========================")
    print("MPI FINISHED")
    print("Total MM candidates:", len(MM_candidates))
    print("Total events:", sum(count))
    print("========================")        
    
