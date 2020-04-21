#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
#Functions modified by Kevin Lee
#
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')

    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    DataDF = DataDF[startDate:endDate]
    
    MissingValues = DataDF["Discharge"].isna().sum()

    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    Qvalues = Qvalues.dropna()

    Tqmean= ((Qvalues > Qvalues.mean()).sum()) / (len( Qvalues ))

    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    Qvalues = Qvalues.dropna()
    differ = Qvalues.diff()
    absval = abs(differ).sum()
    total = Qvalues.sum()
    RBindex = absval / total
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    Qvalues=Qvalues.dropna()
    #rolling function with window defined for the problem 
    val7Q = (Qvalues.rolling(window=7).mean()).min() 
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    Qvalues=Qvalues.dropna()
    median3x = ( Qvalues > ( Qvalues.median()*3) ).sum()   
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    colname = [ 'site_no' , 'Mean Flow' , 'Peak Flow' , 'Median Flow' , 'Coeff Var' , 'Skew' 
    , 'Tqmean' , 'R-B Index' , '7Q' , '3xMedian']
    #convert original data to water year mean
    newDataDF = DataDF.resample('AS-OCT')
    
    #masking for the water year to compute
    waterYavg = newDataDF.mean()
    
    #create dataframe 
    WYDataDF = pd.DataFrame(index=waterYavg.index, columns=colname)
    
    #compute necessary variables
    WYDataDF['site_no'] = newDataDF['site_no'].min()
    WYDataDF['Mean Flow'] = newDataDF['Discharge'].mean()
    WYDataDF['Peak Flow'] = newDataDF['Discharge'].max()
    WYDataDF['Median Flow'] = newDataDF['Discharge'].median()
    WYDataDF['Coeff Var'] = (newDataDF['Discharge'].std() / newDataDF['Discharge'].mean())*100
    WYDataDF['Skew'] = newDataDF['Discharge'].apply(lambda x: stats.skew(x))
    WYDataDF['Tqmean'] = newDataDF['Discharge'].apply(lambda x: CalcTqmean(x))
    WYDataDF['R-B Index'] = newDataDF['Discharge'].apply(lambda x: CalcRBindex(x))
    WYDataDF['7Q'] = newDataDF['Discharge'].apply(lambda x: Calc7Q(x))
    WYDataDF['3xMedian'] = newDataDF['Discharge'].apply(lambda x: CalcExceed3TimesMedian(x))


    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    colname = [ 'site_no' , 'Mean Flow' , 'Coeff Var' , 'Tqmean' , 'R-B Index' ]
    
    Mon_Data=DataDF.resample('MS').mean()
    month=DataDF.resample('MS')

    MoDataDF=pd.DataFrame(index=Mon_Data.index,columns=colname)
    
    MoDataDF['site_no'] = month['site_no'].min()
    MoDataDF['Mean Flow'] = month['Discharge'].mean()
    MoDataDF['Coeff Var'] = (month['Discharge'].std()/month['Discharge'].mean())*100
    MoDataDF['Tqmean'] = month['Discharge'].apply(lambda x: CalcTqmean(x))
    MoDataDF['R-B Index'] = month['Discharge'].apply(lambda x: CalcRBindex(x))

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    AnnualAverages = WYDataDF.mean(  )
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    
    month = [3,4,5,6,7,8,9,10,11,0,1,2]
    colname = ['site_no','Mean Flow','Coeff Var','Tqmean','R-B Index']

    MonthlyAverages = pd.DataFrame( 0, index=range(1, 13), columns = colname)
    

    for i in range(12):
        MonthlyAverages.iloc[i,0]=MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[i,1]=MoDataDF['Mean Flow'][month[i]::12].mean()
        MonthlyAverages.iloc[i,2]=MoDataDF['Coeff Var'][month[i]::12].mean()
        MonthlyAverages.iloc[i,3]=MoDataDF['Tqmean'][month[i]::12].mean()
        MonthlyAverages.iloc[i,4]=MoDataDF['R-B Index'][month[i]::12].mean()

    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])

    #Annual metrics   
    #extract wildcat data
    Wild = WYDataDF['Wildcat']
    Wild['Station'] = 'Wildcat'
    #extract Tippe data
    Tip = WYDataDF['Tippe']
    Tip['Station'] = 'Tippe'
    #combine both data into one
    Wild = Wild.append(Tip)
    Wild.to_csv('Annual_Metrics.csv',sep=',', index=True)
        
    #monthly metrics
    Wild_mon = MoDataDF['Wildcat']
    Wild_mon['Station'] = 'Wildcat'
    Tip_mon = MoDataDF['Tippe']
    Tip_mon['Station'] = 'Tippe'
    Wild_mon = Wild_mon.append(Tip_mon)
    Wild_mon.to_csv('Monthly_Metrics.csv',sep=',', index=True)
    
    #avg anual metrics
    Wild_avg = AnnualAverages['Wildcat']
    Wild_avg['Station'] = 'Wildcat'
    Tip_avg = AnnualAverages['Tippe']
    Tip_avg['Station'] = 'Tippe'
    Wild_avg = Wild_avg.append(Tip_avg)
    Wild_avg.to_csv('Average_Annual_Metrics.txt',sep='\t', index=True)
        
    #avg mon metrics 
    Wild_avg_mon = MonthlyAverages['Wildcat']
    Wild_avg_mon['Station'] = 'Wildcat'
    Tip_avg_mon = MonthlyAverages['Tippe']
    Tip_avg_mon['Station'] = 'Tippe'
    Wild_avg_mon = Wild_avg_mon.append(Tip_avg_mon)
    Wild_avg_mon.to_csv('Average_Monthly_Metrics.txt',sep='\t', index=True)
