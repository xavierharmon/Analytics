##Packages needed for this custom script
import datetime
import itertools
import json
import matplotlib.pyplot as plt
import math
from math import sqrt
import numpy as np
import os
import pandas as pd
from pandas.io.json import json_normalize
import requests
from requests.auth import HTTPBasicAuth
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import sys
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
import textwrap


#Text Colors
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BrightBlack = '\u001b[30;1m'
    BrightRed = '\u001b[31;1m'
    BrightGreen = '\u001b[32;1m'
    BrightYellow = '\u001b[33;1m'
    BrightBlue = '\u001b[34;1m'
    BrightMagenta = '\u001b[35;1m'
    BrightCyan = '\u001b[36;1m'
    BrightWhite = '\u001b[37;1m'
    Highlight = '\033[0;30;43m'
    END = '\033[0m'

#Parameters for the package
class parameters:
    attempts_max = 3
    zero = 0

#Global Variables for the package
group = ''
y = ''
timevariable = ''
testdate = ''
resamplefreq = ''
startdata = pd.DataFrame()
forecaststeps = 12
aggregate = 'Mean'
splitdf = 'Y'


#Setting up text wrapping for the hints and tips throughout the script. There are three different text wrapping types here
#wrapper is a general text wrapper that inserts line breaks at or before 100 characters, whichever comes first
#wrapper_indent is a wrapper for any text that is indented below a header. It will be denoted by an indention text >>...
#wrapper_head is a wrapper for any text that acts as a heading for a section or block of text, it is denoted by *...
wrapper = textwrap.TextWrapper(width = 100)
prefix = color.BOLD + '   >> ' + color.END
heading = color.BOLD + '*  ' + color.END
wrapper_indent = textwrap.TextWrapper(initial_indent=prefix, width=100, subsequent_indent=' '*len('   >> '))
wrapper_head = textwrap.TextWrapper(initial_indent=heading, width=100, subsequent_indent=' '*len('*  '))

#Error handling customized for the script
class InvalidYear(Exception):
    def __init__(self, beginyear_cpi, endyear_cpi):
        self.beginyear_cpi = beginyear_cpi
        self.endyear_cpi = endyear_cpi
        iy_message = f'Please choose a year between {self.beginyear_cpi2} and {self.endyear_cpi2}.'
        super().__init__(iy_message)

class InvalidMonth(Exception):
    def __init__(self):
        im_message = f'You have chosen a month that has not yet occurred, please try again.'
        super().__init__(im_message)

class MissingDataFrame(Exception):
    def __init__(self, previousfunctionname):
        self.previousfunctionname = previousfunctionname
        mdf_message = f'The required dataframe is empty, please be sure the function {previousfunctionname} has been properly run.'
        super().__init__(mdf_message)

class ExceededAttempts(Exception):
    def __init__(self):
        excatt_message = f'You have exceeded the number of attempts, please check your inputs and try again.'
        super().__init__(excatt_message)

##Custom Functions for this package
##This package works with the ARIMA model specifically to ensure that no matter what date is input to begin the forecast, we capture the start of the week on sunday
def last_sunday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead >= 0: # Target day already happened this week
        days_ahead -= 7
    return d + datetime.timedelta(days_ahead)

##This function makes sure that no matter what date is input into the ARIMA model we get the first day of the month if the user selects MonthStart for their aggregation
def first_day_of_month(date):
    date = datetime.datetime.strptime(date, '%m-%d-%Y').date()
    first_day = datetime.datetime(date.year, date.month, 1)
    return first_day.strftime('%Y-%m-%d')

#This function will present the data with columns in the format they need to be in
#for the timeseries function dates need to be in date format not strings
#If there are groups in the data they need to be strings
#This function also normalizes the data using the CPI index (consumer price index)
#The CPI is gathered using an API call, and using the developer key we only have 500 API calls per day
def import_data():
    global startdata
    filepath_attempts = parameters.zero
    while filepath_attempts < parameters.attempts_max:
        filepath = input(wrapper.fill('Please place the complete file path here with the file name for your dataset in .csv or .xlsx format.'))
        if os.path.exists(filepath):
            break
        else:
            filepath_attempts += 1
            print(wrapper.fill(f'The file path you have tried to import is not valid, please check to make sure you have the correct file path with the file name and try again. You have {3-filepath_attempts} attempts remaining.'))
            if filepath_attempts == parameters.attempts_max:
                raise(ExceededAttempts)
                exit()
    filepath_last_chars = filepath[-3:]
    if filepath_last_chars == 'csv':
        startdata = pd.read_csv(filepath)
    elif filepath_last_chars == 'xlsx':
        startdata = pd.read_xlsx(filepath)
    else:
        print(wrapper.fill('Please make sure your file is in .csv or .xlsx format for the import data function and try again.'))
    startdata = pd.read_csv(fr'{filepath}')
    startdata.columns = startdata.columns.str.lower()
    global y
    global timevariable
    global group
    global testdate
    global resamplefreq
    global splitdf
    global aggregate
    timevar_attempts = parameters.zero
    while timevar_attempts < parameters.attempts_max:
        timevariablebox = input('What is the time variable you wish to use for your model?')
        timevariable = timevariablebox.lower()
        if timevariable in startdata.columns:
            break
        else:
            timevar_attempts += 1
            print(f'Time variable cannot be found in data file you imported.Please try again. You have {3-timevar_attempts} attempts remaining before the script will close.')
            if timevar_attempts == parameters.attempts_max:
                raise(ExceededAttempts)
                exit()
    resample_attempt = parameters.zero
    while resample_attempt < parameters.attempts_max:
        resamplefreq = input('\n'+ wrapper.fill('What time frequency do you want for your output? Type MS for Monthly and W for weekly results. If your data is already aggregated to a weekly or monthly level please choose that option.'))
        if resamplefreq in ['MS','W']:
            break
        else:
            resample_attempt += 1
            print(f'You did not select a valid frequency, please use MS for monthly time series results and W for weekly results.')
            if resample_attempt == attempts_max:
                raise(ExceededAttempts)
                exit()
    aggregate = input('\n'+ wrapper.fill('By default your data will be averaged for use in this model, if you need to sum your input data please type sum here otherwise press enter to continue.'))
    forecaststeps = input('\n'+ wrapper.fill('How many periods into the future would you like to generate your forecast for, by default the package has set this value to 12? Keep in mind the granularity of your data. Here 12 would be a year for monthly data while 52 would be a year for weekly data. Please use integers'))
    forecaststeps = int(forecaststeps)
    splitdf = input('\n'+ wrapper.fill('Will you need to split your dataset into a testing and training dataset for validating your model? y/n'))
    if splitdf.lower() == 'y':
        inputdate = input('\n'+ wrapper.fill('What date do you wish to split your dataset into a training and testing dataset? This is not the date the forecast will begin.'))
        if resamplefreq == 'MS':
            testdate = first_day_of_month(inputdate)
        else:
            inputdate = datetime.datetime.strptime(inputdate, '%m-%d-%Y').date()
            testdate = last_sunday(inputdate, 6) 
    groupsindata = input('\n'+ 'Do you have groups in your dataset? With groups you will be able to run a seperate timeseries model for each group. Please type (Y/N)')
    if groupsindata.lower() == 'y':
        group_attempt = parameters.zero
        while group_attempt < parameters.attempts_max:
            groupbox = input('\n'+ 'What is the column name for the groups in your dataset?')
            group = groupbox.lower()
            if group in startdata.columns:
                startdata[group] = startdata[group].astype(str)
                break
            else:
                group_attempt += 1
                print(f'Time variable cannot be found in data file you imported.Please try again. You have {3-group_attempt} attempts remaining before the script will close.')
                if group_attempt == parameters.attempts_max:
                    raise(ExceededAttempts)
                    exit()
    startdata[timevariable] = pd.to_datetime(startdata[timevariable])
    startdata['period'] = (startdata[timevariable].dt.strftime('%B'))
    startdata['year'] = (startdata[timevariable].dt.year)
    normalizedata = input('\n'+ 'Do you want to normalize any monetary data? Yes/No')
    if normalizedata.upper() == 'YES':
        cpi_frame = pd.DataFrame()
        headers = {'Content-type': 'application/json'}
        endyear_cpi = datetime.date.today().year
        beginyear_cpi = endyear_cpi - 10
        jsondata = json.dumps({"seriesid": ['CUUR0000SA0'],"startyear":beginyear_cpi, "endyear": endyear_cpi})
        p = requests.get('https://api.bls.gov/publicAPI/v1/timeseries/data/', data=jsondata, headers=headers, auth = HTTPBasicAuth('apikey', 'e5f82668f98943a6becb6c6dfb08841f'))
        json_data = json.loads(p.text)
        print('\n' + wrapper.fill('You are about to run a function to generate the Consumer Price Index (CPI). The CPI can be used to account for inflation in monetary data.'))
        chooseyear = input('What year do you want to index data to?')
        #Checking to make sure the year is valid otherwise raising a custom error
        if int(chooseyear) < beginyear_cpi or int(chooseyear) > endyear_cpi: 
            raise InvalidYear(beginyear_cpi, endyear_cpi)
        choosemonth = input(f'What month of {chooseyear} do you want to index data to?')
        #Checking to make sure the month is not in the future, if it is raising a custom error
        if int(chooseyear) == endyear_cpi and int(datetime.datetime.strptime(choosemonth.capitalize(),"%B").strftime("%m")) > (date.today().month - 1):
            raise InvalidMonth()
        #If the user inputs an integer instead of spelling out the month this will find the proper month text to avoid errors.
        if choosemonth.isnumeric() == True: 
            datetime_object = datetime.datetime.strptime(choosemonth, "%m")
            choosemonth = datetime_object.strftime("%B")
        for series in json_data['Results']['series']:
            cs = ["series id","year","period","value"]
            for item in series['data']:
                data_ses = np.array([series['seriesID'],item['year'], item['periodName'], item['value']])
                row_seperator = item['year'] + '_' + item['periodName']
                cpi_f = pd.DataFrame([data_ses],[row_seperator],columns = cs)
                cpi_frame = cpi_frame.append(cpi_f)
        x = cpi_frame.loc[(cpi_frame['year'] == chooseyear)&(cpi_frame['period'] == choosemonth.capitalize()), 'value'].values
        cpi_frame['CPI'] = x.astype(float)/cpi_frame['value'].astype(float)
        cpi_frame['year'] = cpi_frame['year'].astype(int)  
    if normalizedata.upper() == 'YES':
        startdata = pd.merge(startdata, cpi_frame, on = ["period", "year"], how = 'left')
        #Here make it to where user can input data to normalize
        startdata['NormalizedValue'] = startdata['Cost'] * startdata['CPI']
    return startdata

class breakthescript():
    def __init__(self):
        global group
        global y
        global timevariable
        global startdata
        global resamplefreq
        global testdate
        print('\n' + '\n' + wrapper.fill(color.BOLD + 'Multivariate Timeseries Package' + color.END)
                + '\n' + wrapper_indent.fill('You do not need to feed any variables into these functions,the variables are captured in the steps you just completed. Simply copy out whatever function you wish to run, make sure the alias (ts) is correct for your imported package and run the function.')
                + '\n' + wrapper_indent.fill('Helpful tips and instructions for each model are provided below.'))
        print('\n' + wrapper_head.fill(color.BLUE + 'Vectorized Auto Regression Model' + color.END)
                + '\n' + wrapper_indent.fill('var = vectorizedAR()'))


#This is the introduction text that will print when a user first imports the package. This will help direct any users to the step by step instructions and features of the package.
print('\n' + '\n' + wrapper.fill(f'Thank you for using the Multivariate time series package.')+ '\n')
import_data()
breakthescript()
              
class vectorizedAR:
    def __init__(self):
        global y
        global startdata
        self.data = startdata
        global group
        #group = 'hbf'
        global timevariable
        global resamplefreq
        global forecaststeps
        global aggregate
        vectorarbox = input(wrapper.fill('Are you sure you wish to continue with the Vectorized Auto-Regressive Time Series model? Type y for yes and press enter otherwise press enter to close the function.'))
        def multivariatedatatransform(self):
            mvdatacolumns = []
            mvdatacol = 'default' #Just made something up so string wouldnt be blank when we do a while loop in a few lines
            self.mvdf = self.data.copy() 
            self.mvdf = self.mvdf.set_index(self.mvdf[timevariable]) #Setting the index for our data frame to be the month/year for each observation
            self.mvdfg = pd.DataFrame()
            [print(col) for col in self.data.columns] #Printing out a list of columns in the dataframe
            print('\n' + wrapper.fill('Your column names are listed above, one by one please enter the columns you wish to use for your Multivariate Time Series Model below.') + '\n') 
            while mvdatacol != '':
                mvdatacol = input(wrapper.fill('Enter next column name here and press enter otherwise press enter. If you wish to use all columns in your dataset please type All and hit enter.'))
                mvdatacol = mvdatacol.lower()
                mvdata_attp = parameters.zero
                if mvdatacol == 'all':
                    allcoldf = startdata.drop(columns = ['date','period','year'])
                    allcoldf = allcoldf.select_dtypes(exclude = ['object'])
                    dfcolumns = allcoldf.columns.get_values()
                    mvdatacolumns = dfcolumns.tolist()
                    print('\n' + f'Your columns are {mvdatacolumns}')
                    mvdatacol = ''
                elif mvdatacol in self.data.columns:
                    mvdatacolumns.append(mvdatacol)
                elif mvdatacol == '':
                    print('\n' + f'Your columns are {mvdatacolumns}' + '\n')
                else:
                    print( '\n' + 'The column you chose cannot be found in the dataframe. Please try again.' + '\n')
            if aggregate == 'Mean':
                self.mvdfg[mvdatacolumns] = self.mvdf[mvdatacolumns].resample(resamplefreq).mean()
            else:
                self.mvdfg[mvdatacolumns] = self.mvdf[mvdatacolumns].resample(resamplefreq).sum()
            return self.mvdfg
        
        def forecast_accuracy(forecasts, actual):
            mape = np.mean(np.abs(forecasts - actual)/ np.abs(actual))
            me = np.mean(forecasts - actual)
            mae = np.mean(np.abs(forecasts-actual))
            mpe = np.mean((forecasts - actual)/actual)
            rmse = np.mean((forecasts-actual)**2)**0.5
            corr = np.corrcoef(forecasts, actual)[0,1]
            return({'mape': mape, 'me':me, 'mae':mae, 'mpe' : mpe, 'rmse':rmse, 'corr':corr})
        
        def grangers_causality_matrix(self):
            maxlags = int(self.mvdfg.shape[0] / 3)-1
            test = 'ssr_chi2test'
            verbose = False
            variables = self.mvdfg.columns
            self.grangerdf = pd.DataFrame(np.zeros((len(variables), len(variables))), columns = variables, index = variables)
            for c in self.grangerdf.columns:
                for r in self.grangerdf.index:
                    grangertest_result = grangercausalitytests(self.mvdfg[[r,c]], maxlag = maxlags, verbose = verbose)
                    p_values = [round(grangertest_result[i+1][0][test][1],4) for i in range(maxlags)]
                    if verbose:
                        print(f'Y = {r}, X = {c}, P-values = {p_values}')
                    min_p_value = np.min(p_values)
                    self.grangerdf.loc[r,c] = min_p_value
            self.grangerdf.columns = [var + '_x' for var in variables]
            self.grangerdf.index = [var + '_y' for var in variables]
            print('\n' + wrapper.fill('The Granger Causality matrix contains response (Y) variables in the rows and predictor variables (X) in the columns. This table represents relationships between our predictors and responses, more simply put for every combination of X and Y, we can use this table to determine if X does in fact cause Y. Each value in the table is a p-value for the Granger Causality Test. The null hypothesis is that the predictor(s) have no effect on the response(s). If the values in the table are less than our significance level of 0.05 then we can conclude that the predictor causes the response. When the majority, if not all, of our p-values in the table are less than our significance level we have a good candidate for using the VAR model for forecasting.') + '\n')
            print(self.grangerdf)
            #Work in a way to allow the user to get to the data frame and put instructions here
            return self.grangerdf
        
        def cointegration_test(self):
            self.coint_df = pd.DataFrame()
            alpha = 0.05
            maxlags = int(self.mvdfg.shape[0] / 3)-1
            out = coint_johansen(self.mvdfg, 0, maxlags)
            d = {'0.90':0, '0.95':1, '0.99':2}
            traces = out.lr1
            cvts = out.cvt[:, d[str(1-alpha)]]
            def adjust(val, length= 6): return str(val).ljust(length)

            # Summary
            coint_data = []
            coint_index = []
            for col, trace, cvt in zip(self.mvdfg.columns, traces, cvts):
                coint_index.append(col)
                coint_data.append([round(trace,2), cvt, trace>cvt])
            self.coint_df = pd.DataFrame(data = coint_data, index = coint_index, columns = ['Test Stat', 'Confidence at 95%', 'Significant'])
            print(self.coint_df)
            return self.coint_df
            
        
        def modelorder(self):
            maxl = 12
            model = VAR(self.mvdfg)
            mo_data = []
            mo_indexed = []
            lags = range(1, len(self.mvdfg.columns)-2)
            for i in lags:
                result = model.fit(i)
                mo_indexed.append(f'Lag Order {i}')
                mo_data.append([result.aic, result.bic, result.fpe, result.hqic])
            self.mo_df = pd.DataFrame(data = mo_data, index = mo_indexed, columns = ['AIC','BIC','FPE','HQIC'])
            print(self.mo_df)
            return self.mo_df
            
        def stationaritycheck(self):
            adf_data = []
            adf_index = []
            significance_level = 0.05
            for name, column in self.mvdfg.iteritems():
                ADFtest = adfuller(column, autolag = 'AIC')
                if ADFtest[1] <= significance_level:
                    adf_hypoth = 'Reject the null hypothesis.'
                else:
                    adf_hypoth = 'Fail to reject the null hypothesis.'
                adf_index.append(name)
                adf_data.append([significance_level, ADFtest[0], ADFtest[2], ADFtest[1], adf_hypoth])
            adf_df = pd.DataFrame(data = adf_data, index = adf_index, columns = ['Significance Level','Test Statistic', 'Number Lags','P-Value', 'Result'])
            print(adf_df)
            return adf_df

           
        def varmodel(self):
            self.mvdfg.index = pd.to_datetime(self.mvdfg.index)
            self.var_predicted = pd.DataFrame()
            self.var_forecast = pd.DataFrame()
            self.var_data_train = pd.DataFrame()
            self.var_data_test = pd.DataFrame()
            maxlag = 3
            if splitdf.upper() == 'Y':
                #Validation Model
                self.var_data_train = self.mvdfg[(pd.to_datetime(self.mvdfg.index)) <= testdate]
                self.var_data_test = self.mvdfg[(pd.to_datetime(self.mvdfg.index)) > testdate]
                var_model = VAR(self.var_data_train)
                results = var_model.fit(maxlags = maxlag, ic = 'aic')
                print(results.summary())
                lag_order = results.k_ar
                var_steps = len(self.var_data_test)
                pred_values = results.forecast(self.var_data_train.values[-lag_order:], var_steps)
                self.predicted = pd.DataFrame(pred_values, index = self.mvdfg.index[-var_steps:], columns = self.mvdfg.columns)
                self.var_predicted = self.predicted
            #Forecast 
            startdate = self.mvdfg.index.max()+ pd.offsets.DateOffset(months = 1)
            maxdate = self.mvdfg.index.max() + pd.offsets.DateOffset(months = forecaststeps + 1)
            var_fc_index = np.asarray((pd.date_range(startdate, maxdate, freq = 'm').strftime('%Y-%m-01')))
            var_fc_index = pd.to_datetime(var_fc_index)
            var_forecast_model = VAR(self.mvdfg)
            fc_results = var_forecast_model.fit(maxlags = maxlag, ic = 'aic')
            print(fc_results.summary())
            fc_lag_order = fc_results.k_ar
            fc_values = fc_results.forecast(self.mvdfg.values[-fc_lag_order:], forecaststeps)
            self.forecast = pd.DataFrame(fc_values, index = var_fc_index, columns = self.mvdfg.columns)
            self.var_forecast = self.forecast
            print(self.var_forecast)
            return self.var_predicted, self.var_forecast
        
        def validation_accuracy(self):
            observations = self.var_data_test
            predictions = self.var_predicted
            validation_results = pd.DataFrame()
            print('\n' + wrapper_head.fill(color.BLUE + 'Accuracy Measures Described' + color.END)
                    + '\n' + wrapper_indent.fill('The descriptions below are to be used as a guide for interpreting the accuracy of your predictions for your testing dataset. Each metric below is accompanied by a description of what specifically it is measuring. These tools would also be useful for comparing your VAR model to nested versions of this same model or to the same model run through ARIMA or other techniques.')
                    + '\n' + wrapper_indent.fill('Mean Percentage Error (MPE): A value of 10 would indicate on average the forecast was 10% higher than the true value. This is measuring the average percentage distance of your forecasts from the truth. The lower the mpe value the more accurately your model fit the data. The mpe value can be positive or negative indicating the direction which the forecast missed on average.')
                    + '\n' + wrapper_indent.fill('Mean Absolute Percentage Error (MAPE): A value of 10 would indicate on average the forecasts distance from the true value is 10% either side. If the true value is 10 then a 10% mape would indicate on average the forecast value was 9 or 11. This differs from the MPE becaues it is expressed in terms of absolute value of the difference in the forecast and the true value. The smaller the value the closer the forecast was to the true value.')
                    + '\n' + wrapper_indent.fill('Mean Error (ME): A value of 10 would indicate that on average the forecast was off by 10 units relative to the truth. If our truth was 100 and our ME value was 10 then our forecast would be 110 on average. This value can be positive or negative indicating the direction the forecast missed the truth on average.')
                    + '\n' + wrapper_indent.fill('Mean Absolute Error (MAE): A value of 10 would indicate on average that the forecast was off by 10 units relative to the true value on average. This value can be positive or negative so if our true value was 100, a MAE of 10 would indicate the forecast was either 110 or 90. This value is expressed in terms of absolute value of the difference of the forecast and the true value.')
                    + '\n' + wrapper_indent.fill('Root Mean Squared Error: Like the MAE, the RMSE is a measure of the error in our forecasts relative to the true value. The difference is that RMSE handles outliers better if they are present. This might be considered a more robust measure of error than the MAE value.')
                    + '\n' + wrapper_indent.fill('Correlation: The correlation is used to see how closely correlated our true test values are to the predicted values. The squared version of this value is what we often call r-squared.'))
            for col in predictions.columns:
                acc = forecast_accuracy(predictions[col], observations[col])
                print('\n', 'Forecast Accuracy for ', col, '\n')
                for x in acc:
                    print(x, acc[x])
                datas = pd.DataFrame(acc, index = [col])
                validation_results = validation_results.append(datas)
            return validation_results
        

        
        
        def plot_predicted_observed(self):
            fig, axes = plt.subplots(nrows = int(len(self.mvdfg.columns)/2), ncols = 2, dpi = 100, figsize = (10,10))
            for i, (col, ax) in enumerate(zip(self.mvdfg.columns, axes.flatten())):
                self.var_predicted[col].plot(legend = True, ax = ax, label = 'Forecast')
                self.mvdfg[col].plot(legend = True, ax = ax, label = 'Observed')

                ax.set_title(col + ' - Actual vs Predicted')
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')

                ax.spines['top'].set_alpha(0)
                ax.tick_params(labelsize = 6)

            plt.tight_layout()
            #plt.show()
        
        def plot_forecast(self):
            historical_values = self.mvdfg[(pd.to_datetime(self.mvdfg.index)) <= self.mvdfg.index.max()]
            forecast_values = self.var_forecast[(pd.to_datetime(self.var_forecast.index)) > self.mvdfg.index.max()]
            historical_values.index = pd.to_datetime(historical_values.index).strftime('%Y-%m-%d')
            forecast_values.index = pd.to_datetime(forecast_values.index).strftime('%Y-%m-%d')
            for i, col in enumerate(self.var_forecast.columns):
                plt.figure(figsize = (15,10))
                plt.plot(historical_values.index, historical_values[col], label = 'Observed')
                plt.plot(forecast_values.index, forecast_values[col], label = 'forecast')
                ax = plt.axes()
                ax.set_title(col + ' - Forecast')
                plt.xticks(rotation = 90)
                plt.show()
            #plt.show()
           

        
        if vectorarbox.upper() == 'Y':
            multivariatedatatransform(self)
            varpttwo = input(wrapper.fill('The next step is to calculate the granger causality matrix, if you wish to proceed type y and press enter otherwise type any other keep and press enter.'))
            if varpttwo.upper() == 'Y':
                grangers_causality_matrix(self)
                varptthree = input('\n' + '\n' + wrapper.fill('The next step is to run a cointegration test, if you wish to proceed type y and press enter otherwise type any key and press enter to end the function.'))
            else:
                varptthree = input( '\n' + '\n' + wrapper.fill('The next step is to run a cointegration test, if you wish to proceed type y and press enter otherwise type any key and press enter to end the function.'))
            if varptthree.upper() == 'Y':
                cointegration_test(self)
                varptfour = input( '\n' + '\n' + wrapper.fill('Next we will look at the auto-regressive order of our model. If you wish to proceed type y and press enter otherwise type any key and press enter to end the function.'))
            else:
                varptfour = input( '\n' + '\n' + wrapper.fill('Next we will look at the auto-regressive order of our model. If you wish to proceed type y and press enter otherwise type any key and press enter to end the function.'))
            if varptfour.upper() == 'Y':
                modelorder(self)
                varptfive = input( '\n' + '\n' + wrapper.fill('Now we will run the model! Type y and press enter to continue otherwise type any other key and press enter to end the function.'))
            else:
                varptfive = input( '\n' + '\n' + wrapper.fill('Now we will run the model! Type y and press enter to continue otherwise type any other key and press enter to end the function.'))
            if varptfive.upper() == 'Y':
                varmodel(self)
                validation_accuracy(self)
                varptsix = input( '\n' + '\n' + wrapper.fill('If you wish to plot the predicted vs observed values for your VAR model type y and press enter otherwise type any other key and press enter.'))
            else:
                varptsix = input( '\n' + '\n' + wrapper.fill('If you wish to plot the predicted vs observed values for your VAR model type y and press enter otherwise type any other key and press enter.'))
            if varptsix.upper() == 'Y':
                plot_predicted_observed(self)
                varptseven = input( '\n' + '\n' + wrapper.fill('If you wish to plot the forecasted values for your VAR model type y and press enter otherwise press enter to close the function.'))
            else:
                varptseven = input( '\n' + '\n' + wrapper.fill('If you wish to plot the forecasted values for your VAR model type y and press enter otherwise press enter to close the function.'))
            if varptseven.upper() == 'Y':
                plot_forecast(self)
                    
                    