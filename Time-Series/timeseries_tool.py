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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller, kpss
import sys
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
import textwrap
import warnings


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
	y_attempts = parameters.zero
	while y_attempts < parameters.attempts_max:
		yvariablebox = input('What is the dependent (y) variable for your dataset?')
		y = yvariablebox.lower()
		if y in startdata.columns:
			break
		else:
			y_attempts += 1
			print(f'Time variable cannot be found in data file you imported.Please try again. You have {3-y_attempts} attempts remaining before the script will close.')
			if y_attempts == parameters.attempts_max:
				raise(ExceededAttempts)
				exit()
	if startdata[y].dtype == np.int64 or startdata[y].dtype == np.float64:
		'Dependent variable datatype is correct, the program will proceed.'
	else:
		startdata[y] = startdata[y].str.replace(',','')
		startdata[y] = pd.to_numeric(startdata[y])
		warnings.warn('\n' + '\n' + wrapper.fill('The specified column was not an integer or float datatype so the package has converted all datapoints in the column to an integer.'), Warning)
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
		resamplefreq = input(wrapper.fill('What time frequency do you want for your output? Type MS for Monthly and W for weekly results. If your data is already aggregated to a weekly or monthly level please choose that option.'))
		if resamplefreq in ['MS','W']:
			break
		else:
			resample_attempt += 1
			print(f'You did not select a valid frequency, please use MS for monthly time series results and W for weekly results.')
			if resample_attempt == parameters.attempts_max:
				raise(ExceededAttempts)
				exit()
	forecaststeps = input(wrapper.fill('How many periods into the future would you like to generate your forecast for? By default the package has set this value to 12, Keep in mind the granularity of your data. Here 12 would be a year for monthly data while 52 would be a year for weekly data. Please use integers'))
	forecaststeps = int(forecaststeps)
	splitdf = input(wrapper.fill('Will you need to split your dataset into a testing and training dataset for validating your model? y/n'))
	if splitdf.lower() == 'y':
		inputdate = input(wrapper.fill('What date do you wish to split your dataset into a training and testing dataset? This is not the date the forecast will begin.'))
		if resamplefreq == 'MS':
			testdate = first_day_of_month(inputdate)
		else:
			inputdate = datetime.datetime.strptime(inputdate, '%m-%d-%Y').date()
			testdate = last_sunday(inputdate, 6) 
	groupsindata = input('Do you have groups in your dataset? With groups you will be able to run a seperate timeseries model for each group. Please type (Y/N)')
	if groupsindata.lower() == 'y':
		group_attempt = parameters.zero
		while group_attempt < parameters.attempts_max:
			groupbox = input('What is the column name for the groups in your dataset?')
			group = groupbox.lower()
			if group in startdata.columns:
				startdata[group] = startdata[group].astype(str)
				break
			else:
				group_attempt += 1
				print(f'Time group variable cannot be found in data file you imported.Please try again. You have {3-group_attempt} attempts remaining before the script will close.')
				if group_attempt == parameters.attempts_max:
					raise(ExceededAttempts)
					exit()
	startdata[timevariable] = pd.to_datetime(startdata[timevariable])
	startdata['period'] = (startdata[timevariable].dt.strftime('%B'))
	startdata['year'] = (startdata[timevariable].dt.year)
	normalizedata = input('Do you want to normalize any monetary data? (acceptable answers are yes/no or y/n)')
	if normalizedata.upper() == 'YES' or normalizedata.upper() == 'Y':
		cpi_frame = pd.DataFrame()
		headers = {'Content-type': 'application/json'}
		endyear_cpi = date.today().year
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
	if normalizedata.upper() == 'YES' or normalizedata.upper() == 'Y':
		startdata = pd.merge(startdata, cpi_frame, on = ["period", "year"], how = 'left')
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
		print('\n' + '\n' + wrapper.fill(color.BOLD + 'Clayton DEA Team Timeseries Functions' + color.END)
				+ '\n' + wrapper_indent.fill('You do not need to feed any variables into these functions,the variables are captured in the steps you just completed. Simply copy out whatever function you wish to run, make sure the alias (ts) is correct for your imported package and run the function.')
				+ '\n' + wrapper_indent.fill('Helpful tips and instructions for each model are provided below.'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Model Selection Tool' + color.END)
				+ '\n' + wrapper_indent.fill('select = ts.selectmodel()')
				+ '\n' + wrapper_indent.fill('To return the fit statistics for every timeseries model please run the code ' + color.Highlight + 'select.rmsframe' + color.END + ' after running the above funciton'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Time Series Diagnostics' + color.END)
				+ '\n' + wrapper_indent.fill('diag = ts.diagnostics()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Naive Time Series Model' + color.END)
				+ '\n' + wrapper_indent.fill('naive_model = ts.naive()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Simple Average Time Series Model' + color.END)
				+ '\n' + wrapper_indent.fill('sa_model = ts.simpleavg()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Moving Average Time Series Model' + color.END)
				+ '\n' + wrapper_indent.fill('ma_model = ts.movingavg()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Simple Exponential Smoothing Time Series Model (weighted average only)' + color.END)
				+ '\n' + wrapper_indent.fill('ses_model = ts.ses()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Holt Linear Time Series Model (weighted average and trend)' + color.END)
				+ '\n' + wrapper_indent.fill('holt_model = ts.holt()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'Holt Winter/Exponential Smoothing Time Series Model (weighted average, trend, and seasonality)' + color.END)
				+ '\n' + wrapper_indent.fill('holtes_model = ts.HoltWinter()'))
		print('\n' + wrapper_head.fill(color.BLUE + 'ARIMA Time Series Model' + color.END)
				+ '\n' + wrapper_indent.fill('arima_model = ts.ARIMA()'))


#This is the introduction text that will print when a user first imports the package. This will help direct any users to the step by step instructions and features of the package.
print('\n' + '\n' + wrapper.fill(f'Thank you for using the {color.BrightBlue}Clayton DEA&G{color.END} time series package.')+ '\n')
import_data()
breakthescript()

##Time Series Diagnostics
##This function will run diagnostics and test assumptions for the time series models you wish to run, the goal here is to be able to look at the data from a variety of 
##functions like simply plotting the data or looking at distributions or testing the mean and standard deviation of your time series data.
class diagnostics:
	def __init__(self):
		global y
		global startdata 
		self.data = startdata
		global group
		global timevariable 
		global forecaststeps
		print('Please select the function you wish to run. The options are:' + '\n'
				+ 'AD Fuller' + '\n'
				+ 'ANOVA' + '\n'
				+ 'acf' + '\n'
				+ 'Time series plots')
		diagbox = input('Which function would you like to run? Input is case sensitive')
		def tsdiag(self,alpha = 0.05):
			change_alpha = input(wrapper.fill(color.BOLD + 'Do you wish to change the significance level for your AD Fuller test? By default it is set to 95%.' + color.END))
			if change_alpha == '':
				alpha = alpha
			else:
				alpha = float(change_alpha)
			anova_filename = input('Do you have a file location you want to save your ADFuller Results in txt format?')
			print(color.BOLD + 'ADFuller test for time series data' + color.END)
			text =  'One of the assumptions of Time Series modeling methods is that data be stationary. Stationary data are considered stationary if they do not have trend or seasonal effects. When data are stationary, summary statistics like mean and variance remain constant over time. If data is non-stationary it should be corrected by removing trends and seasonal effects.'
			print(wrapper.fill(text = text) + '\n')
			if group == '':
				ADFtest = adfuller(self.data[y].values, autolag = 'AIC')
				if anova_filename == '':
					print(color.BOLD + y + color.END,'\n',f'ADF:    {ADFtest[0]}','\n',f'p-value: {ADFtest[1]}')
					for key, value in ADFtest[4].items():
						print(' Critical Values:', f'   {key},{value}')
					if ADFtest[1] <= alpha:
						print(color.UNDERLINE + color.BrightGreen + ' Reject the null hypothesis, the data is stationary.' + color.END,'\n')
					else:
						print(color.UNDERLINE + color.BrightRed + ' Fail to reject the null hypothesis, the data is non-stationary.' + color.END,'\n')
				else:
					adf_text = open(f'{anova_filename}ADFuller_results.txt','w')
					print(y,'\n',f'ADF:    {ADFtest[0]}','\n',f'p-value: {ADFtest[1]}', file = adf_text)
					for key, value in ADFtest[4].items():
						print(' Critical Values:', f'   {key},{value}', file = adf_text)
					if ADFtest[1] <= alpha:
						print(' Reject the null hypothesis, the data is stationary.','\n', file = adf_text)
					else:
						print(' Fail to reject the null hypothesis, the data is non-stationary.','\n', file = adf_text)
					adf_text.close()                
			else:
				diagnostics_group = self.data.groupby(group)
				for g in diagnostics_group.groups:
					diag_group = diagnostics_group.get_group(g)
					ADFtest = adfuller(diag_group[y].values, autolag = 'AIC')
					if anova_filename == '':    
						print(color.BOLD + f'{g}' + color.END,'\n',f'ADF:    {ADFtest[0]}','\n',f'p-value: {ADFtest[1]}')
						for key, value in ADFtest[4].items():
							print(' Critical Values:', f'   {key},{value}')
						if ADFtest[1] <= alpha:
							print(color.UNDERLINE + color.BrightGreen + 'Reject the null hypothesis, the data is stationary.' + color.END,'\n')
						else:
							print(color.UNDERLINE + color.BrightRed + 'Fail to reject the null hypothesis, the data is non-stationary.' + color.END,'\n')
					else:
						adf_group = open(f'{anova_filename}ADFuller_results_{g}.txt','w')
						print(g ,'\n',f'ADF:    {ADFtest[0]}','\n',f'p-value: {ADFtest[1]}', file = adf_group)
						for key, value in ADFtest[4].items():
							print(' Critical Values:', f'   {key},{value}', file = adf_group)
						if ADFtest[1] <= alpha:
							print('Reject the null hypothesis, the data is stationary.','\n', file = adf_group)
						else:
							print('Fail to reject the null hypothesis, the data is non-stationary.','\n', file = adf_group)
						adf_group.close()
		def tsanova(self):
			anova_x = input(wrapper.fill('What is your x-variable for the anova model?'))
			anova_filename = input(wrapper.fill('If you wish to save the results of your ANOVA test to a folder on your local drive them input the file path here.'))
			print(color.BOLD + 'ANOVA test with groupings' + color.END)
			text = 'The ANOVA text looks for differences in the means for your categorical variable. One category will be captured in the "Intercept" term and the other paramter estimates will show the difference(s) from that mean. If you use this grouping model, a seperate anova model will be run for each group.'
			print(wrapper.fill(text = text) + '\n')
			if group == '':
				ANOVA = ols('self.data[y] ~ C(self.data[anova_x])', data = self.data).fit()
				if anova_filename == '':
					print(ANOVA.summary())
				else:
					anova_file = open(f'{anova_filename}ANOVA_results.txt','w')
					anova_file.write(ANOVA.summary().as_text())
					anova_file.close()
			else:
				anova_groups = self.data.groupby(group)
				for g in anova_groups.groups:
					anova_group = anova_groups.get_group(g)
					ANOVA_for_groups = ols('anova_group[y] ~ C(anova_group[str(anova_x)])', data = anova_group).fit()
					if anova_filename == '':
						print(ANOVA_for_groups.summary())
					else:
						anova_groups_file = open(f'{anova_filename}ANOVA_results_{g}.txt','w')
						anova_groups_file.write(ANOVA_for_groups.summary().as_text())
						anova_groups_file.close

		def acf(self, numlags = forecaststeps, pacf = 'False'):
			change_lags = input('If you wish to change the number of lags for your Autocorrelation plot please enter an integer here.')
			if change_lags == '':
				numlags = numlags
			else:
				numlags = int(change_lags)
			change_pacf = input('If you wish to change the plot to the Partial Autocorrelation Function please input Yes and hit enter.')
			if change_pacf == 'Yes':
				pacf = 'True'
			else:
				pacf = pacf
			acf_filename = input('Do you want to save your acf plots to a PDF file on your local drive? If so put the file path here.')
			if group == '':
				if pacf == 'True':
					plt.figure(figsize = (12,6))
					plot_pacf(self.data[y].dropna(), lags = numlags)
					pacfplot = plt.title('PACF Plot')
					if acf_filename != '':
						figs = pacfplot.get_figure()
						figs.savefig(f'{acf_filename}_PACF_Plot.pdf')
						plt.close()
				else:
					plt.figure(figsize = (14,7))
					plot_acf(self.data[y].dropna(), lags = numlags)
					acfplot = plt.title('ACF Plot')
					if acf_filename != '':
						figs = acfplot.get_figure()
						figs.savefig(f'{acf_filename}_ACF_Plot.pdf')
						plt.close()
			else:
				grouped = self.data.groupby(group)
				if len(group) > 10 and acf_filename == '':
					text = input(f'Based on your selected group there will be {len(group)} plots printed. Python experiences rendering issues with many plots. If you wish to proceed press enter, otherwise put your filepath here with a filename at the end. It is recommended to create a folder specifically for this project and then create subfolders for each of your plots.')
					if text != '':
						acf_filename = input('Put the file path here and press enter.')
				else:
					for g in grouped.groups:
						gdata = grouped.get_group(g)
						if pacf == 'True':
							fig, ax = plt.subplots(2, figsize = (12,6))
							plot_pacf(gdata[y].dropna(), lags = numlags)
							pacfplot = plt.title('PACF Plot')
							if acf_filename != '':
								figs = pacfplot.get_figure()
								figs.savefig(f'{acf_filename}PACF_Plot_for{g}.pdf')
								plt.close()
						else:
							plt.figure(figsize = (14,7))
							plot_acf(gdata[y].dropna(), lags = numlags)
							acfplot = plt.title(f'{g}')
							if acf_filename != '':
								figs = acfplot.get_figure()
								figs.savefig(f'{acf_filename}ACF_Plot_for_{g}.pdf')
								plt.close()

		def timeseriesplot(self):
			if group == '':
				plots_filename = input('Do you wish to save your plots to a local file? If so place the file path here and press enter.')
				print(wrapper.fill(f'These plots show the change in your Y variable over time. This is useful for visualizing the data and looking for any trends that might exist.'))
				plt.figure(figsize = (20,10))
				sns.lineplot(x = timevariable, y = y, data = self.data, legend = 'full')
				tsplot = plt.title('Time Series Plot')
				if plots_filename != '':
					figs = tsplot.get_figure()
					figs.savefig(f'{plots_filename}_timeseries_plots')
			else:
				plot_groups = self.data.groupby(group)
				plots_filename = input('Do you wish to save your plots to a local file? If so place the file path here and press enter.')
				print(color.BOLD + 'Time Series Plots' + color.END)
				print(wrapper.fill(f'These plots show the change in your Y variable over time. This is useful for visualizing the data and looking for any trends that might exist. You chose {color.BOLD}{group}{color.END} as your column to group by, you should expect {len(plot_groups.groups)} plots.'))
				for g in plot_groups.groups:
					plot_group = plot_groups.get_group(g)
					plt.figure(figsize = (20,10))
					sns.lineplot(x =timevariable, y = y, data = plot_group, legend = 'full').set_title(g)
					tsplot = plt.title(f'Time Series Plot for {g}')
					if plots_filename != '':
						figs = tsplot.get_figure()
						figs.savefig(f'{plots_filename}_timeseries_plots_for_{g}.pdf')
		if  diagbox == 'AD Fuller':
			tsdiag(self)
		elif diagbox == 'ANOVA':
			tsanova(self)
		elif diagbox == 'acf':
			acf(self)
		elif diagbox == 'Time series plots':
			timeseriesplot(self)
		else:
			sys.exit(color.BrightRed + 'Input is case sensitive, please make sure you type your selection as written in the list.' + color.END)

class selectmodel:      
	def __init__(self):
		global startdata
		self.data = startdata
		global y
		global timevariable
		global testdate
		global group
		global resamplefreq
		global forecaststeps
		self.test_final = pd.DataFrame()
		self.train_final = pd.DataFrame()
		self.naive_y_hat = pd.DataFrame()
		self.select_frame = pd.DataFrame()
		self.rmsframe = pd.DataFrame()
		selectbox = input('Are you sure you wish to run the model selection function? y/n')
		def selectmodelfit(self):
			if group == '':
				print(wrapper.fill('The RMSE is how well our model fit the data so we would want to choose the model that has the lowest RMSE. Please be aware of the model itself though, sometimes the simpler models may have the lowest RMSE but upon visualization may not provide the best fit for trends and seasonality.'))
				test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
				train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
				test_data.timevariable = pd.to_datetime(test_data[timevariable])
				test_data_reindex = test_data.set_index(test_data[timevariable])
				##Come back and add the option to sum the data
				self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
				self.test_final
				train_data.timevariable = pd.to_datetime(train_data[timevariable])
				train_data_reindex = train_data.set_index(train_data[timevariable])
				self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
				self.train_final

				#Naive Model and stats
				naive_train = np.asarray(self.train_final.Y)
				self.naive_y_hat = self.test_final.copy()
				self.naive_y_hat['yhat'] = naive_train[len(naive_train) - 1]
				rms_naive = round(math.sqrt(mean_squared_error(self.test_final.Y, self.naive_y_hat.yhat)),4)

				#Simple Average Model
				sa_train = np.asarray(self.train_final.Y)
				self.simple_avg_yhat = self.test_final.copy()
				self.simple_avg_yhat['yhat'] = sa_train.mean()
				rms_simple_avg = round(math.sqrt(mean_squared_error(self.test_final.Y, self.simple_avg_yhat.yhat)),4)

				#Moving Average Model
				self.moving_avg_yhat = self.test_final.copy()
				self.moving_avg_yhat['yhat'] = self.train_final['Y'].rolling(12).mean().iloc[-1]
				rms_moving_avg = round(math.sqrt(mean_squared_error(self.test_final.Y, self.moving_avg_yhat.yhat)),4)

				#Simple Exponential Smoothing Model
				self.ses_yhat = self.test_final.copy()
				ses_fit = SimpleExpSmoothing(np.asarray(self.train_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
				self.ses_yhat['yhat'] = ses_fit.forecast(len(self.test_final))
				rms_simpleexpsmoothing = round(math.sqrt(mean_squared_error(self.test_final.Y, self.ses_yhat.yhat)),4)

				#Holt Linear Model
				self.holt_yhat = self.test_final.copy()
				holt_fit = Holt(np.asarray(self.train_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
				self.holt_yhat['yhat'] = holt_fit.forecast(len(self.test_final))
				rms_holt = round(math.sqrt(mean_squared_error(self.test_final.Y, self.holt_yhat.yhat)),4)

				#Holt winter Model
				self.holtwinter_yhat = self.test_final.copy()
				holtwinter_fit = ExponentialSmoothing(np.asarray(self.train_final['Y']), seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
				self.holtwinter_yhat['yhat'] = holtwinter_fit.forecast(len(self.test_final))
				rms_holtwinter = round(math.sqrt(mean_squared_error(self.test_final.Y, self.holtwinter_yhat.yhat)),4)  

				#Arima Model
				arima_train = np.asarray(self.train_final.Y)
				arima_test = self.test_final.copy()
				Initial_ARIMA_Estimates_select = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
				optimalarimaest_select = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
				model_results_select= []
				Means_final_select = pd.DataFrame({'Y' : [] })
				ps = ds = qs = range(0,2)
				forecaststeps = [12]
				ss = forecaststeps
				spdq = list(itertools.product(ps,ds,qs))
				sseasonal_pdqs = list(itertools.product(ps,ds,qs,ss))
				means_select = pd.DataFrame()
				for param in spdq:
					for param_seasonal in sseasonal_pdqs:
						try:
							mod = sm.tsa.statespace.SARIMAX(arima_train, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
							arima_results_select = mod.fit()
							Initial_ARIMA_Estimates_select = Initial_ARIMA_Estimates_select.append({'Param1' : param[0],'Param2' : param[1], 'Param3' : param[2], 'Seasonal1' : param_seasonal[0],'Seasonal2' : param_seasonal[1],'Seasonal3' : param_seasonal[2],'Seasonal4' : param_seasonal[3],'AIC' : arima_results_select.aic}, ignore_index = True)
						except:
							continue
				optimalarimaest_select = Initial_ARIMA_Estimates_select.loc[Initial_ARIMA_Estimates_select['AIC'].idxmin()]
				Means_final_select = self.train_final.copy()
				Means_final_select['Param1'], Means_final_select['Param2'], Means_final_select['Param3'], Means_final_select['Seasonal1'], Means_final_select['Seasonal2'], Means_final_select['Seasonal3'], Means_final_select['Seasonal4'] = [optimalarimaest_select['Param1'],optimalarimaest_select['Param2'],optimalarimaest_select['Param3'], optimalarimaest_select['Seasonal1'],optimalarimaest_select['Seasonal2'], optimalarimaest_select['Seasonal3'],optimalarimaest_select['Seasonal4']]
				for col in ['Param1','Param2','Param3','Seasonal1','Seasonal2','Seasonal3','Seasonal4']:
					Means_final_select[col] = Means_final_select[col].astype(int)
				arimamod_final_select = sm.tsa.statespace.SARIMAX(Means_final_select.Y, order = (Means_final_select.Param1[0],Means_final_select.Param2[0],Means_final_select.Param3[0]), seasonal_order = (Means_final_select.Seasonal1[0], Means_final_select.Seasonal2[0], Means_final_select.Seasonal3[0],Means_final_select.Seasonal4[0]), enforce_stationarity = False, enforce_invertibility = False).fit()
				Means_final_select['yhat'] = arimamod_final_select.get_prediction().predicted_mean
				forecast = arimamod_final_select.get_forecast(steps = len(arima_test)).predicted_mean
				forecast_data = forecast.to_frame()
				forecast_data.columns = ['Forecast']
				arima_final = arima_test.merge(forecast_data, left_index = True, right_index = True)
				rms_arima = round(math.sqrt(mean_squared_error(arima_final.Y, arima_final.Forecast)),4)
   

				modelname = np.array(['Naive','Simple Average','Moving Average', 'SES','Holt','Holt Winter', 'ARIMA'])
				rmses = np.array([rms_naive, rms_simple_avg, rms_moving_avg, rms_simpleexpsmoothing, rms_holt, rms_holtwinter, rms_arima])
				self.rmsframe = pd.DataFrame(rmses, modelname, columns = ['RMSE'])
				pd.set_option('display.max_rows', self.rmsframe.shape[0])
				print(self.rmsframe)
				rootmeanframe = pd.DataFrame(self.rmsframe)
			else:
				print(wrapper.fill('For each group, the RMSE helps us understand how well our model fit the data so we would want to choose the model that has the lowest RMSE.') + '\n' + ' **This package currently does not consider the ARIMA or Seasonal ARIMA model**')
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
					train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					self.test_final['Group'] = g
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					self.train_final['Group'] = g
					#Naive Model
					naive = np.asarray(self.train_final.Y)
					self.naive_y_hat = self.test_final.copy()
					self.naive_y_hat['yhat'] = naive[len(naive) - 1] 
					rms_naive = round(math.sqrt(mean_squared_error(self.test_final.Y, self.naive_y_hat.yhat)),4)
                    
					#Simple Average Model
					self.simple_avg_yhat = self.test_final.copy()
					self.simple_avg_yhat['yhat'] = self.train_final['Y'].mean()
					rms_simple_avg = round(math.sqrt(mean_squared_error(self.test_final.Y, self.simple_avg_yhat.yhat)),4)
                   
					#Moving Average Model
					self.moving_avg_yhat = self.test_final.copy()
					self.moving_avg_yhat['yhat'] = self.train_final['Y'].rolling(12).mean().iloc[-1]
					rms_moving_avg = round(math.sqrt(mean_squared_error(self.test_final.Y, self.moving_avg_yhat.yhat)),4)
                    
					#Simple Exponential Smoothing Model
					self.ses_yhat = self.test_final.copy()
					ses_fit = SimpleExpSmoothing(np.asarray(self.train_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
					self.ses_yhat['yhat'] = ses_fit.forecast(len(self.test_final))
					rms_simpleexpsmoothing = round(math.sqrt(mean_squared_error(self.test_final.Y, self.ses_yhat.yhat)),4)
                    
					#Holt Linear Model
					self.holt_yhat = self.test_final.copy()
					holt_fit = Holt(np.asarray(self.train_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
					self.holt_yhat['yhat'] = holt_fit.forecast(len(self.test_final))
					rms_holt = round(math.sqrt(mean_squared_error(self.test_final.Y, self.holt_yhat.yhat)),4)
                    
					#Holt Winter Model
					self.holtwinter_yhat = self.test_final.copy()
					holtwinter_fit = ExponentialSmoothing(np.asarray(self.train_final['Y']), seasonal_periods = 12, trend = 'add', seasonal = 'add').fit()
					self.holtwinter_yhat['yhat'] = holtwinter_fit.forecast(len(self.test_final))
					rms_holtwinter = round(math.sqrt(mean_squared_error(self.test_final.Y, self.holtwinter_yhat.yhat)),4)  
                    
					#ARIMA Model
					arima_train = np.asarray(self.train_final.Y)
					arima_test = self.test_final.copy()
					Initial_ARIMA_Estimates_select = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
					optimalarimaest_select = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
					model_results_select= []
					Means_final_select = pd.DataFrame({'Y' : [] })
					ps = ds = qs = range(0,2)
					forecaststeps = [12]
					ss = forecaststeps
					spdq = list(itertools.product(ps,ds,qs))
					sseasonal_pdqs = list(itertools.product(ps,ds,qs,ss))
					means_select = pd.DataFrame()
					for param in spdq:
						for param_seasonal in sseasonal_pdqs:
							try:
								mod = sm.tsa.statespace.SARIMAX(arima_train, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
								arima_results_select = mod.fit()
								Initial_ARIMA_Estimates_select = Initial_ARIMA_Estimates_select.append({'Param1' : param[0],'Param2' : param[1], 'Param3' : param[2], 'Seasonal1' : param_seasonal[0],'Seasonal2' : param_seasonal[1],'Seasonal3' : param_seasonal[2],'Seasonal4' : param_seasonal[3],'AIC' : arima_results_select.aic}, ignore_index = True)
							except:
								continue
					optimalarimaest_select = Initial_ARIMA_Estimates_select.loc[Initial_ARIMA_Estimates_select['AIC'].idxmin()]
					Means_final_select = self.train_final.copy()
					Means_final_select['Param1'], Means_final_select['Param2'], Means_final_select['Param3'], Means_final_select['Seasonal1'], Means_final_select['Seasonal2'], Means_final_select['Seasonal3'], Means_final_select['Seasonal4'] = [optimalarimaest_select['Param1'],optimalarimaest_select['Param2'],optimalarimaest_select['Param3'], optimalarimaest_select['Seasonal1'],optimalarimaest_select['Seasonal2'], optimalarimaest_select['Seasonal3'],optimalarimaest_select['Seasonal4']]
					for col in ['Param1','Param2','Param3','Seasonal1','Seasonal2','Seasonal3','Seasonal4']:
						Means_final_select[col] = Means_final_select[col].astype(int)
					arimamod_final_select = sm.tsa.statespace.SARIMAX(Means_final_select.Y, order = (Means_final_select.Param1[0],Means_final_select.Param2[0],Means_final_select.Param3[0]), seasonal_order = (Means_final_select.Seasonal1[0], Means_final_select.Seasonal2[0], Means_final_select.Seasonal3[0],Means_final_select.Seasonal4[0]), enforce_stationarity = False, enforce_invertibility = False).fit()
					Means_final_select['yhat'] = arimamod_final_select.get_prediction().predicted_mean
					forecast = arimamod_final_select.get_forecast(steps = len(arima_test)).predicted_mean
					forecast_data = forecast.to_frame()
					forecast_data.columns = ['Forecast']
					arima_final = arima_test.merge(forecast_data, left_index = True, right_index = True)
					rms_arima = round(math.sqrt(mean_squared_error(arima_final.Y, arima_final.Forecast)),4)

                    
					modelname = np.array(['', 'Naive','Simple Average','Moving Average', 'SES', 'Holt','Holt Winter', 'ARIMA'])
					rmses = np.array([f'{g}', rms_naive, rms_simple_avg, rms_moving_avg, rms_simpleexpsmoothing,  rms_holt, rms_holtwinter, rms_arima])
					rmsdata = pd.DataFrame(rmses, modelname, columns = ['RMSE'])
					self.rmsframe = self.rmsframe.append(rmsdata)
				pd.set_option('display.max_rows', self.rmsframe.shape[0])
				self.rmsframe
				print(self.rmsframe)
			return self.rmsframe
		if selectbox.lower() == 'y':
			selectmodelfit(self)
		else:
			sys.exit(color.BrightRed + 'Input is case sensitive, please make sure you type your selection as written in the list.' + color.END)

class ARIMA:
	def __init__(self, exog = '', upper_pdq = 1):
		global resamplefreq
		global startdata
		self.data = startdata
		global group
		global testdate
		global timevariable 
		global y
		global forecaststeps
		global aggregate
		global splitdf
		splitdf = 'Y'
		self.upper_pdq = upper_pdq
		self.testdataset = pd.DataFrame()
		self.exog = exog
		if group == '':
			self.Initial_ARIMA_Estimates = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
		else:
			self.Initial_ARIMA_Estimates = pd.DataFrame({'Group' : [], 'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})
		if group == '':
			self.optimalarimaest = pd.DataFrame({'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})    
		else:
			self.optimalarimaest = pd.DataFrame({'Group' : [], 'Param1' : [], 'Param2' : [], 'Param3' : [], 'Seasonal1' : [],'Seasonal2' : [],'Seasonal3' : [],'Seasonal4' : [], 'AIC' : []})  
		self.Predictions_final = pd.DataFrame()
		self.model_results = []
		self.Means_final = pd.DataFrame({'Y' : [] })
		if group =='':
			print(wrapper.fill(f'You are running a single ARIMA model with{color.BOLD} {y} {color.END}as your dependent variable'))
		else:
			text_groups = self.data.groupby(group)
			print(wrapper.fill(f'You are running a dynamic ARIMA model with {color.BOLD}{y}{color.END} as your dependent variable. You have chosen to group your data on the column {color.BOLD}{group}{color.END}, meaning one ARIMA model will be built for every {group}. You should expect {color.BOLD}{len(text_groups)}{color.END} result sets.'))
		if testdate != '':
			if group == '':
				data_test = self.data[(pd.to_datetime(self.data[timevariable]) < testdate)]
				self.testdataset = self.testdataset.append(data_test, ignore_index = True)
			else:
				test_groups = self.data.groupby(group)
				for g in test_groups.groups:
					test = test_groups.get_group(g)
					data_test = test[(pd.to_datetime(test[timevariable]) < testdate)]
					self.testdataset = self.testdataset.append(data_test,ignore_index = True)
		arimabox = input('Are you sure you wish to run the ARIMA seasonal model? If so type yes, otherwise press enter.')
		def optimal_arima_groups(self):
			if group == '':
				if not sys.warnoptions:
					import warnings
					warnings.simplefilter('ignore')
				p = d = q = range(0,self.upper_pdq+1)
				seasonal = input(wrapper.fill('How many periods do you wish to include for a seasonal trend in your data? For example, if you are using monthly data, 12 would be a yearly trend.'))
				s = seasonal
				s = [pd.to_numeric(s)]
				pdq = list(itertools.product(p,d,q))
				seasonal_pdqs = list(itertools.product(p, d, q, s))
				Means = pd.DataFrame({'Y' : [] })
				if testdate == '':
					self.testdataset['Years'] = pd.DatetimeIndex(self.data[timevariable]).year
					arima_reindex = self.data.set_index(self.data[timevariable])
				else:
					self.testdataset['Years'] = pd.DatetimeIndex(self.testdataset[timevariable]).year
					arima_reindex = self.testdataset.set_index(self.testdataset[timevariable])
				if aggregate == 'Mean':
					Means['Y'] = arima_reindex[y].resample(resamplefreq).mean()
				else:
					Means['Y'] = arima_reindex[y].resample(resamplefreq).sum()
				if self.exog != '' and aggregate == 'Mean':
					Means['ex'] = arima_reindex[self.exog].resample(resamplefreq).mean()
				elif self.exog != '' and aggregate != 'Mean':
					Means['ex'] = arima_reindex[self.exog].resample(resamplefreq).sum()
				print(wrapper.fill('This function will output every combination of the seasonal and non-seasonal parameters with their Akaike Information Criterion (AIC). The AIC values are a test for the fit of the model and will be used to determine the optimal set of paramters. The AIC score aims to find the best fitting model for the data while avoiding over-fitting of the data set. The model with the lowest AIC value is the best fitting model.'))
				for param in pdq:
					for param_seasonal in seasonal_pdqs:
						try:
							if self.exog == '':
								mod = sm.tsa.statespace.SARIMAX(Means.Y, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
							else:
								mod = sm.tsa.statespace.SARIMAX(Means.Y, exog = Means.ex, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
							arima_results = mod.fit()
							self.Initial_ARIMA_Estimates = self.Initial_ARIMA_Estimates.append({
																'Param1' : param[0], 
																'Param2' : param[1],
																'Param3' : param[2],
																'Seasonal1' : param_seasonal[0],
																'Seasonal2' : param_seasonal[1],
																'Seasonal3' : param_seasonal[2],
																'Seasonal4' : param_seasonal[3],
																'AIC' : arima_results.aic}, ignore_index = True)
						except:
							continue
				return self.Initial_ARIMA_Estimates
			else:
				self.data[timevariable] = pd.to_datetime(self.data[timevariable])
				p = d = q = range(0,self.upper_pdq+1)
				seasonal = input(wrapper.fill('How many periods do you wish to include for a seasonal trend in your data? For example, if you are using monthly data, 12 would be a yearly trend.'))
				s = seasonal
				s = [pd.to_numeric(s)]
				pdq = list(itertools.product(p,d,q))
				seasonal_pdqs = list(itertools.product(p, d, q, s))
				Means = pd.DataFrame({'Group' : [], 'Y' : [] })
				if testdate == '':
					arima_groups = self.data.groupby([group])
				else:
					arima_groups = self.testdataset.groupby([group])
				N = len(arima_groups.groups)
				print('\n' + wrapper.fill('This function will output every combination of the seasonal and non-seasonal parameters with their Akaike Information Criterion (AIC). The AIC values are a test for the fit of the model and will be used to determine the optimal set of paramters. The AIC score aims to find the best fitting model for the data while avoiding over-fitting of the data set. The model with the lowest AIC value is the best fitting model.')+'\n')
				print(f'You are about to run {((self.upper_pdq+1)**6)*(len(seasonal)) * (N)} loops, if you wish to cancel type end and hit enter otherwise hit enter.')
				if input() == 'end':
					sys.exit('You chose to end the script.')
				for g in arima_groups.groups:
					arima_g = arima_groups.get_group(g)
					arima_g['Years'] = pd.DatetimeIndex(arima_g[timevariable]).year
					arima_g_reindex = arima_g.set_index(arima_g[timevariable])                    
					if aggregate == 'Mean':
						Means['Y'] = arima_g_reindex[y].resample(resamplefreq).mean()
					else:
						Means['Y'] = arima_g_reindex[y].resample(resamplefreq).sum()
					if self.exog != '' and aggregate == 'Mean':
						Means['ex'] = arima_g_reindex[self.exog].resample(resamplefreq).mean()
					elif self.exog != '' and aggregate != 'Mean':
						Means['ex'] = arima_g_reindex[self.exog].resample(resamplefreq).sum()
					Means['Group'] = g
					modelgroups = Means.groupby(['Group'])
					for g in modelgroups.groups:
						model_g = modelgroups.get_group(g)
						for param in pdq:
							for param_seasonal in seasonal_pdqs:
								try:
									if self.exog == '':
										mod = sm.tsa.statespace.SARIMAX(model_g.Y, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
									else:
										mod = sm.tsa.statespace.SARIMAX(model_g.Y, exog = model_g.ex, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
									arima_results = mod.fit()
									self.Initial_ARIMA_Estimates = self.Initial_ARIMA_Estimates.append({'Group' : g,
																		'Param1' : param[0], 
																		'Param2' : param[1],
																		'Param3' : param[2],
																		'Seasonal1' : param_seasonal[0],
																		'Seasonal2' : param_seasonal[1],
																		'Seasonal3' : param_seasonal[2],
																		'Seasonal4' : param_seasonal[3],
																		'AIC' : arima_results.aic}, ignore_index = True)
								except:
									continue
				return self.Initial_ARIMA_Estimates

		def optimal_parameters(self):
			if self.Initial_ARIMA_Estimates.empty == True:
				raise MissingDataFrame('optimal_arima_groups')
			else:    
				print(wrapper.fill(('This function will choose the optimal parameters based on the AIC values. The output of this model will be a list containing the optimal paramters. These parameters will be used in the optimal_arima() function.')))
			if group == '':
				self.optimalarimaest = self.Initial_ARIMA_Estimates.loc[self.Initial_ARIMA_Estimates['AIC'].idxmin()]
				return self.optimalarimaest
			else:
				self.optimalarimaest = self.Initial_ARIMA_Estimates.loc[self.Initial_ARIMA_Estimates.groupby('Group')['AIC'].idxmin()]
				for col in ['Param1','Param2','Param3','Seasonal1','Seasonal2','Seasonal3','Seasonal4']:
					self.optimalarimaest[col] = self.optimalarimaest[col].astype(int)
				return self.optimalarimaest
				print('\n' + 'The new dataframe with the optimal parameter estimates is "optimalarimaest"')

		def optimal_arima(self):
			arimasummary_filename = input('If you wish to save the summary results to a file put the filepath here without a filename.')
			if resamplefreq == 'MS':
				forecaststeps = input('How many periods into the future would you like to run the forecast for (your periods are months 12 = 1 year)? (please use integers)')
			else:
				forecaststeps = input('How many periods into the future would you like to run the forecast for (your periods are weeks 52 = 1 year)? (please use integers)')
			if group == '':
				self.Predictions_final = pd.DataFrame()
				self.Means_final = pd.DataFrame({ 'Y' : [] })
				self.data['Years'] = pd.DatetimeIndex(self.data[timevariable]).year
				arima_reindex = self.data.set_index(self.data[timevariable])
				if aggregate == 'Mean':
					self.Means_final['Y'] = arima_reindex[y].resample(resamplefreq).mean()
				else:
					self.Means_final['Y'] = arima_reindex[y].resample(resamplefreq).sum()
				if self.exog != '' and aggregate == 'Mean':
					self.Means_final['ex'] = arima_reindex[self.exog].resample(resamplefreq).mean()
				elif self.exog != '' and aggregate != 'Mean':
					self.Means_final['ex'] = arima_reindex[self.exog].resample(resamplefreq).sum()
				self.Means_final['Param1'], self.Means_final['Param2'], self.Means_final['Param3'], self.Means_final['Seasonal1'], self.Means_final['Seasonal2'], self.Means_final['Seasonal3'], self.Means_final['Seasonal4'] = [self.optimalarimaest['Param1'],
																			self.optimalarimaest['Param2'],
																			self.optimalarimaest['Param3'],
																			self.optimalarimaest['Seasonal1'],
																			self.optimalarimaest['Seasonal2'],
																			self.optimalarimaest['Seasonal3'],
																			self.optimalarimaest['Seasonal4']]
				for col in ['Param1','Param2','Param3','Seasonal1','Seasonal2','Seasonal3','Seasonal4']:
					self.Means_final[col] = self.Means_final[col].astype(int)
				if self.exog == '':
					mod = sm.tsa.statespace.SARIMAX(self.Means_final.Y, 
											order = (self.Means_final.Param1[0],
													self.Means_final.Param2[0],
													self.Means_final.Param3[0]), 
											seasonal_order = (self.Means_final.Seasonal1[0],
																self.Means_final.Seasonal2[0],
																self.Means_final.Seasonal3[0],
																self.Means_final.Seasonal4[0]), 
											enforce_stationarity = False, 
											enforce_invertibility = False)
					fittingModel = mod.fit()
					self.model_results = mod.fit()
					#Validating the Model
					if arimasummary_filename == '':
						print( '\n' +  f' The results are '  + '\n' + f'{fittingModel.summary()}' + '\n')
					else:
						arima_summary_txt = open(f'{arimasummary_filename}/_ARIMA_results.txt','w')
						arima_summary_txt.write(fittingModel.summary().as_text())
						arima_summary_txt.close
					if splitdf.upper() == 'Y':
						pred = fittingModel.get_prediction(start = pd.to_datetime(testdate), exog = self.exog, dynamic = True, full_results = True) 
						self.Means_final['Predicted'] = pred.predicted_mean
						self.Means_final[['Lower CI','Upper CI']] = pred.conf_int()      
					forecast = fittingModel.get_forecast(steps = int(forecaststeps))
					forecast_data = forecast.predicted_mean.to_frame()
					forecast_data.columns = ['Forecast']
					self.Means_final = self.Means_final.append(forecast_data)
					self.Means_final[['Lower CI F','Upper CI F']] = forecast.conf_int()
					self.Predictions_final = self.Predictions_final.append(self.Means_final)
					return self.Predictions_final, self.Means_final
				else:
					mod = sm.tsa.statespace.SARIMAX(self.Means_final.Y, exog = self.Means_final.ex, 
																order = (self.Means_final.Param1[0],
																		self.Means_final.Param2[0],
																		self.Means_final.Param3[0]), 
																seasonal_order = (self.Means_final.Seasonal1[0],
																					self.Means_final.Seasonal2[0],
																					self.Means_final.Seasonal3[0],
																					self.Means_final.Seasonal4[0]), 
																enforce_stationarity = False, 
																enforce_invertibility = False)
					fittingModel = mod.fit()
					self.model_results = mod.fit()
					if splitdf.upper() == 'Y':
						pred = fittingModel.get_prediction(start = pd.to_datetime(testdate), exog = self.exog, dynamic = True, full_results = True) 
						self.Means_final['Predicted'] = pred.predicted_mean
						self.Means_final[['Lower CI','Upper CI']] = pred.conf_int()
	#				forecast = fittingModel.get_forecast(steps = forecaststeps, exog = self.Means_final.ex)
					if arimasummary_filename == '':
						print( '\n' +  f' The results are '  + '\n' + f'{fittingModel.summary()}' + '\n')
					else:
						arima_summary_txt = open(f'{arimasummary_filename}/_ARIMA_results.txt','w')
						arima_summary_txt.write(fittingModel.summary().as_text())
						arima_summary_txt.close 
	#				forecast_data = forecast.predicted_mean.to_frame()
	#				forecast_data.columns = ['Forecast']
	#				self.Means_final = self.Means_final.append(forecast_data)
	#				self.Means_final[['Lower CI F','Upper CI F']] = forecast.conf_int()
					self.Predictions_final = self.Predictions_final.append(self.Means_final)
					return self.Predictions_final, self.Means_final
			else:
				arima_groups = self.data.groupby([group])
				self.Predictions_final = pd.DataFrame()
				self.Means_final = pd.DataFrame({'Group' : [], 'Y' : [] })
				for ag in arima_groups.groups:
					arima_g_final = arima_groups.get_group(ag)
					arima_g_final['Years'] = pd.DatetimeIndex(arima_g_final[timevariable]).year
					arima_g_reindex_final = arima_g_final.set_index(arima_g_final[timevariable])
					if aggregate == 'Mean':
						self.Means_final['Y'] = arima_g_reindex_final[y].resample(resamplefreq).mean()
					else:
						self.Means_final['Y'] = arima_g_reindex_final[y].resample(resamplefreq).sum()
					if self.exog != '' and aggregate == 'Mean':
						self.Means_final['ex'] = arima_g_reindex_final[self.exog].resample(resamplefreq).mean()
					elif self.exog != '' and aggregate != 'Mean':
						self.Means_final['ex'] = arima_g_reindex_final[self.exog].resample(resamplefreq).sum()
					self.Means_final['Group'] = ag
					modelgroups_final = self.Means_final.groupby(['Group'])
					for mg in modelgroups_final.groups:
						model_g_final = modelgroups_final.get_group(mg)
						model_g_final['Close_Dates'] = model_g_final.index
						Data = model_g_final.reset_index().merge(self.optimalarimaest, on = 'Group').set_index(timevariable)
						data_final = Data.groupby(['Group'])
						for g in data_final.groups:
							mod_data_final = data_final.get_group(g)
							if self.exog == '':
								mod = sm.tsa.statespace.SARIMAX(model_g_final.Y, 
																order = (mod_data_final.Param1[0],
																		mod_data_final.Param2[0],
																		mod_data_final.Param3[0]), 
																seasonal_order = (mod_data_final.Seasonal1[0],
																					mod_data_final.Seasonal2[0],
																					mod_data_final.Seasonal3[0],
																					mod_data_final.Seasonal4[0]), 
																enforce_stationarity = False, 
																enforce_invertibility = False)
								fittingModel = mod.fit()
								self.model_results = mod.fit()
								if splitdf.upper() == 'Y':
									pred = fittingModel.get_prediction(start = pd.to_datetime(testdate), dynamic = True, full_results = True) 
									mod_data_final['Predicted'] = pred.predicted_mean
									mod_data_final[['Lower CI','Upper CI']] = pred.conf_int()
								if arimasummary_filename == '':
									print( f' The results for {g} are '  + '\n' + f'{fittingModel.summary()}' + '\n')
								else:
									arima_results_groups = open(f'{arimasummary_filename}/_ARIMA_results_for_{g}', 'w')
									arima_results_groups.write(fittingModel.summary().as_text())
									arima_results_groups.close
								forecast = fittingModel.get_forecast(steps = int(forecaststeps))                              
								forecast_data = forecast.predicted_mean.to_frame()
								forecast_data.columns = ['Forecast']
								mod_data_final = mod_data_final.append(forecast_data)
								mod_data_final['Group'] = mod_data_final['Group'].fillna(method='ffill')
								mod_data_final[['Lower CI F','Upper CI F']] = forecast.conf_int()
								self.Predictions_final = self.Predictions_final.append(mod_data_final)
							else:
								mod = sm.tsa.statespace.SARIMAX(model_g_final.Y, exog = model_g_final.ex,
																order = (mod_data_final.Param1[0],
																		mod_data_final.Param2[0],
																		mod_data_final.Param3[0]), 
																seasonal_order = (mod_data_final.Seasonal1[0],
																					mod_data_final.Seasonal2[0],
																					mod_data_final.Seasonal3[0],
																					mod_data_final.Seasonal4[0]), 
																enforce_stationarity = False, 
																enforce_invertibility = False)
								fittingModel = mod.fit()
								self.model_results = mod.fit()
								if splitdf.upper() == 'Y':
									pred = fittingModel.get_prediction(start = pd.to_datetime(testdate), dynamic = True, full_results = True) 
									mod_data_final['Predicted'] = pred.predicted_mean
									mod_data_final[['Lower CI','Upper CI']] = pred.conf_int()
								if arimasummary_filename == '':
									print( f' The results for {g} are '  + '\n' + f'{fittingModel.summary()}' + '\n')
								else:
									arima_results_groups = open(f'{arimasummary_filename}/_ARIMA_results_for_{g}', 'w')
									arima_results_groups.write(fittingModel.summary().as_text())
									arima_results_groups.close
								#forecast = fittingModel.get_forecast(steps = forecaststeps)
								#forecast_data = forecast.predicted_mean.to_frame()
								#forecast_data.columns = ['Forecast']
								#mod_data_final = mod_data_final.append(forecast_data)
								mod_data_final['Group'] = mod_data_final['Group'].fillna(method='ffill')
								#mod_data_final[['Lower CI F','Upper CI F']] = forecast.conf_int()
								cis = pred.conf_int()
								self.Predictions_final = self.Predictions_final.append(mod_data_final)
				return self.Predictions_final, mod_data_final

		def plotPredObserved(self):
			#predictions_final_testplot = self.Predictions_final[(pd.to_datetime(self.Predictions_final.index) <= self.testdate)] #helps trim down the plot so there is no empty space.
			predictions_final_testplot = self.Predictions_final
			arima_po_plots_filename = input('If you wish to save your plots to a folder input the filepath here without a filename.')
			if group == '':
				#axis_observed = predictions_final_testplot.Y.plot(label = 'Observed', figsize = (14,7))
				#predictions_final_testplot.Predicted.plot(ax = axis_observed, label = 'Forecast', alpha = 0.95, figsize = (14,7))
                
				plt.figure(figsize = (14,7))
				plt.plot(predictions_final_testplot.index, predictions_final_testplot['Y'], label = 'Observed')
				plt.plot(predictions_final_testplot.index, predictions_final_testplot['Predicted'], label = 'Predicted')
				plt.fill_between(predictions_final_testplot.index,
											predictions_final_testplot['Lower CI'],
											predictions_final_testplot['Upper CI'], color = 'k', alpha = 0.2)
				arimapo_plots = plt.title('Predicted vs Observed', fontsize = 20)
				plt.legend()
				if arima_po_plots_filename != '':
					arimapo_figs = arimapo_plots.get_figure()
					arimapo_figs.savefig(f'{arima_po_plots_filename}/_timeseries_plots.pdf')
					plt.close()
			else:
				poplotdata = predictions_final_testplot.groupby('Group')
				for plotgroup in poplotdata.groups:
					poplot_groups = poplotdata.get_group(plotgroup)
					#axis_observed = poplot_groups.Y.plot(label = 'Observed', figsize = (14,7))
					#poplot_groups.Predicted.plot(ax = axis_observed, label = 'Forecast', alpha = 0.95, figsize = (14,7))
					plt.figure(figsize = (14,7))
					plt.plot(poplot_groups.index, poplot_groups['Y'], label = 'Observed')
					plt.plot(poplot_groups.index, poplot_groups['Predicted'], label = 'Predicted')
					plt.fill_between(poplot_groups.index,
												poplot_groups['Lower CI'],
												poplot_groups['Upper CI'], color = 'k', alpha = 0.2)
					arimapo_plots_group = plt.title(f'ARIMA Validation for {plotgroup}')
					plt.legend()
					if arima_po_plots_filename != '':
						arimagrouppo_figs = arimapo_plots_group.get_figure()
						arimagrouppo_figs.savefig(f'{arima_po_plots_filename}/timeseries_plots_for_{plotgroup}.pdf')
						plt.close()

		def plotForecast(self):
			arima_forecast_filename = input('If you wish to save your forecast plots to a local folder put the filepath here without a filename.')
			if group == '':
				plt.figure(figsize = (14,7))
				plt.plot(self.Predictions_final.index, self.Predictions_final['Y'], label = 'Observed')
				plt.plot(self.Predictions_final.index, self.Predictions_final['Forecast'], label = 'Forecast')
				#self.Predictions_final.Y.plot(label = 'Observed', figsize = (14,7))
				#self.Predictions_final.Forecast.plot(label = 'Forecast', alpha = 0.95, figsize = (14,7))
				plt.fill_between(self.Predictions_final.index,
											self.Predictions_final['Lower CI F'],
											self.Predictions_final['Upper CI F'], color = 'k', alpha = 0.2)
				arimaf_plots = plt.title('Forecast', fontsize = 20)
				plt.legend()
				if arima_forecast_filename != '':
					arima_forecast_figs = arimaf_plots.get_figure()
					arima_forecast_figs.savefig(f'{arima_forecast_filename}/_ARIMA_forecast_plot.pdf')
					plt.close()
			else:
				poplotdataf = self.Predictions_final.groupby('Group')
				for plotgroupf in poplotdataf.groups:
					poplot_groupsf = poplotdataf.get_group(plotgroupf)
					#poplot_groupsf.Y.plot(label = 'Observed', figsize = (14,7))
					#poplot_groupsf.Forecast.plot(label = 'Forecast', alpha = 0.95, figsize = (14,7))
					plt.figure(figsize = (14,7))
					plt.plot(poplot_groupsf.index, poplot_groupsf['Y'], label = 'Observed')
					plt.plot(poplot_groupsf.index, poplot_groupsf['Forecast'], label = 'Forecast')
					plt.fill_between(poplot_groupsf.index,
												poplot_groupsf['Lower CI F'],
												poplot_groupsf['Upper CI F'], color = 'k', alpha = 0.2)
					arimaf_group_plots = plt.title(f'{plotgroupf}')
					plt.legend()
					if arima_forecast_filename != '':
						arimaf_group_figs = arimaf_group_plots.get_figure()
						arimaf_group_figs.savefig(f'{arima_forecast_filename}/_ARIMA_forecast_plot_for_{plotgroupf}.pdf')
						plt.close()
		if arimabox.upper() == 'YES':
			optimal_arima_groups(self)
			if self.Initial_ARIMA_Estimates.empty == False:
				optimal_parameters(self)
				if self.optimalarimaest.empty == False:
					optimal_arima(self)
					if splitdf.upper() == 'Y':
						plotpobox = input('Do you wish to plot the predicted versus observed valued from the arima model? If so type yes')
						if plotpobox.upper() == 'YES':
							plotPredObserved(self)
							pfbox = input('Do you wish to plot the forecasted values for your ARIMA model? If so type yes')
						else:
							pfbox = input('Do you wish to plot the forecasted values for your ARIMA model? If so type yes')
					else:
						pfbox = input('Do you wish to plot the forecasted values for your ARIMA model? If so type yes')
					if pfbox.upper() == 'YES':
						plotForecast(self)
		else:
			sys.exit(color.BrightRed + 'You have chosen not to run the ARIMA model.' + color.END)

class naive:        
	def __init__(self):
		global y
		global timevariable
		global testdate
		global group
		global startdata
		global resamplefreq
		global aggregate
		global splitdf
		self.data = startdata
		self.results_naive = pd.DataFrame()
		self.test_final = pd.DataFrame()
		self.train_final = pd.DataFrame()
		self.naive_forecast_final = pd.DataFrame()
		self.naive_forecast_yhat = pd.DataFrame()
		self.naive_forecast_results = pd.DataFrame()
		self.training = pd.DataFrame()
		self.summary = pd.DataFrame()
		self.naive_y_hat = pd.DataFrame()
		naivebox = input('Are you sure you wish to run the naive time series model? If yes then type y and press enter otherwise press enter to quit the function.')
		def naivefit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					#Model Validation
					test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					if aggregate == 'Mean':
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					if aggregate == 'Mean':
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
					naive = np.asarray(self.train_final.Y)
					self.naive_y_hat = self.test_final.copy()
					self.naive_y_hat['yhat'] = naive[len(naive) - 1]
					rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.naive_y_hat.yhat)),4)
					self.results_naive = self.results_naive.append(self.naive_y_hat)
					self.summary = self.summary.append({'RMSE' : rms}, ignore_index = True)
					datasets = [self.train_final, self.naive_y_hat]
					final_datasets = pd.concat(datasets)
					self.training = self.training.append(final_datasets)
				#Forecasting
				forecast_data = self.data[[timevariable, y]].copy()
				forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
				forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
				if aggregate == 'Mean':
					self.naive_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
				else:
					self.naive_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
				startdate = forecast_data[timevariable].max()
				enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
				naive_forecast_value = np.asarray(forecast_data[y])
				self.naive_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
				self.naive_forecast_yhat['yhat'] = naive_forecast_value[len(naive_forecast_value)-1]
				self.naive_forecast_yhat.set_index(timevariable, inplace = True)
				naive_frames = [self.naive_forecast_final, self.naive_forecast_yhat]
				naive_dataset = pd.concat(naive_frames)
				self.naive_forecast_results = self.naive_forecast_results.append(naive_dataset)
			else:
				self.data = self.data[[timevariable, y, group]].copy()
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					#Validation
					if splitdf == 'Y':
						test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						test_data.timevariable = pd.to_datetime(test_data[timevariable])
						test_data_reindex = test_data.set_index(test_data[timevariable])
						if aggregate == 'Mean':
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
						self.test_final['Group'] = g
						train_data.timevariable = pd.to_datetime(train_data[timevariable])
						train_data_reindex = train_data.set_index(train_data[timevariable])
						if aggregate == 'Mean':
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
						self.train_final['Group'] = g
						naive = np.asarray(self.train_final.Y)
						self.naive_y_hat = self.test_final.copy()
						self.naive_y_hat['yhat'] = naive[len(naive) - 1]
						rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.naive_y_hat.yhat)),4)
						self.results_naive = self.results_naive.append(self.naive_y_hat)
						self.summary = self.summary.append({'Group' : g, 'RMSE' : rms}, ignore_index = True)
						datasets = [self.train_final, self.naive_y_hat]
						final_datasets = pd.concat(datasets)
						self.training = self.training.append(final_datasets)
					#Forecasting
					forecast_data = data_groups
					forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
					forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
					if aggregate == 'Mean':
						self.naive_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.naive_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
					self.naive_forecast_final['Group'] = g
					startdate = forecast_data[timevariable].max()
					enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
					naive_forecast_value = np.asarray(forecast_data[y])
					self.naive_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
					self.naive_forecast_yhat['yhat'] = naive_forecast_value[len(naive_forecast_value)-1]
					self.naive_forecast_yhat.set_index(timevariable, inplace = True)
					self.naive_forecast_yhat['Group'] = g
					naive_frames = [self.naive_forecast_final, self.naive_forecast_yhat]
					naive_dataset = pd.concat(naive_frames)
					self.naive_forecast_results = self.naive_forecast_results.append(naive_dataset)
				return self.training, self.summary, self.naive_forecast_results

		def plotnaive(self):
			if self.training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function naivefit(), which fits the Naive model and outputs predictions in a data frame used by the function you have chosen.') + '\n' + '\n' + color.BrightRed + 'Please run the function naivefit() and try again.' + color.END)
			else:
				naiveplot_filename = input('Do you wish to save the naive plots to a folder? If so put the file path here otherwise press enter.')
				if group == '':
					train_group = self.training[(pd.to_datetime(self.training.index)< testdate)]
					data_group = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					naiveplot = plt.title('\n' + 'Naive Validation', fontsize = 20)
					plt.legend(loc = 'best')
					if naiveplot_filename != '':
						naiveplots = naiveplot.get_figure()
						naiveplots.savefig(f'{naiveplot_filename}/naiveplot.pdf')
						plt.close()
				else:
					plot_data = self.training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						naiveplot = plt.title('\n' + f'Naive Validation for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if naiveplot_filename != '':
							naiveplots = naiveplot.get_figure()
							naiveplots.savefig(f'{naiveplot_filename}/naiveplot_{g}.pdf')
							plt.close()

		def plotnaiveforecast(self):
			if self.naive_forecast_results.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function naivefit(), which fits the Naive model and outputs predictions in a data frame used by the function you have chosen.') + '\n' + '\n' + color.BrightRed + 'Please run the function naivefit() and try again.' + color.END)
			else:
				naivefcplot_filename = input('Do you wish to save the naive plots to a folder? If so put the file path here otherwise press enter.')
				if group == '':
					plt.figure(figsize = (14,7))
					plt.plot(self.naive_forecast_results.index, self.naive_forecast_results['Y'])
					plt.plot(self.naive_forecast_results.index, self.naive_forecast_results['yhat'])
					naivefcplot = plt.title('\n' + 'Naive Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if naivefcplot_filename != '':
						naivefcplots = naivefcplot.get_figure()
						naivefcplots.savefig(f'{naivefcplot_filename}/naive_forecast_plot.pdf')
						plt.close()
				else:
					plot_data = self.naive_forecast_results.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						plt.figure(figsize = (14,7))
						plt.plot(plot_dataset.index, plot_dataset['Y'], label = 'Observed')
						plt.plot(plot_dataset.index, plot_dataset['yhat'], label = 'Forecast')
						naivefcplot = plt.title('\n' + f'Naive Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if naivefcplot_filename != '':
							naivefcplots = naivefcplot.get_figure()
							naivefcplots.savefig(f'{naivefcplot_filename}/naive_forecast_plot_for_{g}.pdf')
							plt.close()


		def rmsnaive(self):
			if self.training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function naivefit(), which fits the Naive model and outputs predictions in a data frame used by the function you have chosen.') + '\n' + '\n' + color.BrightRed + 'Please run the function naivefit() and try again.' + color.END)
			else:
				naiverms_filename = input('Do you have a local folder you wish to save your results to?')
				if group == '':
					rms_dataset = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if naiverms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						naiverms_print = open(f'{naiverms_filename}/Root_mean_squared_error_Naive.txt','w')
						print(f'The Root Mean Squared Error is {rms}', file = naiverms_print)
						naiverms_print.close()
				else:
					rms_data = self.training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index)>= testdate)]
						rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if naiverms_filename == '':
							print(f' The Root Mean Squared Error for {g} is {rms}')
						else:
							naiverms_print = open(f'{naiverms_filename}/Root_mean_squared_error_Naive.txt','w')
							print(f' The Root Mean Squared Error for {g} is {rms}', file = naiverms_print)
							naiverms_print.close()
		if naivebox.upper() == 'Y' :
			naivefit(self)
			if splitdf.upper() == 'Y':
				parttwo = input('Do you want to plot the data? Type Y for yes and N for no. Else press enter')
				if parttwo.upper() == 'Y':
					plotnaive(self)
					partthree = input('Do you want to print the RMSE for this model? Type Y for yes and N for no. Else press enter.')
				else:
					partthree = input('Do you want to print the RMSE for this model? Type Y for yes and N for no. Else press enter.')
				if partthree.upper() == 'Y':
					rmsnaive(self)
					naivepartfour = input('Do you wish to plot the Forecast for your Naive Forecasting Model? Type y for Yes and hit enter else hit enter.')
				else:
					naivepartfour = input('Do you wish to plot the Forecast for your Naive Forecasting Model? Type y for Yes and hit enter else hit enter.')
			else:
				naivepartfour = input('Do you wish to plot the Forecast for your Naive Forecasting Model? Type y for Yes and hit enter else hit enter.')
		if naivepartfour.upper() == 'Y':
			plotnaiveforecast(self)

class simpleavg:        
	def __init__(self):
		global y
		global startdata
		global timevariable
		global testdate
		global group
		global aggregate
		global splitdf
		self.data = startdata
		self.results_simple_avg = pd.DataFrame()
		self.test_final = pd.DataFrame()
		self.train_final = pd.DataFrame()
		self.training = pd.DataFrame()
		self.summary = pd.DataFrame()
		self.simple_avg_yhat = pd.DataFrame()
		self.sa_forecast_final = pd.DataFrame()
		self.sa_forecast_yhat = pd.DataFrame()
		self.sa_forecast_results = pd.DataFrame()
		sabox = input('If you wish to continue with the Simple Average timeseries model type y for Yes and hit enter otherwise hit enter to exit the function.')
		def SimpleAvgFit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					#Validating Data
					test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					if aggregate == 'Mean':
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					if aggregate == 'Mean':
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
					self.simple_avg_yhat = self.test_final.copy()
					self.simple_avg_yhat['yhat'] = self.train_final['Y'].mean()
					rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.simple_avg_yhat.yhat)),4)
					self.results_simple_avg = self.results_simple_avg.append(self.simple_avg_yhat)
					self.summary = self.summary.append({'RMSE' : rms}, ignore_index = True)
					datasets = [self.train_final, self.simple_avg_yhat]
					final_datasets = pd.concat(datasets)
					self.training = self.training.append(final_datasets)
				#forecasting
				forecast_data = startdata[[timevariable, y]].copy()
				forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
				forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
				if aggregate == 'Mean':
					self.sa_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
				else:
					self.sa_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
				startdate = forecast_data[timevariable].max()
				enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
				sa_value = np.asarray(forecast_data[y])
				self.sa_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
				self.sa_forecast_yhat['yhat'] = sa_value.mean()
				self.sa_forecast_yhat.set_index(timevariable, inplace = True )
				sa_frames = [self.sa_forecast_final, self.sa_forecast_yhat]
				sa_dataset = pd.concat(sa_frames)
				self.sa_forecast_results = self.sa_forecast_results.append(sa_dataset)
			else:
				self.data = self.data[[timevariable, y, group]].copy()
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					if splitdf.upper() == 'Y':
						#Validating Data
						test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						test_data.timevariable = pd.to_datetime(test_data[timevariable])
						test_data_reindex = test_data.set_index(test_data[timevariable])
						if aggregate == 'Mean':
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
						self.test_final['Group'] = g
						train_data.timevariable = pd.to_datetime(train_data[timevariable])
						train_data_reindex = train_data.set_index(train_data[timevariable])
						if aggregate == 'Mean':
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
						self.train_final['Group'] = g
						self.simple_avg_yhat = self.test_final.copy()
						self.simple_avg_yhat['yhat'] = self.train_final['Y'].mean()
						rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.simple_avg_yhat.yhat)),4)
						self.results_simple_avg = self.results_simple_avg.append(self.simple_avg_yhat)
						self.summary = self.summary.append({'Group' : g, 'RMSE' : rms}, ignore_index = True)
						datasets = [self.train_final, self.simple_avg_yhat]
						final_datasets = pd.concat(datasets)
						self.training = self.training.append(final_datasets)   
					#Forecasting Data
					forecast_data = data_groups
					forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
					forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
					if aggregate == 'Mean':
						self.sa_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.sa_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
					self.sa_forecast_final['Group'] = g
					startdate = forecast_data[timevariable].max()
					enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
					sa_value = np.asarray(forecast_data[y])
					self.sa_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
					self.sa_forecast_yhat['yhat'] = sa_value.mean()
					self.sa_forecast_yhat.set_index(timevariable, inplace = True)
					self.sa_forecast_yhat['Group'] = g
					sa_frames = [self.sa_forecast_final, self.sa_forecast_yhat]
					sa_dataset = pd.concat(sa_frames)
					self.sa_forecast_results = self.sa_forecast_results.append(sa_dataset)
				return self.training, self.summary

		def plotSimpleAvg(self):
			if self.training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function SimpleAvgFit(), which fits the Simple Average Time Series model and ouputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function SimpleAvgFit() and try again.' + color.END))
			else:
				saplot_filename = input('Do you wish to save your Simple Average Plots to a local folder? If so put the file name here, otherwise press enter.')
				if group == '':
					train_group = self.training[(pd.to_datetime(self.training.index)< testdate)]
					data_group = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					saplot = plt.title('\n' + 'Simple Average Validation Plot', fontsize = 20)
					plt.legend(loc = 'best')
					if saplot_filename != '':
						saplots = saplot.get_figure()
						saplots.savefig(f'{saplot_filename}/Simple_Avg_Plot.pdf')
						plt.close()
				else:
					plot_data = self.training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						saplot = plt.title('\n' + f'Simple Average Validation Plot for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if saplot_filename != '':
							saplots = saplot.get_figure()
							saplots.savefig(f'{saplot_filename}/Simple_Avg_Plot for {g}.pdf')
							plt.close()

		def plotsaforecast(self):
			if self.sa_forecast_results.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function simpleavg(), which fits the Simple Average model and outputs predictions in a data frame used by the function you have chosen.') + '\n' + '\n' + color.BrightRed + 'Please run the function simpleavg() and try again.' + color.END)
			else:
				saforecastplot_filename = input('Do you wish to save the simple average plots to a folder? If so put the file path here otherwise press enter.')
				if group == '':
					plt.figure(figsize = (14,7))
					plt.plot(self.sa_forecast_results.index, self.sa_forecast_results['Y'])
					plt.plot(self.sa_forecast_results.index, self.sa_forecast_results['yhat'])
					safcplot = plt.title('\n' + 'Simple Average Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if saforecastplot_filename != '':
						safcplots = safcplot.get_figure()
						safcplots.savefig(f'{saforecastplot_filename}/simple_average_forecast_plot.pdf')
						plt.close()
				else:
					plot_data = self.sa_forecast_results.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						plt.figure(figsize = (14,7))
						plt.plot(plot_dataset.index, plot_dataset['Y'], label = 'Observed')
						plt.plot(plot_dataset.index, plot_dataset['yhat'], label = 'Forecast')
						safcplot = plt.title('\n' + f'Simple Average Forecast for {g}', fontsize =20)
						plt.legend(loc = 'best')
						if saforecastplot_filename != '':
							safcplots = safcplot.get_figure()
							safcplots.savefig(f'{saforecastplot_filename}/Simple_average_forecast_plot_for_{g}.pdf')
							plt.close()

		def rmsSimpleAvg(self):
			if self.training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function SimpleAvgFit(), which fits the Simple Average Time Series model and ouputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function SimpleAvgFit() and try again.' + color.END))
			else:
				sarms_filename = input('Do you have a local file you wish to save the RMSE results to? If so put the file path here.')
				if group == '':
					rms_dataset = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if sarms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						simpleavg_print = open(f'{sarms_filename}/Root_mean_squared_error_SimpleAverage.txt','w')
						print(f'The Root Mean Squared Error is {rms}',file = simpleavg_print)
						simpleavg_print.close()
				else:
					rms_data = self.training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index)>= testdate)]
						rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if sarms_filename == '':
							print(f'The Root Mean Squared Error for {g} is {rms}')
						else:
							simpleavg_print = open(f'{sarms_filename}/Root_mean_squared_error_SimpleAverage.txt','w')
							print(f'The Root Mean Squared Error for {g} is {rms}', file = simpleavg_print)
							simpleavg_print.close()
		if sabox.upper() == 'Y':
			SimpleAvgFit(self)
			if splitdf.upper() == 'Y':
				saparttwo = input('Do you wish to plot the results of your predicted and observed values?')
				if saparttwo.upper() == 'Y':
					plotSimpleAvg(self)
					sapartthree = input('Do you wish to calculate the RMSE for this model?')
				else:
					sapartthree = input('Do you wish to calculate the RMSE for this model?')
				if sapartthree.upper() == 'Y':
					rmsSimpleAvg(self)
					sapartfour = input('Do you wish to plot the forecast for your Simple Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
				else:
					sapartfour = input('Do you wish to plot the forecast for your Simple Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
			else:
				sapartfour = input('Do you wish to plot the forecast for your Simple Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
		if sapartfour.upper() == 'Y':
			plotsaforecast(self)

class movingavg:        
	def __init__(self):
		global y
		global timevariable
		global testdate
		global group
		global resamplefreq
		global startdata
		global aggregate
		global splitdf
		self.data = startdata
		self.results_moving_avg = pd.DataFrame()
		self.test_final = pd.DataFrame()
		self.train_final = pd.DataFrame()
		self.training = pd.DataFrame()
		self.summary = pd.DataFrame()
		self.moving_avg_yhat = pd.DataFrame()
		self.ma_forecast_final = pd.DataFrame()
		self.ma_forecast_yhat = pd.DataFrame()
		self.ma_forecast_results = pd.DataFrame()
		mabox = input('If you wish to proceed with the Moving Average Model type y for yes otherwise press enter to end the function.')        
		def MovingAvgFit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					if aggregate == 'Mean':
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					if aggregate == 'Mean':
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
					self.moving_avg_yhat = self.test_final.copy()
					self.moving_avg_yhat['yhat'] = self.train_final['Y'].rolling(12).mean().iloc[-1]
					rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.moving_avg_yhat.yhat)),4)
					self.results_moving_avg = self.results_moving_avg.append(self.moving_avg_yhat)
					self.summary = self.summary.append({'RMSE' : rms}, ignore_index = True)
					datasets = [self.train_final, self.moving_avg_yhat]
					final_datasets = pd.concat(datasets)
					self.training = self.training.append(final_datasets)
				#Forecasting
				forecast_data = startdata[[timevariable, y]].copy()
				forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
				forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
				if aggregate == 'Mean':
					self.ma_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
				else:
					self.ma_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
				startdate = forecast_data[timevariable].max()
				enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
				ma_value = np.asarray(forecast_data[y])
				self.ma_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
				self.ma_forecast_yhat['yhat'] = self.ma_forecast_final['Y'].rolling(12).mean().iloc[-1]
				self.ma_forecast_yhat.set_index(timevariable, inplace = True )
				ma_frames = [self.ma_forecast_final, self.ma_forecast_yhat]
				ma_dataset = pd.concat(ma_frames)
				self.ma_forecast_results = self.ma_forecast_results.append(ma_dataset)
			else:
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					if splitdf.upper() == 'Y':
						#Valiating Data
						test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						test_data.timevariable = pd.to_datetime(test_data[timevariable])
						test_data_reindex = test_data.set_index(test_data[timevariable])
						if aggregate == 'Mean':
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
						self.test_final['Group'] = g
						train_data.timevariable = pd.to_datetime(train_data[timevariable])
						train_data_reindex = train_data.set_index(train_data[timevariable])
						if aggregate == 'Mean':
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
						self.train_final['Group'] = g
						self.moving_avg_yhat = self.test_final.copy()
						self.moving_avg_yhat['yhat'] = self.train_final['Y'].rolling(12).mean().iloc[-1]
						rms = round(math.sqrt(mean_squared_error(self.test_final.Y, self.moving_avg_yhat.yhat)),4)
						self.results_moving_avg = self.results_moving_avg.append(self.moving_avg_yhat)
						self.summary = self.summary.append({'Group' : g, 'RMSE' : rms}, ignore_index = True)
						datasets = [self.train_final, self.moving_avg_yhat]
						final_datasets = pd.concat(datasets)
						self.training = self.training.append(final_datasets)
					#Forecasting
					forecast_data = data_groups
					forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
					forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
					if aggregate == 'Mean':
						self.ma_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.ma_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
					self.ma_forecast_final['Group'] = g
					startdate = forecast_data[timevariable].max()
					enddate = forecast_data[timevariable].max() + pd.offsets.DateOffset(months = 12)
					self.ma_forecast_yhat[timevariable] = ((pd.date_range(startdate, enddate, freq = 'm').strftime('%Y-%m-01')))
					self.ma_forecast_yhat['yhat'] = self.ma_forecast_final['Y'].rolling(12).mean().iloc[-1]
					self.ma_forecast_yhat.set_index(timevariable, inplace = True)
					self.ma_forecast_yhat['Group'] = g
					ma_frames = [self.ma_forecast_final, self.ma_forecast_yhat]
					ma_dataset = pd.concat(ma_frames)
					self.ma_forecast_results = self.ma_forecast_results.append(ma_dataset)
				return self.training, self.summary

		def plotMovingAvg(self):
			if self.training.empty == True:
					sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function MovingAvgFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function MovingAvgFit() and try again.' + color.END))
			else:
				maplot_filename = input('Do you want to save your plots to a local folder? If so copy the file path here, otherwise hit enter.')
				if group == '':
					train_group = self.training[(pd.to_datetime(self.training.index)< testdate)]
					data_group = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					maplot = plt.title('\n' + 'Moving Average Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if maplot_filename != '':
						maplots = maplot.get_figure()
						maplots.savefig(f'{maplot_filename}/Moving_Avg_Plot.pdf')
						plt.close()
				else:
					plot_data = self.training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						maplot = plt.title('\n' + f'Moving Average Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if maplot_filename != '':
							maplots = maplot.get_figure()
							maplots.savefig(f'{maplot_filename}/Moving_Avg_Plot_for_{g}.pdf')
							plt.close()

		def plotmaforecast(self):
			if self.ma_forecast_results.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function MovingAvgFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function MovingAvgFit() and try again.' + color.END))
			else:
				mafcplot_filename = input('Do you have a local folder you wish to save the RMSE results to? If so put the file path here.')
				if group == '':
					plt.figure(figsize = (14,7))
					plt.plot(self.ma_forecast_results.index, self.ma_forecast_results['Y'])
					plt.plot(self.ma_forecast_results.index, self.ma_forecast_results['yhat'])
					mafcplot = plt.title('\n' + 'Moving Average Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if mafcplot_filename != '':
						mafcplots = mafcplot.get_figure()
						mafcplots.savefig(f'{mafcplot_filename}/Moving_average_forecast_plot.pdf')
						plt.close()
				else:
					plot_data = self.ma_forecast_results.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						plt.figure(figsize = (14,7))
						plt.plot(plot_dataset.index, plot_dataset['Y'], label = 'Observed')
						plt.plot(plot_dataset.index, plot_dataset['yhat'], label = 'Forecast')
						mafcplot = plt.title('\n' + f'Moving Average Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if mafcplot_filename != '':
							mafcplots = mafcplot.get_figure()
							mafcplots.savefig(f'{mafcplot_filename}/Moving_average_forecast_plt_for_{g}.pdf')
							plt.close()

		def rmsMovingAvg(self, marms_filename = ''):
			if self.training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function MovingAvgFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function MovingAvgFit() and try again.' + color.END))
			else:
				marms_filename = input('Do you have a local folder you wish to save the RMSE results to? If so put the file path here.')
				if group == '':
					rms_dataset = self.training[(pd.to_datetime(self.training.index) >= testdate)]
					rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if marms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						movingavg_print = open(f'{marms_filename}/Root_mean_squared_error_MovingAvg.txt','w')
						print(f'The Root Mean Squared Error is {rms}', file = movingavg_print)
						movingavg_print.close()
				else:
					rms_data = self.training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index) >= testdate)]
						rms = round(math.sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if marms_filename == '':
							print(f'The Root Mean Squared Error for {g} is {rms}')
						else:
							movingavg_print = open(f'{marms_filename}/Root_mean_squared_error_MovingAvg.txt','w')
							print(f'The Root Mean Squared Error for {g} is {rms}', file = movingavg_print)
							movingavg_print.close()
		if mabox.upper() == 'Y':
			MovingAvgFit(self)
			if splitdf.upper() == 'Y':
				maparttwo = input('Do you want to plot the results of your Moving Average validation model?')
				if maparttwo.upper() == 'Y':
					plotMovingAvg(self)
					mapartthree = input('Do you want to print out the RMSE values for this model? If so type yes.')
				else:
					mapartthree = input('Do you want to print out the RMSE values for this model? If so type yes.')
				if mapartthree.upper() == 'Y':
					rmsMovingAvg(self)
					mapartfour = input('Do you wish to plot the forecast for your Moving Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
				else:
					mapartfour = input('Do you wish to plot the forecast for your Moving Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
			else:
				mapartfour = input('Do you wish to plot the forecast for your Moving Average Forecasting Model? Type y for Yes and hit enter else hit enter.')
		if mapartfour.upper() == 'Y':
			plotmaforecast(self)

class ses:        
	def __init__(self):
		global y
		global timevariable
		global group
		global testdate
		global resamplefreq
		global startdata
		global aggregate
		global splitdf
		self.data = startdata
		self.results_ses = pd.DataFrame()
		self.ses_test_final = pd.DataFrame()
		self.ses_train_final = pd.DataFrame()
		self.ses_training = pd.DataFrame()
		self.ses_summary = pd.DataFrame()
		self.ses_yhat = pd.DataFrame()
		self.ses_forecast_final = pd.DataFrame()
		self.ses_forecast_yhat = pd.DataFrame()
		self.ses_forecast_results = pd.DataFrame()
		sesbox = input('If you wish to continue with the Simple Exponential Smoothing model type y for yes and press enter otherwise press enter.')
		def sesfit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					#Validation Model
					test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					if aggregate == 'Mean':
						self.ses_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.ses_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					if aggregate == 'Mean':
						self.ses_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.ses_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
					self.ses_yhat = self.ses_test_final.copy()
					ses_fit = SimpleExpSmoothing(np.asarray(self.ses_train_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
					self.ses_yhat['yhat'] = ses_fit.forecast(len(self.ses_test_final))
					sesrms = round(sqrt(mean_squared_error(self.ses_test_final.Y, self.ses_yhat.yhat)),4)
					self.results_ses = self.results_ses.append(self.ses_yhat)
					ses_datasets = [self.ses_train_final, self.ses_yhat]
					ses_final_datasets = pd.concat(ses_datasets)
					self.ses_training = self.ses_training.append(ses_final_datasets)
				#Forecasting Model
				forecast_data = self.data
				forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
				forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
				if aggregate == 'Mean':
					self.ses_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
				else:
					self.ses_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
				ses_forecast_fit = SimpleExpSmoothing((self.ses_forecast_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
				self.ses_forecast_yhat['yhat'] = (ses_forecast_fit.forecast(12))
				self.ses_forecast_results = self.ses_forecast_results.append(self.ses_forecast_yhat)
				ses_forecast_datasets = [self.ses_forecast_final, self.ses_forecast_yhat]
				ses_final_forecast_dataset = pd.concat(ses_forecast_datasets)
				self.ses_summary = self.ses_summary.append(ses_final_forecast_dataset)
                
			else:
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					if splitdf.upper() == 'Y':
						test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						test_data.timevariable = pd.to_datetime(test_data[timevariable])
						test_data_reindex = test_data.set_index(test_data[timevariable])
						if aggregate == 'Mean':
							self.ses_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.ses_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
						self.ses_test_final['Group'] = g
						train_data.timevariable = pd.to_datetime(train_data[timevariable])
						train_data_reindex = train_data.set_index(train_data[timevariable])
						if aggregate == 'Mean':
							self.ses_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.ses_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
						self.ses_train_final['Group'] = g
						self.ses_yhat = self.ses_test_final.copy()
						ses_fit = SimpleExpSmoothing(np.asarray(self.ses_train_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
						self.ses_yhat['yhat'] = ses_fit.forecast(len(self.ses_test_final))
						sesrms = round(sqrt(mean_squared_error(self.ses_test_final.Y, self.ses_yhat.yhat)),4)
						self.results_ses = self.results_ses.append(self.ses_yhat)
						ses_datasets = [self.ses_train_final, self.ses_yhat]
						ses_final_datasets = pd.concat(ses_datasets)
						self.ses_training = self.ses_training.append(ses_final_datasets) 
					#forecast
					forecast_data = data_groups
					forecast_data.timevariable = pd.to_datetime(forecast_data[timevariable])
					forecast_data_reindex = forecast_data.set_index(forecast_data[timevariable])
					if aggregate == 'Mean':
						self.ses_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.ses_forecast_final['Y'] = forecast_data_reindex[y].resample(resamplefreq).sum()
					self.ses_forecast_final['Group'] = g
					ses_forecast_fit = SimpleExpSmoothing((self.ses_forecast_final['Y'])).fit(smoothing_level = 0.3, optimized = False)
					self.ses_forecast_yhat['yhat'] = (ses_forecast_fit.forecast(12))
					self.ses_forecast_yhat['Group'] = g
					self.ses_forecast_results = self.ses_forecast_results.append(self.ses_forecast_yhat)
					ses_forecast_datasets = [self.ses_forecast_final, self.ses_forecast_yhat]
					ses_final_forecast_dataset = pd.concat(ses_forecast_datasets)
					self.ses_summary = self.ses_summary.append(ses_final_forecast_dataset)
			return self.ses_training, self.ses_summary

		def plotses(self):
			if self.ses_training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function sesfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function sesfit() and try again.' + color.END))
			else:
				sesplot_filename = input('If you wish to save the plots to a file on your local drive put the file path here. Otherwise press enter.')
				if group == '':
					train_group = self.ses_training[(pd.to_datetime(self.ses_training.index)< testdate)]
					data_group = self.ses_training[(pd.to_datetime(self.ses_training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					sesplot = plt.title('\n' + 'Simple Exponential Smoothing Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if sesplot_filename != '':
						sesplots = sesplot.get_figure()
						sesplots.savefig(f'{sesplot_filename}/Linear_Exponential_Smoothing_Plot.pdf')
						plt.close()
				else:
					plot_data = self.ses_training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						sesplot = plt.title('\n' + f'Simple Exponential Smoothing Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if sesplot_filename != '':
							sesplots = sesplot.get_figure()
							sesplots.savefig(f'{sesplot_filename}/Linear_Exponential_Smoothing_Plot_for_{g}.pdf')
							plt.close()

		def rmsses(self):
			if self.ses_training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function sesfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function sesfit() and try again.' + color.END))
			else:
				sesrms_filename = input('If you wish to save the RMSE results to a local file please put the file path here, otherwise press enter.')
				if sesrms_filename != '':
					ses_print = open(f'{sesrms_filename}/Root_Mean_Squared_Error_LinearExpSmoothing.txt','w')
				if group == '':
					rms_dataset = self.ses_training[(pd.to_datetime(self.ses_training.index) >= testdate)]
					rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if sesrms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						ses_print = open(f'{sesrms_filename}/Root_mean_squared_error_MovingAvg.txt','w')
						print(f'The Root Mean Squared Eroor is {rms}', file = ses_print)
						ses_print.close
				else:
					rms_data = self.ses_training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index)>= testdate)]
						rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if sesrms_filename == '':
							print(f' The Root Mean Squared Error for {g} is {rms}')
						else:
							ses_print = open(f'{sesrms_filename}/Root_mean_squared_error_MovingAvg.txt','w')
							print(f'The Root Mean Squared Error for {g} is {rms}',file = ses_print)
							ses_print.close()

		def sesforecastplot(self):
			if self.ses_summary.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function sesfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function sesfit() and try again.' + color.END))
			else:
				sesf_filename = input('If you wish to save the forecast results to a local file type the filepath here. Otherwise hit enter.')
				if group == '':
					observed_group = self.ses_summary[(pd.to_datetime(self.ses_summary.index) <= self.data[timevariable].max())]
					forecast_group = self.ses_summary[(pd.to_datetime(self.ses_summary.index) > self.data[timevariable].max())]
					plt.figure(figsize = (14,7))
					plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
					plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
					sesfplot = plt.title('\n' + 'Simple Exponential Smoothing Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if sesf_filename != '':
						sesforecastplot = sesfplot.get_figure()
						sesforecastplot.savefig(f'{sesf_filename}/Linear_ExponentialSmoothing_Forecast.pdf')
						plt.close()
				else:
					plot_data = self.ses_summary.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						observed_group = plot_dataset[(pd.to_datetime(plot_dataset.index) <= self.data[timevariable].max())]
						forecast_group = plot_dataset[(pd.to_datetime(plot_dataset.index) > self.data[timevariable].max())]
						plt.figure(figsize = (14,7))
						plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
						plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
						sesfplot = plt.title('\n' + 'Simple Exponential Smoothing Forecast', fontsize = 20)
						plt.legend(loc = 'best')
						if sesf_filename != '':
							sesforecastplot = sesfplot.get_figure()
							sesforecastplot.savefig(f'{sesf_filename}/Linear_ExponentialSmoothing_Forecast for {g}.pdf')
							plt.close()
		if sesbox.upper() == 'Y':
			sesfit(self)
			if splitdf.upper() == 'Y':
				sesparttwo = input('Do you wish to plot the results of your Simple Exponential Smoothing model? If so type yes and hit enter, else hit enter.')
				if sesparttwo.upper() == 'Y':
					plotses(self)
					sespartthree = input('Do you wish to print out the RMSE values for your model? If so type yes and hit enter, else hit enter.')
				else:
					sespartthree = input('Do you wish to print out the RMSE values for your model? If so type yes and hit enter, else hit enter.')
				if sespartthree.upper() == 'Y':
					rmsses(self)
					sespartfour = input('Do you wish to plot the forecast for your Simple Exponential Smoothing Model? If so type yes and hit enter else hit enter.')
				else:
					sespartfour = input('Do you wish to plot the forecast for your Simple Exponential Smoothing Model? If so type yes and hit enter else hit enter.')
			else:
				sespartfour = input('Do you wish to plot the forecast for your Simple Exponential Smoothing Model? If so type yes and hit enter else hit enter.')
		if sespartfour.upper() == 'Y':
			sesforecastplot(self)

			
class holt:        
	def __init__(self):
		global y
		global timevariable
		global testdate
		global group
		global resamplefreq
		global startdata
		global aggregate
		global splitdf
		self.data = startdata
		self.results_holt = pd.DataFrame()
		self.holt_test_final = pd.DataFrame()
		self.holt_train_final = pd.DataFrame()
		self.holt_training = pd.DataFrame()
		self.holt_summary = pd.DataFrame()
		self.holt_yhat = pd.DataFrame()
		self.holt_forecast_final = pd.DataFrame()
		self.holt_forecast_yhat = pd.DataFrame()
		self.holt_forecast_results = pd.DataFrame()
		hbox = input('If you wish to continue with the Holt Linear Exponential smoothing model type y for yes otherwise hit enter to end function.')
		def HoltFit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					#Validtion model
					test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					test_data.timevariable = pd.to_datetime(test_data[timevariable])
					test_data_reindex = test_data.set_index(test_data[timevariable])
					if aggregate == 'Mean':
						self.holt_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.holt_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
					train_data.timevariable = pd.to_datetime(train_data[timevariable])
					train_data_reindex = train_data.set_index(train_data[timevariable])
					if aggregate == 'Mean':
						self.holt_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.holt_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
					self.holt_yhat = self.holt_test_final.copy()
					holt_fit = Holt(np.asarray(self.holt_train_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
					self.holt_yhat['yhat'] = holt_fit.forecast(len(self.holt_test_final))
					rms = round(sqrt(mean_squared_error(self.holt_test_final.Y, self.holt_yhat.yhat)),4)
					self.results_holt = self.results_holt.append(self.holt_yhat)
					datasets = [self.holt_train_final, self.holt_yhat]
					final_datasets = pd.concat(datasets)
					self.holt_training = self.holt_training.append(final_datasets)
				#Forecast Model
				forecast_data_holt = self.data
				forecast_data_holt.timevariable = pd.to_datetime(forecast_data_holt[timevariable])
				forecast_data_holt_reindex = forecast_data_holt.set_index(forecast_data_holt[timevariable])
				if aggregate == 'Mean':
					self.holt_forecast_final['Y'] = forecast_data_holt_reindex[y].resample(resamplefreq).mean()
				else:
					self.holt_forecast_final['Y'] = forecast_data_holt_reindex[y].resample(resamplefreq).sum()
				forecast_holt_fit = Holt((self.holt_forecast_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
				self.holt_forecast_yhat['yhat'] = forecast_holt_fit.forecast(12)
				self.holt_forecast_results = self.holt_forecast_results.append(self.holt_forecast_yhat)
				holt_forecast_datasets = [self.holt_forecast_final, self.holt_forecast_yhat]
				final_holt_fc_dataset = pd.concat(holt_forecast_datasets)
				self.holt_summary = self.holt_summary.append(final_holt_fc_dataset)
			else:
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					if splitdf.upper() == 'Y':
						#Validation Model
						test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						test_data.timevariable = pd.to_datetime(test_data[timevariable])
						test_data_reindex = test_data.set_index(test_data[timevariable])
						if aggregate == 'Mean':
							self.holt_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.holt_test_final['Y'] = test_data_reindex[y].resample(resamplefreq).sum()
						self.holt_test_final['Group'] = g
						train_data.timevariable = pd.to_datetime(train_data[timevariable])
						train_data_reindex = train_data.set_index(train_data[timevariable])
						if aggregate == 'Mean':
							self.holt_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.holt_train_final['Y'] = train_data_reindex[y].resample(resamplefreq).sum()
						self.holt_train_final['Group'] = g
						self.holt_yhat = self.holt_test_final.copy()
						holt_fit = Holt(np.asarray(self.holt_train_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
						self.holt_yhat['yhat'] = holt_fit.forecast(len(self.holt_test_final))
						rms = round(sqrt(mean_squared_error(self.holt_test_final.Y, self.holt_yhat.yhat)),4)
						self.results_holt = self.results_holt.append(self.holt_yhat)
						datasets = [self.holt_train_final, self.holt_yhat]
						final_datasets = pd.concat(datasets)
						self.holt_training = self.holt_training.append(final_datasets)
					#Forecast
					forecast_data_holt = data_groups
					forecast_data_holt.timevariable = pd.to_datetime(forecast_data_holt[timevariable])
					forecast_data_holt_reindex = forecast_data_holt.set_index(forecast_data_holt[timevariable])
					if aggregate == 'Mean':
						self.holt_forecast_final['Y'] = forecast_data_holt_reindex[y].resample(resamplefreq).mean()
					else:
						self.holt_forecast_final['Y'] = forecast_data_holt_reindex[y].resample(resamplefreq).sum()
					self.holt_forecast_final['Group'] = g
					forecast_holt_fit = Holt((self.holt_forecast_final['Y'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
					self.holt_forecast_yhat['yhat'] = forecast_holt_fit.forecast(12)
					self.holt_forecast_yhat['Group'] = g
					self.holt_forecast_results = self.holt_forecast_results.append(self.holt_forecast_yhat)
					holt_forecast_datasets = [self.holt_forecast_final, self.holt_forecast_yhat]
					final_holt_fc_dataset = pd.concat(holt_forecast_datasets)
					self.holt_summary = self.holt_summary.append(final_holt_fc_dataset)
			return self.holt_training, self.holt_summary

		def plotHolt(self):
			if self.holt_training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function HoltFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function HoltFit() and try again.' + color.END))
			else:
				hplot_filename = input('Do you wish to save your plots to a local file? If so put the file path here otherwise hit enter.')
				if group == '':
					train_group = self.holt_training[(pd.to_datetime(self.holt_training.index)< testdate)]
					data_group = self.holt_training[(pd.to_datetime(self.holt_training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					holtplot = plt.title('\n' + 'Holt Linear Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if hplot_filename != '':
						hplot = holtplot.get_figure()
						hplot(f'{hplot_filename}/Holt_linear_trend_plot.pdf')
						plt.close()
				else:
					plot_data = self.holt_training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						holtplot = plt.title('\n' + f'Holt Linear Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if hplot_filename != '':
							hplot = holtplot.get_figure()
							hplot(f'{hplot_filename}/Holt_linear_trend_forecast_plot_for_{g}.pdf')
							plt.close()

		def rmsHolt(self):
			if self.holt_training.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function HoltFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function HoltFit() and try again.' + color.END))
			else:	
				hrms_filename = input('Do you wish to save the RMSE results to a local file? if so put the file path here and press enter otherwise press enter.')
				if group == '':
					rms_dataset = self.holt_training[(pd.to_datetime(self.holt_training.index) >= testdate)]
					rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if hrms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						holt_print = open(f'{hrms_filename}/Root_mean_squared_error_HoltLinear.txt','w')
						print('The Root Mean Squared Error is {rms}',file = holt_print)
						holt_print.close()
				else:
					rms_data = self.holt_training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index)>= testdate)]
						rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if hrms_filename == '':
							print(f'The Root Mean Squared Error for {g} is {rms}')
						else:
							holt_print = open(f'{hrms_filename}/Root_mean_squared_error_HoltLinear.txt','w')
							print(f'The Root Mean Squared Error for {g} is {rms}', file = holt_print)
							holt_print.close()

		def holtforecastplot(self):
			if self.holt_summary.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function HoltFit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function HoltFit() and try again.' + color.END))
			else:
				hw_filename = input('Do you wish to save your forecast plots to a local file? If so put the file path here and press enter, otherwise press enter.')
				if group == '':
					observed_group = self.holt_summary[(pd.to_datetime(self.holt_summary.index) <= self.data[timevariable].max())]
					forecast_group = self.holt_summary[(pd.to_datetime(self.holt_summary.index) > self.data[timevariable].max())]
					plt.figure(figsize = (14,7))
					plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
					plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
					holtfplot = plt.title('\n' + 'Holt Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if hw_filename != '':
						hfplot = holtfplot.get_figure()
						hfplot.savefig(f'{hw_filename}/Holt_linear_trend_forecast_plot.pdf')
						plt.close()
				else:
					plot_data = self.holt_summary.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						observed_group = plot_dataset[(pd.to_datetime(plot_dataset.index) <= self.data[timevariable].max())]
						forecast_group = plot_dataset[(pd.to_datetime(plot_dataset.index) > self.data[timevariable].max())]
						plt.figure(figsize = (14,7))
						plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
						plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
						holtfplot = plt.title('\n' + f'Holt Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if hw_filename != '':
							hfplot = holtfplot.get_figure()
							hfplot.savefig(f'{hw_filename}/Holt_linear_trend_forecast_plot_for_{g}.pdf')
							plt.close()
		if hbox.upper() == 'Y':
			HoltFit(self)
			if splitdf.upper() == 'Y':
				holtparttwo = input('Do you wish to plot the predicted and observed values from your Holt Model? If so type yes and hit enter, else hit enter.')
				if holtparttwo.upper() == 'Y':
					plotHolt(self)
					holtpartthree = input('Do you wish to print out the RMSE results from your Holt Time Series Model? If so type yes and hit enter, else hit enter.')
				else:
					holtpartthree = input('Do you wish to print out the RMSE results from your Holt Time Series Model? If so type yes and hit enter, else hit enter.')
				if holtpartthree.upper() == 'Y':
					rmsHolt(self)
					holtpartfour = input('Do you wish to plot the forecasted values for your Holt Time Series Model? If so type yes and hit enter, else hit enter.')
				else:
					holtpartfour = input('Do you wish to plot the forecasted values for your Holt Time Series Model? If so type yes and hit enter, else hit enter.')
			else:
				holtpartfour = input('Do you wish to plot the forecasted values for your Holt Time Series Model? If so type yes and hit enter, else hit enter.')
		if holtpartfour.upper() == 'Y':
			holtforecastplot(self)

class HoltWinter:        
	def __init__(self):
		global y
		global timevariable
		global testdate
		global group
		global resamplefreq
		global startdata
		global aggregate
		global splitdf
		global forecaststeps
		self.data = startdata
		self.results_holtwinter = pd.DataFrame()
		self.hw_test_final = pd.DataFrame()
		self.hw_train_final = pd.DataFrame()
		self.hw_training = pd.DataFrame()
		self.hw_summary = pd.DataFrame()
		self.holtwinter_yhat = pd.DataFrame()
		self.hw_forecast_final = pd.DataFrame()
		self.hw_forecast_yhat = pd.DataFrame()
		self.hw_forecast_results = pd.DataFrame()
		hwbox = input('If you wish to continue with the Exponential Smoothing model type y for yes and hit enter otherwise press enter to exit the function.')
		def holtwinterfit(self):
			if group == '':
				if splitdf.upper() == 'Y':
					hw_test_data = self.data[(pd.to_datetime(self.data[timevariable]) >= testdate)]
					hw_train_data = self.data[(pd.to_datetime(self.data[timevariable])< testdate)]
					hw_test_data.timevariable = pd.to_datetime(hw_test_data[timevariable])
					hw_test_data_reindex = hw_test_data.set_index(hw_test_data[timevariable])
					if aggregate == 'Mean':
						self.hw_test_final['Y'] = hw_test_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.hw_test_final['Y'] = hw_test_data_reindex[y].resample(resamplefreq).sum()
					hw_train_data.timevariable = pd.to_datetime(hw_train_data[timevariable])
					hw_train_data_reindex = hw_train_data.set_index(hw_train_data[timevariable])
					if aggregate == 'Mean':
						self.hw_train_final['Y'] = hw_train_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.hw_train_final['Y'] = hw_train_data_reindex[y].resample(resamplefreq).sum()
					self.holtwinter_yhat = self.hw_test_final.copy()
					holtwinter_fit = ExponentialSmoothing(np.asarray(self.hw_train_final['Y']), seasonal_periods = forecaststeps, trend = 'add', seasonal = 'add').fit()
					self.holtwinter_yhat['yhat'] = holtwinter_fit.forecast(len(self.hw_test_final))
					rms = round(sqrt(mean_squared_error(self.hw_test_final.Y, self.holtwinter_yhat.yhat)),4)
					self.results_holtwinter = self.results_holtwinter.append(self.holtwinter_yhat)
					hw_datasets = [self.hw_train_final, self.holtwinter_yhat]
					hw_final_datasets = pd.concat(hw_datasets)
					self.hw_training = self.hw_training.append(hw_final_datasets)
				#Forecast
				hw_forecast_data = self.data
				hw_forecast_data.timevariable = pd.to_datetime(hw_forecast_data[timevariable])
				hw_forecast_data_reindex = hw_forecast_data.set_index(hw_forecast_data[timevariable])
				if aggregate == 'Mean':
					self.hw_forecast_final['Y'] = hw_forecast_data_reindex[y].resample(resamplefreq).mean()
				else:
					self.hw_forecast_final['Y'] = hw_forecast_data_reindex[y].resample(resamplefreq).sum()
				hw_forecast_fit = ExponentialSmoothing((self.hw_forecast_final['Y']), seasonal_periods = forecaststeps, trend = 'add', seasonal = 'add').fit()
				self.hw_forecast_yhat['yhat'] = hw_forecast_fit.forecast(forecaststeps)
				self.hw_forecast_results = self.hw_forecast_results.append(self.hw_forecast_yhat)
				hw_forecast_datasets = [self.hw_forecast_final, self.hw_forecast_yhat]
				hw_final_forecast_dataset = pd.concat(hw_forecast_datasets)
				self.hw_summary = self.hw_summary.append(hw_final_forecast_dataset)
			else:
				data_group = self.data.groupby(group)
				for g in data_group.groups:
					data_groups = data_group.get_group(g)
					if splitdf.upper() == 'Y':
						hw_test_data = data_groups[(pd.to_datetime(data_groups[timevariable]) >= testdate)]
						hw_train_data = data_groups[(pd.to_datetime(data_groups[timevariable])< testdate)]
						hw_test_data.timevariable = pd.to_datetime(hw_test_data[timevariable])
						hw_test_data_reindex = hw_test_data.set_index(hw_test_data[timevariable])
						if aggregate == 'Mean':
							self.hw_test_final['Y'] = hw_test_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.hw_test_final['Y'] = hw_test_data_reindex[y].resample(resamplefreq).sum()
						self.hw_test_final['Group'] = g
						hw_train_data.timevariable = pd.to_datetime(hw_train_data[timevariable])
						hw_train_data_reindex = hw_train_data.set_index(hw_train_data[timevariable])
						if aggregate == 'Mean':
							self.hw_train_final['Y'] = hw_train_data_reindex[y].resample(resamplefreq).mean()
						else:
							self.hw_train_final['Y'] = hw_train_data_reindex[y].resample(resamplefreq).sum()
						self.hw_train_final['Group'] = g
						self.holtwinter_yhat = self.hw_test_final.copy()
						holtwinter_fit = ExponentialSmoothing(np.asarray(self.hw_train_final['Y']), seasonal_periods = forecaststeps, trend = 'add', seasonal = 'add').fit()
						self.holtwinter_yhat['yhat'] = holtwinter_fit.forecast(len(self.hw_test_final))
						hwrms = round(sqrt(mean_squared_error(self.hw_test_final.Y, self.holtwinter_yhat.yhat)),4)
						self.results_holtwinter = self.results_holtwinter.append(self.holtwinter_yhat)
						hw_datasets = [self.hw_train_final, self.holtwinter_yhat]
						hw_final_datasets = pd.concat(hw_datasets)
						self.hw_training = self.hw_training.append(hw_final_datasets)
					#Forecasting Data
					hw_forecast_data = data_groups
					hw_forecast_data.timevariable = pd.to_datetime(hw_forecast_data[timevariable])
					hw_forecast_data_reindex = hw_forecast_data.set_index(hw_forecast_data[timevariable])
					if aggregate == 'Mean':
						self.hw_forecast_final['Y'] = hw_forecast_data_reindex[y].resample(resamplefreq).mean()
					else:
						self.hw_forecast_final['Y'] = hw_forecast_data_reindex[y].resample(resamplefreq).sum()
					self.hw_forecast_final['Group'] = g       
					hw_forecast_fit = ExponentialSmoothing((self.hw_forecast_final['Y']), seasonal_periods = forecaststeps, trend = 'add', seasonal = 'add').fit()
					self.hw_forecast_yhat['yhat'] = hw_forecast_fit.forecast(forecaststeps)
					self.hw_forecast_yhat['Group'] = g
					self.hw_forecast_results = self.hw_forecast_results.append(self.hw_forecast_yhat)
					hw_forecast_datasets = [self.hw_forecast_final, self.hw_forecast_yhat]
					hw_final_forecast_dataset = pd.concat(hw_forecast_datasets)
					self.hw_summary = self.hw_summary.append(hw_final_forecast_dataset)
			return self.hw_training, self.hw_summary

		def plotHoltWinter(self):
			if self.hw_summary.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function holtwinterfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function holtwinterfit() and try again.' + color.END))
			else:	
				hwp_filename = input('Do you wish to save your predicted vs observed plots for the Exp Smoothing model to a local folder? If so put the filepath here, otherwise hit enter.')
				if group == '':
					train_group = self.hw_training[(pd.to_datetime(self.hw_training.index)< testdate)]
					data_group = self.hw_training[(pd.to_datetime(self.hw_training.index) >= testdate)]
					plt.figure(figsize = (14,7))
					plt.plot(train_group.index, train_group['Y'], label = 'Train')
					plt.plot(data_group.index, data_group['Y'], label = 'Test')
					plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
					holtwinterplot = plt.title('\n' + 'Simple Average Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if hwp_filename != '':
						hwplots = holtwinterplot.get_figure()
						hwplots.savefig(f'{hwp_filename}/Exponential_smoothing_plot.pdf')
						plt.close
				else:
					plot_data = self.hw_training.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						train_group = plot_dataset[(pd.to_datetime(plot_dataset.index)< testdate)]
						data_group = plot_dataset[(pd.to_datetime(plot_dataset.index) >= testdate)]
						plt.figure(figsize = (14,7))
						plt.plot(train_group.index, train_group['Y'], label = 'Train')
						plt.plot(data_group.index, data_group['Y'], label = 'Test')
						plt.plot(data_group.index, data_group['yhat'], label = 'Forecast')
						holtwinterplot = plt.title('\n' + f'Simple Average Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')
						if hwp_filename != '':
							hwplots = holtwinterplot.get_figure()
							hwplots.savefig(f'{hwp_filename}/Exponential_smoothing_plot_for_{g}.pdf')
							plt.close()

		def rmsHoltWinter(self):
			if self.hw_summary.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function holtwinterfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function holtwinterfit() and try again.' + color.END))
			else:
				hwrms_filename = input('Do you wish to save the RMSE results to a file? If so put the file path here, otherwise hit enter.')
				if group == '':
					rms_dataset = self.hw_training[(pd.to_datetime(self.hw_training.index) >= testdate)]
					rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
					if hwrms_filename == '':
						print(f'The Root Mean Squared Error is {rms}')
					else:
						hw_print = open(f'{hwrms_filename}/Root_mean_squared_error_ExpSmoothing.txt','w')
						print(f'The Root Mean Squared Eroor is {rms}', file = hw_print)
						hw_print.close()
				else:
					rms_data = self.hw_training.groupby('Group')
					for g in rms_data.groups:
						rms_dataset = rms_data.get_group(g)
						rms_dataset = rms_dataset[(pd.to_datetime(rms_dataset.index)>= testdate)]
						rms = round(sqrt(mean_squared_error(rms_dataset.Y, rms_dataset.yhat)),4)
						if hwrms_filename == '':
							print(f' The Root Mean Squared Error for {g} is {rms}')
						else:
							hw_print = open(f'{hwrms_filename}/Root_mean_squared_error_ExpSmoothing.txt','w')
							print(f'The Root Mean Squared Error for {g} is {rms}', file = hw_print)
							hw_print.close()

		def hwforecastplot(self):
			if self.hw_summary.empty == True:
				sys.exit(wrapper.fill(color.BOLD + 'You have not yet run the function holtwinterfit which fits the Moving Average Time series model and outputs predictions in a data frame used by the function you have chosen.' + '\n' + '\n' + color.BrightRed + 'Please run the function holtwinterfit() and try again.' + color.END))
			else:
				hwf_filename = input('Do you wish to save the forecast plots to a local folder? If so put the file path here, otherwise hit enter.')
				if group == '':
					observed_group = self.hw_summary[(pd.to_datetime(self.hw_summary.index) <= self.data[timevariable].max())]
					forecast_group = self.hw_summary[(pd.to_datetime(self.hw_summary.index) > self.data[timevariable].max())]
					plt.figure(figsize = (14,7))
					plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
					plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
					hwfplot = plt.title('\n' + 'Exponential Smoothing Forecast', fontsize = 20)
					plt.legend(loc = 'best')
					if hwf_filename != '':
						hwfplots = hwfplot.get_figure()
						hwfplots.savefig(f'{hwf_filename}/ExpSmoothing_forecast_plot.pdf')
						plt.close()
				else:
					plot_data = self.hw_summary.groupby('Group')
					for g in plot_data.groups:
						plot_dataset = plot_data.get_group(g)
						observed_group = plot_dataset[(pd.to_datetime(plot_dataset.index) <= self.data[timevariable].max())]
						forecast_group = plot_dataset[(pd.to_datetime(plot_dataset.index) > self.data[timevariable].max())]
						plt.figure(figsize = (14,7))
						plt.plot(observed_group.index, observed_group['Y'], label = 'Observed')
						plt.plot(forecast_group.index, forecast_group['yhat'], label = 'Forecast')
						hwfplot = plt.title('\n' + f'Exponential Smoothing Forecast for {g}', fontsize = 20)
						plt.legend(loc = 'best')   
						if hwf_filename != '':
							hwfplots = hwfplot.get_figure()
							hwfplots.savefig(f'{hwf_filename}/ExpSmoothing_forecast_plot_for_{g}.pdf')
							plt.close()
		if hwbox.upper() == 'Y':
			holtwinterfit(self)
			if splitdf.upper() == 'Y':
				hwparttwo = input('Do you want to plot the predicted vs observed plots for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
				if hwparttwo.upper() == 'Y':
					plotHoltWinter(self)
					hwpartthree = input('Do you want to print out the RMSE value for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
				else:
					hwpartthree = input('Do you want to print out the RMSE value for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
				if hwpartthree.upper() == 'Y':
					rmsHoltWinter(self)
					hwpartfour = input('Do you want to plot the forecasted values for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
				else:
					hwpartfour = input('Do you want to plot the forecasted values for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
			else:
				hwpartfour = input('Do you want to plot the forecasted values for your Exp Smoothing Model? If so type yes and press enter otherwise press enter.')
		if hwpartfour.upper() == 'Y':
			hwforecastplot(self)
