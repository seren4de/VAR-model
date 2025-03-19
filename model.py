import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')

nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]


# if the the series are not stationary we difference them which is my case for two itteration 
df_differenced = df_train.diff().dropna() #1
df_differenced = df_differenced.diff().dropna() #2

#checking for stationarity
def adfuller_test(series, signif=0.05, name='', verbose=False):

    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    

for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """

    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx


#for j in df_differenced.index :
#    add_freq(df_differenced.index)

model = VAR(df_differenced)

# checking which order p has the lowest AIC score
#for i in [1,2,3,4,5,6,7,8,9]:
#   result = model.fit(i)
#    print('Lag Order =', i)
#    print('AIC : ', result.aic)
#    print('BIC : ', result.bic)
#    print('FPE : ', result.fpe)
#    print('HQIC: ', result.hqic, '\n')

#training the VAR model 
model_fitted = model.fit(4)
print (model_fitted.summary())


#checking for serial correlation
from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(model_fitted.resid)

def adjust(val, length= 6): return str(val).ljust(length)

print("\nResults for serial correlation :")
for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))      

#getting the lag order 
lag_order=model_fitted.k_ar
print("\n\nthe lag order is :",lag_order)


#input data to use for the forcast
forecast_input = df_differenced.values[-lag_order:]
print("\n\n",forecast_input)

#forcasting     
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
print(df_forecast)

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
print(df_results.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
                   'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']])


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();
#plt.show()

###################################checking for the accuracy of the forcast###############################################
from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

print('Forecast Accuracy of: rgnp')
accuracy_prod = forecast_accuracy(df_results['rgnp_forecast'].values, df_test['rgnp'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: pgnp')
accuracy_prod = forecast_accuracy(df_results['pgnp_forecast'].values, df_test['pgnp'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: ulc')
accuracy_prod = forecast_accuracy(df_results['ulc_forecast'].values, df_test['ulc'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: gdfco')
accuracy_prod = forecast_accuracy(df_results['gdfco_forecast'].values, df_test['gdfco'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: gdf')
accuracy_prod = forecast_accuracy(df_results['gdf_forecast'].values, df_test['gdf'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: gdfim')
accuracy_prod = forecast_accuracy(df_results['gdfim_forecast'].values, df_test['gdfim'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: gdfcf')
accuracy_prod = forecast_accuracy(df_results['gdfcf_forecast'].values, df_test['gdfcf'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

print('\nForecast Accuracy of: gdfce')
accuracy_prod = forecast_accuracy(df_results['gdfce_forecast'].values, df_test['gdfce'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

################################################anomaly detection#####################################
def find_anomalies(squared_errors):
    threshold = np.mean(squared_errors) + np.std(squared_errors)
    predictions = (squared_errors >= threshold).astype(int)
    return predictions, threshold


squared_errors = model_fitted.resid.sum(axis=1) ** 2
predictions, threshold = find_anomalies(squared_errors)
predictions.to_csv('anomalies.csv', header=None, index=None, sep=',', mode='a') 
#print(predictions)
#print(threshold)

#################################anomaly reapair via IMR  iterrative minimum############################################

#df_repair = df_differenced
data = df_differenced
#data.iloc[lag_order:,:] = df_differenced.iloc[lag_order:, :]
data.insert(len(data.columns),'predictions',' ')


itr=lag_order
for p in data.iloc[:lag_order,8] :
    data.iloc[lag_order-itr,8] = 0
    itr=itr-1

data.iloc[lag_order:,8] = predictions
print(data)


def anomaly_repair(data,lag_order):
    itr = 0
    for k in data.loc[:,"predictions"] :
            if k ==  1 :
                print(k)
                #anomalous vector
                nobs=1
                print (data.iloc[(itr-lag_order):itr,0:8])

                forcast_input1 = data.iloc[(itr-lag_order):itr,0:8].values

                fc2 = model_fitted.forecast(y=forcast_input1, steps=nobs)
                pred = pd.DataFrame(fc2)#result of the first for forecast
                pred1 = np.array(pred)
                y1 = np.array(forcast_input1)

                element1 = np.array(data.iloc[itr-lag_order,0:8])
                element2 = np.array(data.iloc[itr-lag_order+1,0:8])
                element3 = np.array(data.iloc[itr-lag_order+2,0:8])
                element4 = np.array(data.iloc[itr-lag_order+3,0:8])

                tresh = (np.absolute(element1-element2)+np.absolute(element1-element3)+np.absolute(element1-element4)+np.absolute(element2-element3)+np.absolute(element2-element4)+np.absolute(element3-element4))/6
                print ("the tresh is : ",tresh)
                x = np.array(data.iloc[itr,0:8])
                for m in range(0,5) : # 8 is the maximum number of iterration to make if the minimum did not converge 
                    treshy = np.absolute(pred1-x)
                    x=pred1
                    sample = data.iloc[itr-lag_order:itr,0:8]
                    sample.iloc[lag_order-1,0:8]= pred
                    if treshy.all() <= tresh.all() :
                        #print(data.iloc[itr,0:8],"  ",itr)
                        data.iloc[itr,0:8] = pred
                        #print(data.iloc[itr,0:8],"  ",itr) 
                        break
                    fcx = model_fitted.forecast(sample.values, steps=1)
                    pred = pd.DataFrame(fcx)
                    pred1 = np.array(pred)
                data.iloc[itr,0:8] = pred

                #print(data.iloc[itr,0:8],"  ",itr)
            itr = itr+1
    
    dict = {'rgnp': 'rgnp_2d',
        'pgnp': 'pgnp_2d',
        'ulc': 'ulc_2d',
        'gdfco': 'gdfco_2d',
        'gdf': 'gdf_2d',
        'gdfim': 'gdfim_2d',
        'gdfcf': 'gdfcf_2d',
        'gdfce': 'gdfce_2d'}

    data.rename(columns=dict,
          inplace=True)
    return data
            
            

#print (anomaly_repair(data, lag_order))
#dg= np.array(data.iloc[51])
#print (dg)

df_rep = anomaly_repair(data, lag_order)
#print(df_rep)
df_repaired = invert_transformation(df_train, df_rep.iloc[:,0:8], second_diff=True)
print ("the result of real repaired dataset :\n ",df_repaired.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
                   'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']])


###############################the reapired dataset for analysis use ####################################################
test = df_repaired.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
                   'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']]


