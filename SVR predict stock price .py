# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 05:23:28 2019

@author: yx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
import itertools

#Step1:Open file,get raw data 
def prices_from_csv(fname):
    df = pd.read_csv(fname)
    df.set_index(pd.to_datetime(df['Date']), inplace=True)
    df=df.sort_index(ascending=True, axis=0)
    df=df.dropna()
    return df
def get_data_y_x1(company,predict_days=1):
    daily_prices = company['Close']
    total_y=pd.DataFrame(daily_prices)
    total_y=total_y.iloc[predict_days:]
    sentiment=company.iloc[:-(predict_days),2:]
    return total_y,sentiment
def get_data_x2(df_ti,predict_days=1):
    Total_TI=df_ti.iloc[:-(predict_days),:]
    Total_TI=Total_TI.dropna()
    return Total_TI
def get_raw_xy(company,predict_days):
    df_nt=company+'30MIN_NEWS+TWITTER.csv'
    df_ti=company+'_Total_TI.csv'
    company=prices_from_csv(df_nt)
    company_ti=prices_from_csv(df_ti)
    (total_y,sentiment)=get_data_y_x1(company,predict_days)   
    Total_TI=get_data_x2(company_ti,predict_days)
    return total_y,sentiment,Total_TI


company='AMZN'
predict_days=1
(total_y,sentiment,Total_TI)=get_raw_xy(company,i)




#Step2: search for important feature

              #Method1: Correlation Matrix with Heatmap
def Correlation_Matrix(Total_TI):
    plt.figure(figsize=(20,18))
    cor = Total_TI.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.savefig('Correlation Matrix with Heatmap.png')
    return plt.show()

Correlation_Matrix(Total_TI)


              #Method2: Feature Importance with XGBoost##
def get_feature_importance(data_income,s,e):
    data = data_income.copy()
    y=data['Close']
    X = data.iloc[:, s:e]    
    train_samples = int(X.shape[0] * 0.7) 
    X_train_FI = X.iloc[:train_samples]
    X_test_FI = X.iloc[train_samples:]
    y_train_FI = y.iloc[:train_samples]
    y_test_FI = y.iloc[train_samples:]    
    regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=100,base_score=0.7,colsample_bytree=1,learning_rate=0.15)
    xgbModel = regressor.fit(X_train_FI,y_train_FI, \
                         eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
                         verbose=False)
    eval_result = regressor.evals_result()
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    
    fig1 = plt.figure(figsize=(8,8))
    plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
    plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.show()

    fig2 = plt.figure(figsize=(12,10))
    plt.xticks(rotation='vertical')
    plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
    plt.title('Feature importance of the technical indicators.')
    #plt.savefig(company+':Feature importance of the technical indicators.png')
    plt.show()
    return 

get_feature_importance(Total_TI,2,30)
Total_TI.columns.values.tolist()
final_allti=Total_TI.loc[:,['EMA_5','NDXT_Close']]

#Step3: StandardScaler_data

def generate_datasets(y,x1,x2):
    x1st=x1.copy()
    x2nd=x2.copy()
    y_data=y.copy()
    x_data=pd.concat([x1st,x2nd],axis=1)
    x_data=x_data.dropna()
    xcount=len(x_data)
    y=y.iloc[:xcount,:]
    return y_data,x_data
def StandardScaler_data(fname):
    scaler= StandardScaler()
    vname=fname.columns.values.tolist()
    for i in vname:
        if i =='NEWS_SENTIMENT_DAILY_AVG' or i=='TWITTER_SENTIMENT_DAILY_AVG':
            fname[i]=fname[[i]]
        else:
            fname[i] = scaler.fit_transform(fname[[i]])
    return fname
def get_dividen_data(x,y):    
    train_samples = int(x.shape[0] * 0.7) 
    x_train = x.iloc[:train_samples]
    y_train = y.iloc[:train_samples]    
    x_test = x.iloc[train_samples:]
    y_ture = y.iloc[train_samples:]    
    return (x_train, y_train), (x_test, y_ture)


(y_data,x_data)=generate_datasets(total_y,sentiment,final_allti)
x_data2=StandardScaler_data(x_data)
y_data2=StandardScaler_data(y_data)
(X_train, Y_train), (X_test, Y_ture) =get_dividen_data(x_data2,y_data2)

#Step6:SVR Parameter Setting
svr=SVR()
params = [
        {'kernel': ['linear'], 'C': [1,10,1e2,1e3,1e4]},
        {'kernel': ['poly'], 'C': [1,10,1e2,1e3,1e4], 'degree': [1,2,3,4]},
        {'kernel': ['rbf'], 'C': [1,10,1e2,1e3,1e4], 'gamma':[1e-5,1e-4,1e-3,1e-2,0.1,1]}
        ]
model = GridSearchCV(svr, params, refit=True,scoring='neg_mean_squared_error',cv=5)
model.fit(X_train, Y_train)

print('-----------------------------------------------------------------')
print('best_estimator_:',model.best_estimator_) 
print('best_params_:',model.best_params_) 
print('best_score_:',model.best_score_)
print('-----------------------------------------------------------------')

#Step7:SVR Model
def mape_value(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def mse_value(y_true, y_pred):    
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse
def fix_svr(X_train, Y_train,X_test, Y_ture, k='rbf', Cc=1000, g=0.0001,predict_days=1):
    svr_rbf = SVR(kernel=k, C=Cc, gamma=g)
    y_predic = svr_rbf.fit(X_train, Y_train).predict(X_test)
    Trainscore=svr_rbf.score(X_train,Y_train)
    Testscore=svr_rbf.score(X_test,Y_ture)
    print("Train score:",svr_rbf.score(X_train,Y_train))
    print("Test score:",svr_rbf.score(X_test,Y_ture))
    
    y_t=Y_ture.copy()
    YYY= pd.DataFrame(y_t)
    YYY['y_p']=y_predic
    yture=Y_ture['Close'].tolist()
    svr_mse=mse(YYY['Close'],YYY['y_p'])
    svr_rmse=np.sqrt(svr_mse)
    svr_mape=mape_value(yture,y_predic)
    #print('fixed MSE:',svr_mse)
    print('fixed RMSE:',svr_rmse)
    print('fixed MAPE:',svr_mape)
    return Trainscore,Testscore,svr_rmse,svr_mape
def floating_window_predict(y_data,x_data,k, Cc, g,predict_period):
    totalpredict= int(x_data.shape[0] * 0.2)
    firsttrain =int( x_data.shape[0]-totalpredict)
    last=int(totalpredict/predict_period)*predict_period
    Y_ture = y_data.iloc[firsttrain:]
    y_t=Y_ture.copy()
    svr_rbf = SVR(kernel=k, C=Cc, gamma=g)
    y_p={}
    name=0
    for i in range(0,last,predict_period):
        X_train = x_data.iloc[i:firsttrain+i]
        Y_train = y_data.iloc[i:firsttrain+i]    
        X_test = x_data.iloc[firsttrain+i:firsttrain+i+predict_period]
        y_predict = svr_rbf.fit(X_train, Y_train).predict(X_test)
        y_p[name]=y_predict
        name=name+1
    if last<totalpredict:
        X_train = x_data.iloc[last:firsttrain+last]
        Y_train = y_data.iloc[last:firsttrain+last] 
        X_test = x_data.iloc[firsttrain+last:]
        y_predict = svr_rbf.fit(X_train, Y_train).predict(X_test)
        y_p[name]=y_predict
    y_p=[i for i in y_p.values()] 
    y_p = list(itertools.chain.from_iterable(y_p))
    y_t['y_p']=y_p    
    return y_t
def model_evaluation(twoy):
    y_true=np.array(twoy.iloc[:,0].tolist())
    y_predict=np.array(twoy.iloc[:,1].tolist())
    floating_mse=mse_value(y_true,y_predict)
    floationg_rmse=np.sqrt(floating_mse)
    floating_mape=mape_value(y_true,y_predict)    
    #print('floating MSE:',floating_mse)
    print('floating RMSE:',floationg_rmse)
    print('floating MAPE:',floating_mape)
    return floating_mse,floationg_rmse,floating_mape


#Step7:SVR Result
companylist=['AAPL','MSFT','GOOGL','FB','AMZN','NFLX','SBUX','JD']

for n in companylist:
    print(n)
    if n=='AAPL':
        company='AAPL'
        variables=['EMA_5','SMA_60','SMA_5','UBB_2','BollMA_20','NDXT_Close']
        k='rbf'
        c=1000
        g=0.0001
    elif n=='MSFT':
        company='MSFT'
        variables=['BollMA_20','RSI_14','NDXT_SMA_120']
        k='rbf'
        c=1000
        g=0.0001
    elif n=='GOOGL':
        company='GOOGL'
        variables=['EMA_5','RSI_14','NDXT_Close']
        k='rbf'
        c=1000
        g=0.001
    elif n=='FB':
        company='FB'
        variables=['LBB_2','RSI_14','NDXT_SMA_120']
        k='rbf'
        c=100
        g=0.01    
       
    if n=='AMZN':
        company='AMZN'
        variables=['EMA_5','NDXT_Close']
        k='rbf'
        c=100
        g=0.001
    elif n=='NFLX':
        company='NFLX'
        variables=['SMA_5','BollW','NDXT_Close']
        k='rbf'
        c=1000
        g=0.0001
    elif n=='SBUX':
        company='SBUX'
        variables=['EMA_5','MACD(12,26)','BollW','NDXT_Close']
        k='rbf'
        c=1000
        g=0.0001
    elif n=='JD':
        company='JD'
        variables=['EMA_5','MACD(12,26)','NDXT_SMA_120']
        k='rbf'
        c=1000
        g=0.0001   
    predict_days=[6,7,8,9,10,11,12]
    for i in predict_days:
        (total_y,sentiment,Total_TI)=get_raw_xy(company,i)
        final_allti=Total_TI.loc[:,variables]
        (y_data,x_data)=generate_datasets(total_y,sentiment,final_allti)
    
        x_data2=StandardScaler_data(x_data)
        y_data2=StandardScaler_data(y_data)
        (XX_train, YY_train), (XX_test, YY_ture) =get_dividen_data(x_data2,y_data2)
        print('when predict for next %s days:'%i)
        (Trainscore,Testscore,svr_rmse,svr_mape)=fix_svr(XX_train, YY_train,XX_test, YY_ture, k, c, g,predict_days=i)
        trainscore.append(Trainscore)
        testscore.append(Testscore)
        fixrmse.append(svr_rmse)
        fixmape.append(svr_mape)
        y_predic_float=floating_window_predict(y_data2,x_data2,k, c, g,predict_period=1)    
        (f_mse,f_rmse,f_mape)=model_evaluation(y_predic_float)
        floatrmse.append(f_rmse)
        floatmape.append(f_mape) 
        print('')
    print('————————————————————————————————————————————')

dict_float={"r_RMSE":floatrmse,'r_MAPE':floatmape}  
dict_fixed={"Train score":trainscore,"Test Score":testscore,"RMSE":fixrmse,'MAPE':fixmape} 
indexlist=['AAPL T+1','AAPL T+3', 'AAPL T+5','AAPL T+10','AAPL T+20','AAPL T+6.5','AAPL T+7',
           'MSFT T+1','MSFT T+3', 'MSFT T+5','MSFT T+10','MSFT T+20','MSFT T+6.5','MSFT T+7',
           'GOOGL T+1','GOOGL T+3', 'GOOGL T+5','GOOGL T+10','GOOGL T+20','GOOGL T+6.5','GOOGL T+7',
           'FB T+1','FB T+3', 'FB T+5','FB T+10','FB T+20','FB T+6.5','FB T+7',
           'AMZN T+1','AMZN T+3', 'AMZN T+5','AMZN T+10','AMZN T+20','AMZN T+6.5','AMZN T+7',
           'NFLX T+1','NFLX T+3', 'NFLX T+5','NFLX T+10','NFLX T+20','NFLX T+6.5','NFLX T+7',
           'SBUX T+1','SBUX T+3', 'SBUX T+5','SBUX T+10','SBUX T+20','SBUX T+6.5','SBUX T+7',
           'JD T+1','JD T+3', 'JD T+5','JD T+10','JD T+20','JD T+6.5','JD T+7']
df_float=pd.DataFrame(dict_float,index=indexlist)
df_float.to_csv('TI+N+T float.csv')
df_fixed=pd.DataFrame(dict_fixed,index=indexlist)   
df_fixed.to_csv('TI+N+T fixed.csv') 