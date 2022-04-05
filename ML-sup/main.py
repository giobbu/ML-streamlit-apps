# use pandas-datareader to import the stock info
from xml.dom import minicompat
import pandas_datareader as web
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import pandas as pd
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 11.7,8.27
from rls import RLS
import time
import streamlit as st
import altair as alt
from sklearn import datasets


st.header('Recursive Least Squares')


TIME_STEP = st.empty()
WEIGHTS = st.empty()



df_regr = pd.DataFrame({ 'obs_x':[], 'obs_y': []})
dot_regr = alt.Chart(df_regr).mark_circle(size = 100).encode(x = 'obs_x', y = 'obs_y').properties(width=1000, height=200)
CHART_REG = st.empty() #st.altair_chart(dot_regr, use_container_width=True)


df_error = pd.DataFrame({'step':[],'error':[]})
chart_error = alt.Chart(df_error).transform_fold(['error']).mark_line().encode(x ='step:N', y='error:Q').properties(width=1000, height=200)
CHART_ERROR = st.altair_chart(chart_error, use_container_width=True)

test_size = 500

# Test function
x, y, coef = datasets.make_regression(n_samples=test_size ,#number of samples
                                      n_features=1,#number of features
                                      n_informative=1,#number of useful features 
                                      noise = 10,#bias and standard deviation of the guassian noise
                                      coef=True,#true coefficient used to generated the data
                                      random_state=0) #set for same data points for each run

print(coef)
lam = 0.98

LS = RLS(1, lam, 1)

# Not using the RLS.fit function because I want to remember all the predicted values
step_x = []
obs_x = []
obs_y = []
pred_y = []

x_ = np.matrix(np.zeros((1,1)))

for i in range(test_size):

    TIME_STEP.subheader(f'Step : {i}')
    
    x_ = np.matrix(x[i])
    pred_ = float(x_*LS.w)


    step_x.append(i)
    obs_x.append(x[i][0])
    obs_y.append(y[i])
    pred_y.append(pred_)

    
    df_regr = pd.DataFrame({ '_x': x[i], '_y': [y[i]], 'p_y': [pred_y[i]]})
    dot_regr = alt.Chart(df_regr).transform_fold(['_y', 'p_y']).mark_circle(size = 200).encode(x = '_x:Q', y = 'value:Q', color='key:N').properties(width=1000, height=500)
    

    mini =  min(obs_x)
    maxi =  max(obs_x)

    y_min = float(mini*LS.w)
    y_max = float(maxi*LS.w)

    df_pred = pd.DataFrame({'_x': [mini, x[i][0], maxi], 'p_y': [ y_min, pred_, y_max]})
    chart = alt.Chart(df_pred).mark_point().encode(x='_x', y='p_y')
    line_pred = chart.transform_regression('_x', 'p_y').mark_line()
    
    params = chart.transform_regression('_x', 'p_y', params=True ).transform_calculate(
    intercept='datum.coef[0]',
    slope='datum.coef[1]',).mark_text(align='left').encode( x=alt.value(20),  y=alt.value(20),  # pixels from top
    text='slope:N'
)

    if i >= 1:
        df = pd.DataFrame({ 'obs_x': obs_x[:-1], 'obs_y': obs_y[:-1]})
    else:
        df = pd.DataFrame({ 'obs_x': [], 'obs_y': []})

    dot = alt.Chart(df).mark_point(size = 100).encode(x = 'obs_x:Q', y = 'obs_y:Q').properties(width=1000, height=500)


    CHART_REG.altair_chart(params + line_pred + dot_regr + dot, use_container_width=True)

    error =  float(x_*LS.w) - y[i]
    df_error = pd.DataFrame({'step':[i], 'error': [error]})
    CHART_ERROR.add_rows(df_error)


    WEIGHTS.header(LS.w)

    # update
    LS.add_obs(x_.T, y[i])
    time.sleep(5)


    

print(LS.w)
# plot the predicted values against the non-noisy output
ax = plt.scatter(obs_x, y)
ax = plt.scatter(obs_x, pred_y)
ax = plt.plot(step_x, pred_y - y)
plt.show()