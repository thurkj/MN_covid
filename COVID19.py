#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary packages
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px        # high-level plotly module
import plotly.graph_objects as go  # lower-level plotly module with more functions
import pandas_datareader as pdr    # we are grabbing the data and wb functions from the package
import datetime as dt              # for time and date
import requests                    # api module
from urllib.request import urlopen
import json

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

# Modeled after 
# https://covid19-bayesian.fz-juelich.de/


# # 1. Load Data via API
# 
# ## (a) MN County-Level Data from Johns Hopkins 

# In[2]:


url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 
day = dt.datetime.now().strftime('%w')  # today's day of week where sunday = 0.

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=60) # Collect 44 days of data => 30 days with a 14-day window
delta = dt.timedelta(days=1)

minnesota_data = pd.DataFrame() # Initialize datframe
while start_date <= end_date:
    
    request = requests.get(url+start_date.strftime('%m-%d-%Y')+'.csv')
    if request.status_code == 200:
        df = pd.read_csv(url+start_date.strftime('%m-%d-%Y')+'.csv')
        df = df.dropna(subset=['Admin2'])
        df = df[df['Province_State']=="Minnesota"]
        df['Date'] = start_date
        minnesota_data = minnesota_data.append(df, ignore_index=True)
        as_of = start_date
        del df  # erase df from memory
    start_date += delta
    
# Convert "Date" field to datetime
minnesota_data['Date'] = pd.to_datetime(minnesota_data['Date'], format='%Y-%m-%d')


# In[3]:


# Merge in county-level population.
pop = pd.read_excel('county_population.xlsx')
minnesota_data = minnesota_data.merge(pop,on='FIPS',how='outer')


# In[4]:


minnesota_data.drop(minnesota_data[minnesota_data['Admin2'] == 'Unassigned'].index, inplace=True)   # drop values not assigned to a county
minnesota_data.dropna(subset=['Date'], inplace = True)  # Some dates are screwed up so drop them.

minnesota_data = minnesota_data.sort_values(by=['Admin2','Date'])   # Sort data
minnesota_data.reset_index(inplace=True)
minnesota_data['month'] = minnesota_data['Date'].dt.strftime('%B %Y') 
minnesota_data['new_cases'] = minnesota_data.groupby('Admin2')['Confirmed'].diff().fillna(0)
minnesota_data['new_cases_rolling'] = minnesota_data.groupby('Admin2')['new_cases'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
minnesota_data['new_deaths'] = minnesota_data.groupby('Admin2')['Deaths'].diff().fillna(0)
minnesota_data['new_deaths_rolling'] = minnesota_data.groupby('Admin2')['new_deaths'].rolling(7).mean().reset_index(0,drop=True)


# In[5]:


# Percent Infected
minnesota_data['perc_infected'] = 100*minnesota_data['Confirmed']/minnesota_data['pop2019']
minnesota_data['perc_infected'] = minnesota_data['perc_infected'].fillna(0)

minnesota_data['infect'] = 'Less than 5%'
minnesota_data.loc[(minnesota_data['perc_infected']>=5) & (minnesota_data['perc_infected']<10), 'infect'] = 'Between 5% and 10%'
minnesota_data.loc[(minnesota_data['perc_infected']>=10) & (minnesota_data['perc_infected']<15), 'infect'] = 'Between 10% and 15%'
minnesota_data.loc[(minnesota_data['perc_infected']>=15) & (minnesota_data['perc_infected']<20), 'infect'] = 'Between 15% and 20%'
minnesota_data.loc[minnesota_data['perc_infected']>=20, 'schooling'] = 'Greater than 20%'


# In[6]:


# Days to herd immunity
minnesota_data['change'] = minnesota_data.groupby('Admin2')['perc_infected'].diff().fillna(0)
minnesota_data['change'] = minnesota_data.groupby('Admin2')['change'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
minnesota_data['herd_days'] = (80-minnesota_data['perc_infected'])/minnesota_data['change'] # Assumes 80% exposure needed for herd immunity

# Replace missing values
minnesota_data['herd_days'] = minnesota_data['herd_days'].fillna(0)
minnesota_data['herd_days'].replace(np.inf,0,inplace=True)
minnesota_data['max_herd_days'] = minnesota_data.groupby('Admin2')['herd_days'].transform('max')
minnesota_data.loc[minnesota_data['herd_days']==0,'herd_days'] = minnesota_data['max_herd_days']


# Construct the "14-day Case Count" statistic proposed by the Minnesota Department of Health. Also add the school recommendations.

# In[7]:


# MN Dept of Health Statistic
minnesota_data['new_cases_MNDH'] = minnesota_data.groupby('Admin2')['new_cases'].rolling(14).sum().fillna(0).reset_index(0,drop=True)
minnesota_data['ratio'] = 1e+4*minnesota_data['new_cases_MNDH']/minnesota_data['pop2019']

# Assess trend
minnesota_data['new_cases_21days'] = minnesota_data.groupby('Admin2')['new_cases'].rolling(21).sum().fillna(0).reset_index(0,drop=True)
minnesota_data['new_cases_7days'] = minnesota_data.groupby('Admin2')['new_cases'].rolling(7).sum().fillna(0).reset_index(0,drop=True)
minnesota_data['new_cases_MNDH_previous'] = minnesota_data['new_cases_21days'] - minnesota_data['new_cases_7days']
minnesota_data['ratio_previous'] = 1e+4*minnesota_data['new_cases_MNDH_previous']/minnesota_data['pop2019']

minnesota_data['trend'] = 'Downward'
minnesota_data.loc[(minnesota_data['new_cases_MNDH']>minnesota_data['new_cases_MNDH_previous']), 'trend'] = 'Upward'

# MN Dept of Health School Guidelines 
# In-person learning for all students 0 to less than 10
# Elem. in-person, Middle/high school hybrid 10 to less than 20
# Both hybrid 20 to less than 30
# Elem. hybrid, Middle/high school distance 30 to less than 50
# Both distance 50 or more
minnesota_data['schooling'] = 'Elem. & MS/HS in-person'
minnesota_data.loc[(minnesota_data['ratio']>=10) & (minnesota_data['ratio']<20), 'schooling'] = 'Elem. in-person, MS/HS hybrid'
minnesota_data.loc[(minnesota_data['ratio']>=20) & (minnesota_data['ratio']<30), 'schooling'] = 'Elem. & MS/HS hybrid'
minnesota_data.loc[(minnesota_data['ratio']>=30) & (minnesota_data['ratio']<50), 'schooling'] = 'Elem. hybrid, MS/HS distance'
minnesota_data.loc[(minnesota_data['ratio']>=50) & (minnesota_data['ratio']<100), 'schooling'] = 'Elem. & MS/HS distance'
minnesota_data.loc[minnesota_data['ratio']>=100, 'schooling'] = 'Armageddon?'

minnesota_data['text'] = 'County: ' + minnesota_data['Admin2'] + '<br>' +                    'MN Dept of <br>Health Statistic:    '+ minnesota_data['ratio'].astype(float).round(2).astype(str) + '<br>'+                    'Trending:             '+ minnesota_data['trend'] + '<br>'+                    'Percent Infected:      '+ minnesota_data['perc_infected'].astype(float).round(2).astype(str) + '<br>'+                    'New Cases / Total Cases: '+ minnesota_data['new_cases'].fillna(0).astype(int).astype(str) + ' / '+ minnesota_data['Confirmed'].fillna(0).astype(int).astype(str) + '<br>' +                    'New Deaths/ Total Deaths:         '+ minnesota_data['new_deaths'].fillna(0).astype(int).astype(str) + ' / ' + minnesota_data['Deaths'].fillna(0).astype(int).astype(str)

#                    'New Cases / Day: '+ minnesota_data['new_cases_rolling'].astype(float).round(2).astype(str) + '<br>'+\
#                    'Deaths/ Day:         '+ minnesota_data['new_deaths_rolling'].astype(float).round(2).astype(str)

# Adds all available categories to each time frame
dts = minnesota_data['Date'].unique()
catg = minnesota_data['schooling'].unique()
perc = minnesota_data['infect'].unique()
for tf in dts:
    for i in catg:
        minnesota_data = minnesota_data.append({
            'Date' : tf,
            'Admin2': "NA",
            'text' : 'N',
            'FIPS' : '0',
            'schooling' : i
        }, ignore_index=True)
        
minnesota_data_today = minnesota_data.groupby('Admin2').tail(1) # Keep only the last observation

# Adds all available categories for the legend
for i in catg:
    minnesota_data_today = minnesota_data_today.append({
        'Admin2': "NA",
        'text' : 'N',
        'FIPS' : '0',
        'schooling' : i
    }, ignore_index=True)    
    
for i in perc:
    minnesota_data_today = minnesota_data_today.append({
        'Admin2': "NA",
        'text' : 'N',
        'FIPS' : '0',
        'infect' : i
    }, ignore_index=True)    


# ## (b) State-Level COVID-19 Data
# 
# Load State-level COVID-19 Data via The Covid-19 Tracking Project API

# In[8]:


url = 'https://api.census.gov/data/2019/pep/population?get=NAME,POP&for=state:*'
    
response = requests.get(url)
population = pd.read_json(response.text)  # convert to dataframe
population.head()

population.rename(columns=population.iloc[0],inplace=True)
population.drop(0,inplace=True)
population.drop(['state'], axis=1,inplace=True)

abbrev = pd.read_csv('state_abbrev.csv',header=None)  # convert to dataframe
abbrev.rename(columns={0:'NAME',1:'state'},inplace=True)

population = population.merge(abbrev,on='NAME',how='outer')
population.drop(['NAME'], axis=1,inplace=True)
population['POP'] = population.POP.astype(float)

url = 'https://api.covidtracking.com/v1/states/daily.json'
    
response = requests.get(url)

if response.status_code == 200: 
    print('Download successful')
    covid = pd.read_json(response.text)  # convert to dataframe
elif response.status_code == 301: 
    print('The server redirected to a different endpoint.')
elif response.status_code == 401: 
    print('Bad request.')
elif response.status_code == 401: 
    print('Authentication required.')
elif response.status_code == 403: 
    print('Access denied.')
elif response.status_code == 404: 
    print('Resource not found')
else: 
    print('Server busy')

# Clean-up data
covid['date'] = pd.to_datetime(covid['date'], format='%Y%m%d')
covid['month'] = covid['date'].dt.strftime('%B %Y') 
covid.drop(covid[covid['date'] < '2020-03-01'].index, inplace=True)   # Drop January and February when there are few cases

months = covid['month'].unique().tolist()
months.reverse()


# Trim covid data to just those we're interested in.

# In[9]:


state_df = covid[['date','state','positiveIncrease','hospitalizedIncrease','deathIncrease','death']]
del covid # Drop to save application memory

# Sort
state_df = state_df.sort_values(by=['state','date'])
state_df.reset_index(inplace=True)

state_df['new_cases'] = state_df.groupby('state')['positiveIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
state_df['new_hospitalized'] = state_df.groupby('state')['hospitalizedIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
state_df['new_deaths'] = state_df.groupby('state')['deathIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
state_df = state_df.rename(columns={"death":"total_deaths"})

# Add population information
state_df = state_df.merge(population,on='state',how='left')
state_df.dropna(subset=['state'],inplace=True)


# # 2. Build Application
# 

# ## Define Application Structure
# 
# Set-up main html and call-back structure for the application.

# In[10]:


# Initialize Dash
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Covid-19 U.S. Dashboard'
server = app.server


# In[11]:


minnesota_data_today['perc_infected'].describe()


# ## (Row 1, Col 1) Minnesota Maps (Snapshots):

# ### Map of Positivity Rates

# In[23]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================

# Load geojson county location information, organized by FIPS code
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
df = minnesota_data_today.dropna(subset=['infect'])

fig_infect_map = px.choropleth(df, geojson=counties, locations='FIPS', color='infect',
                           color_discrete_map={
                            'Less than 5%':'lightyellow',
                            'Between 5% and 10%':'yellow',
                            'Between 10% and 15%':'orange',
                            'Between 15% and 20%':'red',
                            'Greater than 20%':'black'},
                           category_orders = {
                            'infect':['Less than 5%',
                            'Between 5% and 10%',
                            'Between 10% and 15%',
                            'Between 15% and 20%',
                            'Greater than 20%'
                            ]},
                           projection = "mercator",
                           labels={'infect':'Percent Infected:'},
                           hover_name = df['text'],
                           hover_data={'FIPS':False,'infect':False},
                          )

fig_infect_map.update_geos(fitbounds="locations", visible=False)
fig_infect_map.update_layout(legend=dict(
                        yanchor="top",
                        y=0.5,
                        xanchor="left",
                        x=0.6,
                        font_size=10
                      ),
                      margin={"r":0,"t":0,"l":0,"b":0},
                      dragmode=False
                      )


# ### Map of 14-day Case Rates

# In[13]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================

# Load geojson county location information, organized by FIPS code
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
df = minnesota_data_today.dropna(subset=['schooling'])

fig_school_map = px.choropleth(df, geojson=counties, locations='FIPS', color='schooling',
                           color_discrete_map={
                            'Elem. & MS/HS in-person':'green',
                            'Elem. in-person, MS/HS hybrid':'tan',
                            'Elem. & MS/HS hybrid':'yellow',
                            'Elem. hybrid, MS/HS distance':'orange',
                            'Elem. & MS/HS distance':'red',
                            'Armageddon?':'black'},
                           category_orders = {
                            'schooling':['Elem. & MS/HS in-person',
                            'Elem. in-person, MS/HS hybrid',
                            'Elem. & MS/HS hybrid',
                            'Elem. hybrid, MS/HS distance',
                            'Elem. & MS/HS distance',
                            'Armageddon?'
                            ]},
                           projection = "mercator",
                           labels={'schooling':'Recommended Format:'},
                           hover_name = df['text'],
                           hover_data={'FIPS':False,'schooling':False},
                          )

fig_school_map.update_geos(fitbounds="locations", visible=False)
fig_school_map.update_layout(legend=dict(
                        yanchor="top",
                        y=0.5,
                        xanchor="left",
                        x=0.6,
                        font_size=10
                      ),
                      margin={"r":0,"t":0,"l":0,"b":0},
                      dragmode=False
                      )


# ## (Row 1, Col 2)  County Trends

# In[14]:


#===========================================
# County 14-day Case Rate Trend (Line Graphs by County)
#===========================================

@app.callback(
    Output('county_infect_trend', 'figure'),
    [Input('county-dropdown', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = minnesota_data.pivot(index='Date',columns='Admin2',values='perc_infected')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = minnesota_data.loc[minnesota_data['Admin2'].isin(county_values)]
            
        dff = temp.pivot(index='Date',columns='Admin2',values='perc_infected')              
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics
        
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
            )
        )

    # Update remaining layout properties
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=12),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            title="Percent of Residents Which Have Tested Positive",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: Minnesota Department of Health. Author's calculations.")
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


#===========================================
# County 14-day Case Rate Trend (Line Graphs by County)
#===========================================

@app.callback(
    Output('county_trend', 'figure'),
    [Input('county-dropdown', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = minnesota_data.pivot(index='Date',columns='Admin2',values='ratio')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = minnesota_data.loc[minnesota_data['Admin2'].isin(county_values)]
            
        dff = temp.pivot(index='Date',columns='Admin2',values='ratio')              
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics
        
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
            )
        )

    # Update remaining layout properties
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=12),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            title="14-day Case Count per 10k Residents",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: Minnesota Department of Health. Author's calculations.")
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig

#===========================================
# Days to Herd Immunity (Line Graphs by County)
# Assumes 80% required exposure
#===========================================

@app.callback(
    Output('county_herd', 'figure'),
    [Input('county-dropdown', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = minnesota_data.pivot(index='Date',columns='Admin2',values='herd_days')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = minnesota_data.loc[minnesota_data['Admin2'].isin(county_values)]
            
        dff = temp.pivot(index='Date',columns='Admin2',values='herd_days')              
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics
        
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
            )
        )

    # Update remaining layout properties
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=12),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            title="Days to Herd Immunity (ie, 80% Exposure)",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: Minnesota Department of Health. Author's calculations.")
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


# ##  (Row 2, Col 1) Line Graph:  Positive Cases over Time by State (7-day Rolling Average)

# In[15]:


#===========================================
# Daily Positive Cases - Raw Data
#===========================================

@app.callback(
    Output('positive_raw', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):
        
    if state_values is None:
        dff = state_df.copy()
        
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
                    
        dff = temp.pivot(index='date',columns='state',values='new_cases')        

    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]

    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

#===========================================
# Daily Positive Cases - Per Capita
#===========================================

@app.callback(
    Output('positive_pc', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):
        
    if state_values is None:
        dff = state_df.copy()
        
        # normalization
        dff['new_cases'] = 1e+4*dff['new_cases']/dff['POP'] 
        
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
        
        # Normalization
        temp['new_cases'] = 1e+4*temp['new_cases']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_cases')        

    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]

    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## (Row 2, Col 2)  Line Graph: Hospitalizations over Time by State (7-day Rolling Average)

# In[16]:


#===========================================
# Currently Hospitalized - Raw
#===========================================

@app.callback(
    Output('curhospital_raw', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='new_hospitalized')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ],
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

#===========================================
# Currently Hospitalized - Per Capita
#===========================================

@app.callback(
    Output('curhospital_pc', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
        
        # Normalization
        dff['new_hospitalized'] = 1e+4*dff['new_hospitalized']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
        
        # Normalization
        temp['new_hospitalized'] = 1e+4*temp['new_hospitalized']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_hospitalized')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ],
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## (Row 3, Col 1)  Line Graph: Daily Deaths by State (7-day Rolling Average)

# In[17]:


#===========================================
# Daily Deaths - Raw
#===========================================

@app.callback(
    Output('newdeaths_raw', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='new_deaths')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16), 
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

#===========================================
# Daily Deaths - Per Capita
#===========================================

@app.callback(
    Output('newdeaths_pc', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
        
        # Normalization
        dff['new_deaths'] = 1e+4*dff['new_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
        
        # Normalization
        temp['new_deaths'] = 1e+4*temp['new_deaths']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_deaths')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16), 
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## (Row 3, Col 2) Line Graph: Cumulative Deaths by State

# In[18]:


#===========================================
# Total Number of Deaths - Raw
#===========================================

@app.callback(
    Output('totdeaths_raw', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='total_deaths')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),

        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

#===========================================
# Total Number of Deaths - Per Capita
#===========================================

@app.callback(
    Output('totdeaths_pc', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = state_df.copy()
        
        # Normalization
        dff['total_deaths'] = 1e+4*dff['total_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
        
        # Normalization
        temp['total_deaths'] = 1e+4*temp['total_deaths']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='total_deaths')
    
    # Filter by months
    dff = dff.loc[dt.datetime.strptime(months[month_values[0]],"%B %Y") : dt.datetime.strptime(months[month_values[1]],"%B %Y")+ MonthEnd(1)]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        margin={"r":0,"t":10,"l":0,"b":0},
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),

        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## Call-backs and Control Utilities

# In[19]:


# County Dropdown
county_dropdown =  html.P([
            dcc.Dropdown(
            id='county-dropdown',
            options=[{'label': i, 'value': i} for i in minnesota_data['Admin2'].unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})

# Dropdown
state_dropdown =  html.P([
            html.Label("Select One or More States"),
            dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': i, 'value': i} for i in state_df['state'].unique().tolist()],
            multi=True,
            value=['MN','WI','IA','ND','SD'],
            searchable= True)
            ], style = {'width' : '80%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'})
    
# range slider
slider =    html.P([
            html.Label("Select Time Period"),
            dcc.RangeSlider(id = 'slider',
                        marks = {i : months[i] for i in range(0, len(months))},
                        min = 0,
                        max = len(months)-1,
                        value = [0, len(months)-1])
            ], style = {'width' : '90%',
                        'padding-left': '20px',
                        'fontSize' : '20px',
                        'display': 'inline-block'})


# ## Define HTML

# In[20]:


#####################
# Header and Footer
#####################
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/

navbar = dbc.NavbarSimple(
    brand="COVID-19 DASHBOARD: " + today ,
    brand_href="#",
    color="dark",
    fixed="top",
    dark=True
    )

navbar_footer = dbc.NavbarSimple(
    brand="Jeff Thurk // jeffthurk.com // Department of Economics // University of Georgia",
    color="light",
    #fixed="bottom",
    #sticky=True,
    #dark=True,
    )


# In[21]:


#---------------------------------------------------------------------------
# DASH App formating
#---------------------------------------------------------------------------
header = html.H1(children="COVID-19 TRENDS (as of " + today + ")")

desc = dcc.Markdown(
f"""
#### The following graphs depict Covid-19 trends. The graphs are interactive; e.g., hover your cursor over a data-series to observe specific values.

-----
"""    
)

mn_head = html.H1(children="1. Covid-19 Prevalance Across Minnesota Counties")
mn_desc = dcc.Markdown(
            f"""
The following figures present county-level COVID-19 data. Source: Minnesota Dept of Health. 


The left panel presents current variation in COVID-19 across Minnesota counties.
Hover your cursor over a county to observe relevant characteristics. The 14-day case rate is defined as total new 
positive cases in the last 14 days per 10,000 residents. Data are organized according to school guidelines presented
by the state of Minnesota in July 2020.


The right panel presents the evolution of COVID-19 by county.
Select which counties to analyze using the pull-down menu or by entering in the county name.
            """   
)

state_head = html.H1(children="2. State COVID-19 Trends")
state_desc = dcc.Markdown(
            f"""
The following graphs compare COVID-19 statistics. 
Select midwestern states are included by default but you can modify the analysis by choosing a different subset of
states, periods, and/or standardize the statistics by population.
            """   
)

# App Layout
app.layout = dbc.Container(fluid=True, children=[
    ## Top
    navbar, 
    html.Br(),html.Br(),html.Br(),html.Br(),
    desc, mn_head, mn_desc, 
    html.Br(),html.Br(),
    ## Body: MN Counties
    dbc.Row([
        ### plots
        dbc.Col(width=6, children=[
            dbc.Col(html.H4("State Snapshot")), 
            dbc.Tabs(className="nav", children=[            
                dbc.Tab(dcc.Graph(id="school_map", figure = fig_school_map, style={'margin-left': '-100px'}), label="School Guidelines"),
                dbc.Tab(dcc.Graph(id="infected_map", figure = fig_infect_map, style={'margin-left': '-100px'}), label="Percent Infected")
                ]),
            ]),
                
        dbc.Col(width=6, children=[
            dbc.Col(html.H4("County-level Trends")), 
            dbc.Tabs(className="nav", children=[            
                dbc.Tab(dcc.Graph(id="county_trend"), label="14-day Case Rate"),
                dbc.Tab(dcc.Graph(id="county_infect_trend"), label="Percent Infected"),
                dbc.Tab(dcc.Graph(id="county_herd"), label="Herd Immunity")
                ]),
            ]),
        ]),
    dbc.Row([
        dbc.Col(width=6, children = county_dropdown)
        ], justify="end"),    
    html.Br(),
    
    dbc.Row([
        dbc.Col(width=12, children=[
        state_head, state_desc, state_dropdown, slider,
        html.Br(),html.Br()
        ]),
        
        ### left plots
        dbc.Col(width=6, children=[   
            dbc.Col(html.H4("New Cases (7-day Moving Avg.)")), 
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="positive_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="positive_pc"), label="Per 10,000")
            ]),
            html.Br(),html.Br(),
            dbc.Col(html.H4("New Deaths (7-day Moving Avg.)")),
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="newdeaths_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="newdeaths_pc"), label="Per 10,000")
            ]),
        ]),
                
        ### right plots
        dbc.Col(width=6, children=[            
            dbc.Col(html.H4("New Hospitalizations (7-day Moving Avg.)")), 
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="curhospital_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="curhospital_pc"), label="Per 10,000")
            ]),
            html.Br(),html.Br(),
            dbc.Col(html.H4("Total Deaths")),
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="totdeaths_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="totdeaths_pc"), label="Per 10,000")
            ]),
        ]),
    ], no_gutters=False),
    html.Br(),html.Br(),
    navbar_footer
])


# # 3. Run Application

# In[22]:


if __name__ == '__main__':
    #app.run_server(debug=True, use_reloader=False)  # Jupyter
    app.run_server(debug=False)    # Comment above line and uncomment this line prior to heroku deployment

