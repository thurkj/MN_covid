#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary packages
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
import dash_html_components as html
from dash.dependencies import Input, Output


# In[2]:


# Initialize Dash
app = dash.Dash()
app.title = 'Covid-19 Trends'
server = app.server


# In[3]:


url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=30)  # Only need 28 days but plugging in a little wiggle room in the event data are not available
delta = dt.timedelta(days=1)

dataset = pd.DataFrame() # Initialize datframe
while start_date <= end_date:
    
    request = requests.get(url+start_date.strftime('%m-%d-%Y')+'.csv')
    if request.status_code == 200:
        df = pd.read_csv(url+start_date.strftime('%m-%d-%Y')+'.csv')
        df = df.dropna(subset=['Admin2'])
        df = df[df['Province_State']=="Minnesota"]
        df['Date'] = start_date
        dataset = dataset.append(df, ignore_index=True)
        as_of = start_date
        
    start_date += delta


# In[4]:


# Merge in county-level population.
pop = pd.read_excel('county_population.xlsx')
dataset = dataset.merge(pop,on='FIPS',how='outer')


# In[5]:


dataset.drop(dataset[dataset['Admin2'] == 'Unassigned'].index, inplace=True)   # drop values not assigned to a county
dataset = dataset.sort_values(by=['Admin2','Date'])   # Sort data
dataset.reset_index(inplace=True)
dataset['new_cases'] = dataset.groupby('Admin2')['Confirmed'].diff().fillna(0)
dataset['new_cases_rolling'] = dataset.groupby('Admin2')['new_cases'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
dataset['new_deaths'] = dataset.groupby('Admin2')['Deaths'].diff().fillna(0)
dataset['new_deaths_rolling'] = dataset.groupby('Admin2')['new_deaths'].rolling(7).mean().reset_index(0,drop=True)


# In[6]:


# MN Dept of Health Statistic
dataset['new_cases_MNDH'] = dataset.groupby('Admin2')['new_cases'].rolling(14).sum().fillna(0).reset_index(0,drop=True)
dataset['ratio'] = 1e+4*dataset['new_cases_MNDH']/dataset['pop2019']

# Assess trend
dataset['new_cases_21days'] = dataset.groupby('Admin2')['new_cases'].rolling(21).sum().fillna(0).reset_index(0,drop=True)
dataset['new_cases_7days'] = dataset.groupby('Admin2')['new_cases'].rolling(7).sum().fillna(0).reset_index(0,drop=True)
dataset['new_cases_MNDH_previous'] = dataset['new_cases_21days'] - dataset['new_cases_7days']
dataset['ratio_previous'] = 1e+4*dataset['new_cases_MNDH_previous']/dataset['pop2019']

dataset['trend'] = 'Downward'
dataset.loc[(dataset['new_cases_MNDH']>dataset['new_cases_MNDH_previous']), 'trend'] = 'Upward'

# MN Dept of Health School Guidelines 
# In-person learning for all students 0 to less than 10
# Elementary in-person, Middle/high school hybrid 10 to less than 20
# Both hybrid 20 to less than 30
# Elementary hybrid, Middle/high school distance 30 to less than 50
# Both distance 50 or more
dataset['schooling'] = 'Elementary & MS/HS in-person (x<10)'
dataset.loc[(dataset['ratio']>=10) & (dataset['ratio']<20), 'schooling'] = 'Elementary in-person, MS/HS hybrid'
dataset.loc[(dataset['ratio']>=20) & (dataset['ratio']<30), 'schooling'] = 'Elementary & MS/HS hybrid'
dataset.loc[(dataset['ratio']>=30) & (dataset['ratio']<50), 'schooling'] = 'Elementary hybrid, MS/HS distance'
dataset.loc[(dataset['ratio']>=50) & (dataset['ratio']<100), 'schooling'] = 'Elementary & MS/HS distance'
dataset.loc[dataset['ratio']>=100, 'schooling'] = 'WTF! Are you even listening?'

dataset['text'] = 'County: ' + dataset['Admin2'] + '<br>' +                    'MN Dept of <br>Health Statistic:    '+ dataset['ratio'].astype(float).round(2).astype(str) + '<br>'+                    'Trending:             '+ dataset['trend'] + '<br>'+                    'New Cases / Day: '+ dataset['new_cases_rolling'].astype(float).round(2).astype(str) + '<br>'+                    'Deaths/ Day:         '+ dataset['new_deaths_rolling'].astype(float).round(2).astype(str)


# In[7]:


df = dataset.groupby('Admin2').tail(1) # Keep only the last observation

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
fig_map = px.choropleth(df, geojson=counties, locations='FIPS', color='schooling',
                           color_discrete_map={
                            'Elementary & MS/HS in-person':'green',
                            'Elementary in-person, MS/HS hybrid':'tan',
                            'Elementary & MS/HS hybrid':'yellow',
                            'Elementary hybrid, MS/HS distance':'orange',
                            'Elementary & MS/HS distance':'red',
                            'WTF! Are you even listening?':'black'},
                           category_orders = {
                            'schooling':['Elementary & MS/HS in-person',
                            'Elementary in-person, MS/HS hybrid',
                            'Elementary & MS/HS hybrid',
                            'Elementary hybrid, MS/HS distance',
                            'Elementary & MS/HS distance',
                            'WTF! Are you even listening?'
                            ]},
                           projection = "mercator",
                           labels={'schooling':'Recommended Education Format:'},
                           hover_name = df['text'],
                           hover_data={'FIPS':False,'schooling':False},
                          )

fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(legend=dict(
                        yanchor="top",
                        y=0.40,
                        xanchor="left",
                        x=0.60,
                        font_size=10
                      ),
                      margin={"r":0,"t":0,"l":0,"b":0},
                      dragmode=False
                      )


# In[8]:


#===========================================
# County 14-day Case Rate
#===========================================

@app.callback(
    Output('county_trend', 'figure'),
    [Input('county-dropdown', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = dataset.pivot(index='Date',columns='Admin2',values='ratio')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = dataset.loc[dataset['Admin2'].isin(county_values)]
            
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
        margin=dict(l=10, r=0, t=0, b=0),
        title={
                'text': "14-day COVID-19 Case Count per 10,000 Residents",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
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
            title="14-day Case Count per 10k Residents",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.1,
                            showarrow=False,
                            text ='Source: Minnesota Department of Health')
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


# In[9]:


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


# In[10]:


df = covid[['date','state','positiveIncrease','hospitalizedIncrease','deathIncrease','death']]
del covid # Drop for memory issues

# Trime dataframe to states we're interested in
#df.drop(df[(df['state'] != "MN") & (df['state'] != "WI") & (df['state'] != "SD") & (df['state'] != "ND") & (df['state'] != "IA")].index, inplace = True)  # Keep only MN

# Sort
df = df.sort_values(by=['state','date'])
df.reset_index(inplace=True)

df['new_cases'] = df.groupby('state')['positiveIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
df['new_hospitalized'] = df.groupby('state')['hospitalizedIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
df['new_deaths'] = df.groupby('state')['deathIncrease'].rolling(7).mean().fillna(0).reset_index(0,drop=True)
df = df.rename(columns={"death":"total_deaths"})

# Add population information
df = df.merge(population,on='state',how='left')
df.dropna(subset=['state'],inplace=True)

df2 = df.copy()
df2.state = 'All States'
df = df.append(df2, ignore_index=True)


# In[11]:


#---------------------------------------------------------------------------
# DASH App formating
#---------------------------------------------------------------------------
header = html.H1(children="COVID-19 TRENDS (as of " + today + ")")

markdown = dcc.Markdown(
f"""
-----

#### The following graphs depict Covid-19 trends. The graphs are interactive; e.g., hover your cursor over a data-series to observe specific values.

-----
"""    
)

subheader1 = html.H1(children="1. Minnesota School Guidance")
markdown_text = 'The following figure presents county-level COVID-19 case rates organized by MN Dept of Health School guidelines.' +' Source: Minnesota Dept of Health, retrieved ' + today + '.' +' The left panel presents current variation in COVID-19 Case Rates.' +' Hover your cursor over a county to observe relevant characteristics.' +' The right panel presents the evolution of COVID-19 Case Rates by County.' +' Select which counties to analyze using the pull-down menu or by entering in the county name.',
markdown1 = dcc.Markdown(children=markdown_text)

markdown_text = 'The following graphs compare COVID-19 statistics. Mid-western states are selected by default but you can modify the analysis by choosing a different subset of states, periods, and/or standardize the statistics by population.'
markdown2 = dcc.Markdown(children=markdown_text)

subheader2 = html.H1(children="2. State COVID-19 Trends")

# County Dropdown
dropdown0 =  html.P([
            html.Label("Select One or More Counties"),
            dcc.Dropdown(
            id='county-dropdown',
            options=[{'label': i, 'value': i} for i in dataset['Admin2'].unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})

# Dropdown
dropdown1 =  html.P([
            html.Label("Select One or More States"),
            dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': i, 'value': i} for i in df['state'].unique().tolist()],
            multi=True,
            value=['MN','WI','IA','ND','SD'],
            searchable= True)
            ], style = {'width' : '40%',
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
            ], style = {'width' : '80%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'})
    
dropdown2 =  html.P([
            html.Label("Present Raw Data or Population-adjusted (per 10,000 residents)"),
            dcc.Dropdown(
            id='normalization-dropdown',
             options=[
            {'label': 'Raw Data', 'value': 'Yes'},
            {'label': 'Per 10,000 Residents', 'value': 'No'},
            ],
            value='Yes',
            multi=False,
            searchable= True)
            ], style = {'width' : '40%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'})

state_map = dcc.Graph(id="map", figure = fig_map, style={'margin-left': "0px", 'width': '48%', 'display': 'inline-block'})
county_trend = dcc.Graph(id="county_trend", style={'display': 'inline-block'})
graph1 = dcc.Graph(id="positive", style={'display': 'inline-block'})
graph2 = dcc.Graph(id="curhospital", style={'display': 'inline-block'})
graph3 = dcc.Graph(id="newdeaths", style={'display': 'inline-block'})
graph4 = dcc.Graph(id="totdeaths", style={'display': 'inline-block'})

dropdown = html.Div(children=[dropdown1, dropdown2])
county_trend_fig = html.Div(html.Div([dropdown0, county_trend]), style={'width': '48%', 'display': 'inline-block'})
row0 = html.Div(children=[state_map, county_trend_fig])
row1 = html.Div(children=[graph1, graph2])
row2 = html.Div(children=[graph3, graph4])

layout = html.Div(children=[header, markdown, subheader1, markdown1, row0, subheader2, markdown2, dropdown, slider, row1, row2], style={"text-align": "center","width":"95%"})
app.layout = layout


# In[12]:


#===========================================
# Daily Positive Cases
#===========================================

@app.callback(
    Output('positive', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):
        
    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_cases'] = 1e+4*dff['new_cases']/dff['POP'] 
        
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values[0]=="No":
            dff['new_cases'] = 1e+4*dff['new_cases']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values[0]=="No":
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
        title={
                'text': "New Daily Cases (7-day Moving Average)",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,   
        xaxis=dict(
            title="Date",
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
                            x=0.5, y=1.1,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[13]:


#===========================================
# Currently Hospitalized
#===========================================

@app.callback(
    Output('curhospital', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_hospitalized'] = 1e+4*dff['new_hospitalized']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_hospitalized'] = 1e+4*dff['new_hospitalized']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
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
        title={
                'text': "New Daily Hospitalizations (7-day Moving Average)",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
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
                            x=0.5, y=1.1,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ],
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[14]:


#===========================================
# Daily Deaths
#===========================================

@app.callback(
    Output('newdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_deaths'] = 1e+4*dff['new_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_deaths'] = 1e+4*dff['new_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
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
        title={
                'text': "Daily Deaths (7-day Moving Average)",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
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
                            x=0.5, y=1.1,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[15]:


#===========================================
# Total Number of Deaths
#===========================================

@app.callback(
    Output('totdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['total_deaths'] = 1e+4*dff['total_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['total_deaths'] = 1e+4*dff['total_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
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
        title={
                'text': "Total Deaths (Cumulative)",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
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
                            x=0.5, y=1.1,
                            showarrow=False,
                            text ='Source: The Atlantic Covid-19 Tracking Project')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[ ]:


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)  # Jupyter
    #app.run_server(debug=True)    # Comment above line and uncomment this line prior to heroku deployment

