#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Load necessary packages
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px        # high-level plotly module
import plotly.graph_objects as go  # lower-level plotly module with more functions
import datetime as dt              # for time and date
import json                        # For loading county FIPS data

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Modeled after 
# https://covid19-bayesian.fz-juelich.de/
# https://github.com/FZJ-JSC/jupyter-jsc-dashboards/blob/master/covid19/covid19dynstat-dash.ipynb


# # 1. Read Data

# In[21]:


excess_deaths = pd.read_csv('s3://mncovid19data/excess_deaths.csv',index_col=False)
minnesota_data = pd.read_csv('s3://mncovid19data/minnesota_data.csv',index_col=False)
minnesota_data_today = pd.read_csv('s3://mncovid19data/minnesota_data_today.csv',index_col=False)
state_df = pd.read_csv('s3://mncovid19data/state_df.csv',index_col=False)

#excess_deaths = pd.read_csv('excess_deaths.csv',index_col=False)
#minnesota_data = pd.read_csv('minnesota_data.csv',index_col=False)
#minnesota_data_today = pd.read_csv('minnesota_data_today.csv',index_col=False)
#state_df = pd.read_csv('state_df.csv',index_col=False)

# Load json file
with open('geojson-counties-fips.json') as response:  # Loads local file
    counties = json.load(response)    


# In[22]:


today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 

# Set dates to datetime
excess_deaths['Date'] = pd.to_datetime(excess_deaths['Date'], format='%Y-%m-%d')
minnesota_data['Date'] = pd.to_datetime(minnesota_data['Date'], format='%Y-%m-%d')
state_df['date'] = pd.to_datetime(state_df['date'], format='%Y-%m-%d')

# Create list of months
temp = state_df['date'].dt.strftime('%B %Y') 
months = temp.unique().tolist()


# # 2. Build Web Application
# 

# ## Define Application Structure
# 
# Set-up main html and call-back structure for the application.

# In[23]:


# Initialize Dash
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Name Heroku will look for
app.title = 'Covid-19 U.S. Dashboard'


# ## (Row 1, Col 1) Minnesota Maps (Snapshots):

# ### Map of Positivity Rates

# In[24]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================

# Load geojson county location information, organized by FIPS code
#with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#    counties = json.load(response)
    
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

# In[25]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================
    
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

# In[26]:


#===========================================
# County Infections (Line Graphs by County)
#===========================================

@app.callback(
    Output('county_infect_trend', 'figure'),
    [Input('county-dropdown2', 'value')])
    
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
    [Input('county-dropdown1', 'value')])
    
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


# ## (Row 2, Col 1) U.S. Excess Deaths

# In[27]:



@app.callback(
    Output('excess_deaths', 'figure'),
    [Input('state-dropdown_alt', 'value')])

# Update Figure
def update_figure(state_values):

    dff = excess_deaths.loc[excess_deaths['state'].isin([state_values])]
    dff.set_index('Date',inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = dff.index,
            y = dff['average_expected_count'],
            name = 'Expected',
            mode='lines',
            marker_color='orange',
            opacity=0.9,
            #hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
        )
    )

    fig.add_trace(
        go.Bar(
            x = dff.index,
            y = dff['observed_number'],
            name = 'Observed',
            opacity=0.3,
            marker_color='blue',
            #hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
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
            title="Total Weekly Deaths",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: National Center for Health Statistics.")
                    ]
    )

    fig.add_shape(
                type="line",
                yref= 'paper', y0= 0, y1= .95,
                xref= 'x', x0= dt.datetime(2020, 3, 1), x1= dt.datetime(2020, 3, 1),
                line=dict(
                    color="Black",
                    width=2,
                    dash="dash",
                ),
                layer = 'above'
    )

    fig.add_annotation(
                x=dt.datetime(2019, 10, 1),
                y=.9,
                xref="x",
                yref="paper",
                text="<b>Covid-19 Declared</b><br><b> a Pandemic</b><br><b> (March 2020)</b>",
                showarrow=True,
                arrowhead=5,
                ax=0,
                ay=0)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## (Row 2, Col 2) Excess Deaths in Different States

# In[28]:


@app.callback(
    Output('excess_deaths_states', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):
        
    if state_values is None:
        dff = excess_deaths.copy()
                
        dff.drop(dff[dff['state'] == "United States"].index, inplace=True)
        dff = dff.pivot(index='Date',columns='state',values='excess_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = excess_deaths.loc[excess_deaths['state'].isin(state_values)]
        
        dff = temp.pivot(index='Date',columns='state',values='excess_deaths')        
    
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
            title="Weekly Excess Deaths = Observed - Expected Deaths",
            zeroline=True,   # Show the zero line. Otherwise say "False"
            zerolinecolor='Black', # Make the zero line "Black"      
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: National Center for Health Statistics.")
                    ]
    )

    fig.add_shape(
                type="line",
                yref= 'paper', y0= 0, y1= .95,
                xref= 'x', x0= dt.datetime(2020, 3, 1), x1= dt.datetime(2020, 3, 1),
                line=dict(
                    color="Black",
                    width=2,
                    dash="dash",
                ),
                layer = 'above'
    )

    fig.add_annotation(
                x=dt.datetime(2019, 10, 1),
                y=.9,
                xref="x",
                yref="paper",
                text="<b>Covid-19 Declared</b><br><b> a Pandemic</b><br><b> (March 2020)</b>",
                showarrow=True,
                arrowhead=5,
                ax=0,
                ay=0)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ##  (Row 2, Col 1) Line Graph:  Positive Cases over Time by State (7-day Rolling Average)

# In[29]:


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

# In[30]:


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

# In[31]:


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

# In[32]:


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


# In[33]:


modal_calc = html.Div(
    [
        dbc.Button("Calculating Expected Deaths", id="open_calc"),
        dbc.Modal(
            [
                dbc.ModalHeader("Intuitive Primer on Calculating Excess Deaths"),
                dbc.ModalBody(
                dcc.Markdown(
                            f"""
                            "Excess Deaths" defined as "Observed Deaths" minus "Expected Deaths" where 
                            the latter requires some explanation. The calculation of "Expected Deaths" 
                            is based on historical counts of deaths (from 2013 to present) using the 
                            approach of Farrington et al (1996) which amounts to estimating a negative binomial 
                            (because deaths are counts, not continuous) non-linear regression equation. 
                            The equation uses cosines and sines to account for trends and seasonality 
                            (you can see these patterns in the left chart). Yes, this sounds fancy and the math 
                            looks daunting but you should think of this as approximately
                             "expected number of deaths is equal to counting the number of 
                            deaths this week of the year in past data and taking an average."
                            """
                            )
                            ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_calc", className="ml-auto")
                ),
            ],
            id="modal_calc",
        ),
    ],
    style={"margin-left": "15px"}
)


@app.callback(
    Output("modal_calc", "is_open"),
    [Input("open_calc", "n_clicks"), Input("close_calc", "n_clicks")],
    [State("modal_calc", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

modal_data = html.Div(
    [
        dbc.Button("About Weekly Death Data", id="open_data"),
        dbc.Modal(
            [
                dbc.ModalHeader("About the Data"),
                dbc.ModalBody(
                dcc.Markdown(
                            f"""
                            Death counts from the National Vital Statistics System database since this is the timeliest mortality data.
                            Number of deaths reported correspond to total number of deaths received and coded as of the date of analysis
                            but may not represent all deaths that occurred in that period, especially for recent data as
                            the time lag between when the death occurred and when the death certificate is completed, submitted to NCHS and 
                            processed for reporting purposes can be large -- between 1 and 8 weeks depending upon jurisdiction. 
                            Data for New York state exclude New York City. 
                            """
                            )
                            ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_data", className="ml-auto")
                ),
            ],
            id="modal_data",
        ),
    ],
    style={"margin-left": "15px"}
)

@app.callback(
    Output("modal_data", "is_open"),
    [Input("open_data", "n_clicks"), Input("close_data", "n_clicks")],
    [State("modal_data", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# ## Call-backs and Control Utilities

# In[34]:


# County Dropdown
county_dropdown1 =  html.P([
            dcc.Dropdown(
            id='county-dropdown1',
            options=[{'label': i, 'value': i} for i in minnesota_data['Admin2'].dropna().unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})

county_dropdown2 =  html.P([
            dcc.Dropdown(
            id='county-dropdown2',
            options=[{'label': i, 'value': i} for i in minnesota_data['Admin2'].dropna().unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})

# Dropdown
state_dropdown = html.P([
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
    
# Dropdown
state_dropdown_alt = html.P([
            dcc.Dropdown(
            id='state-dropdown_alt',
            options=[{'label': i, 'value': i} for i in excess_deaths['state'].dropna().unique().tolist()],
            multi=False,
            value='USA',
            searchable= True)
            ], style = {'height': '20px',
                        'width' : '25%',
                        'fontSize' : '15px',
                        'display': 'inline-block'})

# range slider
slider = html.P([
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

# In[35]:


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


# In[36]:


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

In the second tab, I present estimates of the percent of each county's residents who have been infected.
This calculation begins with cumulative positive cases by location and time. I then adjust for un-reported
cases assuming that for every 1 positive case, there exist 7.1 unreported cases (i.e., I multiply case counts by 8.1).
Finally, I divide by total county population using 2019 estimates. Results are sensitive to the above report:unreport statistic. The 1:7.1 statistic I use 
is on the low-end of estimates and is sourced from a recent publication by CDC researchers using
data from February through September 2020 [article](https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciaa1780/6000389).
                #
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
        
tab1_content = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row([    
        dbc.Col([html.Br(),html.Br(),dcc.Graph(id="school_map", figure = fig_school_map)],width=6), 
        dbc.Col([county_dropdown1,dcc.Graph(id="county_trend")],width=6)
            ])
        ],
    ),
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row([
        dbc.Col([html.Br(),html.Br(),dcc.Graph(id="infected_map", figure = fig_infect_map)],width=6), 
        dbc.Col([county_dropdown2,dcc.Graph(id="county_infect_trend")],width=6)
            ])
        ],
    ),
)
    
# App Layout
app.layout = dbc.Container(fluid=True, children=[
    ## Top
    navbar, 
    html.Br(),html.Br(),html.Br(),html.Br(),
    desc, mn_head, mn_desc, 
    html.Br(),html.Br(),
    
    ## 
    dbc.Row(dbc.Col(width=12, children = [
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="14-Day Case Rates"),
                dbc.Tab(tab2_content, label="Level of Infection"),
            ]
        )           
        ])
    ),    
    html.Br(),
    
    dbc.Row([
        dbc.Col(width=12, children=[
        state_head, state_desc, state_dropdown, slider,
        html.Br(),html.Br()
        ]),

        ### left plots
        dbc.Col(width=6, children=[   
            dbc.Row(
            children=[html.H4("Observed vs Expected Deaths in:  "),state_dropdown_alt]),
            dbc.Col(dcc.Graph(id="excess_deaths")),
            dbc.Row([modal_data, modal_calc]),
            html.Br(),html.Br(),
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
            dbc.Col(html.H4("Excess Deaths by States")), 
            dbc.Col(dcc.Graph(id="excess_deaths_states")),
            html.Br(),html.Br(),html.Br(),html.Br(),
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

# In[37]:


if __name__ == '__main__':
    #app.run_server(debug=True, use_reloader=False)  # Jupyter
    app.run_server(debug=True)    # Use this line prior to heroku deployment
    #application.run(debug=False, port=8080) # Use this line for AWS

