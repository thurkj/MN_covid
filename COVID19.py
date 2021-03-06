#!/usr/bin/env python
# coding: utf-8

# In[197]:


# Load necessary packages
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px        # high-level plotly module
import plotly.graph_objects as go  # lower-level plotly module with more functions
from plotly.subplots import make_subplots
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

# In[198]:


excess_deaths = pd.read_csv('s3://mncovid19data/excess_deaths.csv',index_col=False)
excess_deaths_age = pd.read_csv('s3://mncovid19data/excess_deaths_age.csv',index_col=False)
excess_deaths_race = pd.read_csv('s3://mncovid19data/excess_deaths_race.csv',index_col=False)
state_df = pd.read_csv('s3://mncovid19data/state_df.csv',index_col=False)
vaccines = pd.read_csv('s3://mncovid19data/vaccines.csv',index_col=False)

# Address some missing values
vaccines['doses_admin_total'].replace(0, np.nan, inplace=True)

vaccines['utilization'] = 100*vaccines['doses_admin_total']/vaccines['total_allocation'] 
vaccines['share'] = 100*vaccines['doses_admin_total']/vaccines['POP'] 

util_avg = np.mean(vaccines['utilization'])  # Average Utilization
share_avg = np.mean(vaccines['share'])  # Average per capita vaccination


# In[199]:


today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 

# Set dates to datetime
excess_deaths['Date'] = pd.to_datetime(excess_deaths['Date'], format='%Y-%m-%d')
state_df['date'] = pd.to_datetime(state_df['date'], format='%Y-%m-%d')

# Create list of months
temp = state_df['date'].dt.strftime('%B %Y') 
months = temp.unique().tolist()


# # 2. Build Web Application
# 

# ## Define Application Structure
# 
# Set-up main html and call-back structure for the application.

# In[200]:


# Initialize Dash
app = dash.Dash(external_stylesheets=[dbc.themes.YETI])
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
#app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Covid-19 U.S. Dashboard'
server = app.server  # Name Heroku will look for


# ## (Row 2, Col 1) U.S. Excess Deaths

# In[201]:



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
                x=dt.datetime(2019, 9, 1),
                y=.85,
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

# In[202]:


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
                x=dt.datetime(2019, 9, 1),
                y=.85,
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


# ## Excess Deaths by Age

# In[203]:


@app.callback(
    Output('excess_deaths_age', 'figure'),
    [Input('state-dropdown_age', 'value')],
    [Input('state-dropdown_age_group', 'value')])

# Update Figure
def update_figure(state_values,age_group):

    dff = excess_deaths_age.loc[(excess_deaths_age['state'].isin([state_values])&excess_deaths_age['age_group'].isin([age_group]))]
    dff.set_index('week',inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = dff.index,
            y = dff['expected_deaths'],
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
            y = dff['observed_deaths'],
            name = '2020-2021',
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
            title="Week of Year",
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

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ## Excess Deaths by Race and Ethnicity

# In[204]:


@app.callback(
    Output('excess_deaths_race', 'figure'),
    [Input('state-dropdown_race', 'value')],
    [Input('state-dropdown_race_group', 'value')])

# Update Figure
def update_figure(state_values,race):

    dff = excess_deaths_race.loc[(excess_deaths_race['state'].isin([state_values])&excess_deaths_race['race_ethnicity'].isin([race]))]
    dff.set_index('week',inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = dff.index,
            y = dff['expected_deaths'],
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
            y = dff['observed_deaths'],
            name = '2020-2021',
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
            title="Week of Year",
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

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# ##  (Row 3, Col 1) Line Graph:  Positive Cases over Time by State (7-day Rolling Average)

# In[205]:


#===========================================
# Daily Positive Cases - Raw Data
#===========================================

@app.callback(
    Output('positive_raw', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):
        
    if state_values is None:
        dff = state_df.copy()
        
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
                    
        dff = temp.pivot(index='date',columns='state',values='new_cases')        

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
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):
        
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


# ## (Row 3, Col 2)  Line Graph: Hospitalizations over Time by State (7-day Rolling Average)

# In[206]:


#===========================================
# Currently Hospitalized - Raw
#===========================================

@app.callback(
    Output('curhospital_raw', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='new_hospitalized')
    
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
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

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


# ## (Row 4, Col 1)  Line Graph: Daily Deaths by State (7-day Rolling Average)

# In[207]:


#===========================================
# Daily Deaths - Raw
#===========================================

@app.callback(
    Output('newdeaths_raw', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='new_deaths')
    
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
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

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


# ## (Row 4, Col 2) Line Graph: Cumulative Deaths by State

# In[208]:


#===========================================
# Total Number of Deaths - Raw
#===========================================

@app.callback(
    Output('totdeaths_raw', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

    if state_values is None:
        dff = state_df.copy()
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = state_df.loc[state_df['state'].isin(state_values)]
            
        dff = temp.pivot(index='date',columns='state',values='total_deaths')
    
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
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

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


# In[209]:


#===========================================
# Vaccinations - Numbers
#===========================================
@app.callback(
    Output('vaccines_raw', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):

    if state_values is None:
        dff = vaccines.copy()  
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        dff = vaccines.loc[vaccines['state'].isin(state_values)]
    
    fig = px.bar(dff, x='state', 
                 y='doses_admin_total', 
                 text='doses_admin_total', 
                 labels={'doses_admin_total':'Total Vaccinated','state':'State'})
      
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

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
                            text ='Source: State Departments of Health')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

#===========================================
# Vaccinations - % of Total Population
#===========================================
@app.callback(
    Output('vaccines_pc', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):
    
    if state_values is None:
        dff = vaccines.copy()  
        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        dff = vaccines.loc[vaccines['state'].isin(state_values)]
            
    fig = px.bar(dff, x='state', y='share', 
                 text='share', 
                 labels={'share':'Percent of Total Population Vaccinated',
                         'state':'State (Red Line Indicates Average Across All States)'})

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
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
                            text ='Source: State Departments of Health')
                    ]
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
        # Add average
    fig.add_shape(
                type="line",
                xref= 'paper',
                x0= 0,
                y0= share_avg, # use absolute value or variable here
                x1= 1,
                y1= share_avg, # ditto
                line=dict(
                    color="Red",
                    width=2,
                    dash="dash",
                ),
                layer = 'above'
    )
    return fig

#===========================================
# Vaccinations - Utilization
#===========================================
@app.callback(
    Output('vaccines_util', 'figure'),
    [Input('state-dropdown', 'value')])
    
# Update Figure
def update_figure(state_values):
    
    if state_values is None:
        dff = vaccines.copy()  
        
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        dff = vaccines.loc[vaccines['state'].isin(state_values)]
     
    fig = px.bar(dff, x='state', y='utilization', 
                 text='utilization', 
                 labels={'utilization':'Percent of Vaccinates Admistered of Total Allocated',
                         'state':'State (Red Line Indicates Average Across All States)'})

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
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
                            text ='Source: CDC and State Departments of Health')
                    ]   
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
    # Add average
    fig.add_shape(
                type="line",
                xref= 'paper',
                x0= 0,
                y0= util_avg, # use absolute value or variable here
                x1= 1,
                y1= util_avg, # ditto
                line=dict(
                    color="Red",
                    width=2,
                    dash="dash",
                ),
                layer = 'above'
    )
        
    return fig


# In[210]:


modal_calc = html.Div(
    [
        dbc.Button("Calculating Expected Deaths", id="open_calc"),
        dbc.Modal(
            [
                dbc.ModalHeader("Intuitive Primer on Calculating Expected Deaths"),
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
    style={"margin-left": "15px","margin-top": "15px"}
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
    style={"margin-left": "15px","margin-top": "15px"}
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

modal_data_age = html.Div(
    [
        dbc.Button("About Weekly Death Data", id="open_data_age"),
        dbc.Modal(
            [
                dbc.ModalHeader("About the Data"),
                dbc.ModalBody(
                dcc.Markdown(
                            f"""
                            Data presented are raw death counts from all causes by week-of-year.
                            Death counts are from the National Vital Statistics System database since this is the timeliest mortality data.
                            Number of deaths reported correspond to total number of deaths received and coded as of the date of analysis
                            but may not represent all deaths that occurred in that period, especially for recent data as
                            the time lag between when the death occurred and when the death certificate is completed, submitted to NCHS and 
                            processed for reporting purposes can be large -- between 1 and 8 weeks depending upon jurisdiction. 
                            Data for New York state exclude New York City. Expected deaths based on simple average deaths 
                            by week-of-year for 2015-2019.
                            """
                            )
                            ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_data_age", className="ml-auto")
                ),
            ],
            id="modal_data_age",
        ),
    ],
    style={"margin-left": "15px","margin-top": "15px"}
)

@app.callback(
    Output("modal_data_age", "is_open"),
    [Input("open_data_age", "n_clicks"), Input("close_data_age", "n_clicks")],
    [State("modal_data_age", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

modal_data_race = html.Div(
    [
        dbc.Button("About Weekly Death Data", id="open_data_race"),
        dbc.Modal(
            [
                dbc.ModalHeader("About the Data"),
                dbc.ModalBody(
                dcc.Markdown(
                            f"""
                            Data presented are raw death counts from all causes by week-of-year.
                            Death counts are from the National Vital Statistics System database since this is the timeliest mortality data.
                            Number of deaths reported correspond to total number of deaths received and coded as of the date of analysis
                            but may not represent all deaths that occurred in that period, especially for recent data as
                            the time lag between when the death occurred and when the death certificate is completed, submitted to NCHS and 
                            processed for reporting purposes can be large -- between 1 and 8 weeks depending upon jurisdiction. 
                            Data for New York state exclude New York City. Expected deaths based on simple average deaths 
                            by week-of-year for 2015-2019.
                            """
                            )
                            ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_data_race", className="ml-auto")
                ),
            ],
            id="modal_data_race",
        ),
    ],
    style={"margin-left": "15px","margin-top": "15px"}
)

@app.callback(
    Output("modal_data_race", "is_open"),
    [Input("open_data_race", "n_clicks"), Input("close_data_race", "n_clicks")],
    [State("modal_data_race", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# ## Call-backs and Control Utilities

# In[211]:


# Dropdown
state_dropdown = html.P([
            html.Label("Select One or More States"),
            dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': i, 'value': i} for i in state_df['state'].unique().tolist()],
            multi=True,
            value=['CA','TX','FL','GA','MN','WI','ND','SD'],
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
                        'display': 'inline-block',
                        'padding-left' : '10px'})

# Dropdown
state_dropdown_age = html.P([
            dcc.Dropdown(
            id='state-dropdown_age',
            options=[{'label': i, 'value': i} for i in excess_deaths_age['state'].dropna().unique().tolist()],
            multi=False,
            value='US',
            searchable= True)
            ], style = {'height': '20px',
                        'width' : '15%',
                        'fontSize' : '15px',
                        'display': 'inline-block',
                        'padding-left' : '10px'})

# Dropdown
state_dropdown_age_group = html.P([
            dcc.Dropdown(
            id='state-dropdown_age_group',
            options=[{'label': i, 'value': i} for i in excess_deaths_age['age_group'].dropna().unique().tolist()],
            multi=False,
            value='85 years and older',
            searchable= True)
            ], style = {'height': '20px',
                        'width' : '25%',
                        'fontSize' : '15px',
                        'display': 'inline-block',
                        'padding-left' : '10px'})

# Dropdown
state_dropdown_race = html.P([
            dcc.Dropdown(
            id='state-dropdown_race',
            options=[{'label': i, 'value': i} for i in excess_deaths_race['state'].dropna().unique().tolist()],
            multi=False,
            value='US',
            searchable= True)
            ], style = {'height': '20px',
                        'width' : '15%',
                        'fontSize' : '15px',
                        'display': 'inline-block',
                        'padding-left' : '10px'})

# Dropdown
state_dropdown_race_group = html.P([
            dcc.Dropdown(
            id='state-dropdown_race_group',
            options=[{'label': i, 'value': i} for i in excess_deaths_race['race_ethnicity'].dropna().unique().tolist()],
            multi=False,
            value='Non-Hispanic White',
            searchable= True)
            ], style = {'height': '20px',
                        'width' : '25%',
                        'fontSize' : '15px',
                        'display': 'inline-block',
                        'padding-left' : '10px'})


# ## Define HTML

# In[212]:


#---------------------------------------------------------------------------
# DASH App formating
#---------------------------------------------------------------------------
            
# App Layout
app.layout = dbc.Container(fluid=True, children=[
    ## Top
    html.Label(children="Data Retrieved " + today),
    html.Br(),
    
    ## 
    dbc.Row([
        dbc.Col(width=12, children=[
        state_dropdown,
        html.Br(),html.Br()
        ]),
    ]),
    
    dbc.Row([
        dbc.Col(width=12, children=[   
            dbc.Col(dbc.Row(
            children=[html.H4("Figure 1: Observed vs Expected Deaths in  "),state_dropdown_alt])),
            dbc.Col(dcc.Graph(id="excess_deaths")),
            modal_data,html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 2: Excess Deaths by States")), 
            dbc.Col(dcc.Graph(id="excess_deaths_states")),
            modal_calc,html.Br(),html.Br(),

            dbc.Col(dbc.Row(
            children=[html.H4("Figure 3: Observed vs Expected Deaths in  "),state_dropdown_age, 
                      html.H4("; Age Group  "), state_dropdown_age_group])),
            dbc.Col(dcc.Graph(id="excess_deaths_age")),
            modal_data_age,html.Br(),html.Br(),
            
            dbc.Col(dbc.Row(
            children=[html.H4("Figure 4: Observed vs Expected Deaths in  "),state_dropdown_race, 
                      html.H4("; Race/Ethnicity  "), state_dropdown_race_group])),
            dbc.Col(dcc.Graph(id="excess_deaths_race")),
            modal_data_race,html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 5: Vaccination Progress")), 
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="vaccines_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="vaccines_pc"), label="% of Total Population"),
                dbc.Tab(dcc.Graph(id="vaccines_util"), label="Vaccine Utilization")
            ]),html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 6: New Cases (7-day Moving Avg.)")), 
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="positive_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="positive_pc"), label="Per 10,000")
            ]),html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 7: New Hospitalizations (7-day Moving Avg.)")), 
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="curhospital_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="curhospital_pc"), label="Per 10,000")
            ]),html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 8: New Deaths (7-day Moving Avg.)")),
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="newdeaths_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="newdeaths_pc"), label="Per 10,000")
            ]),html.Br(),html.Br(),
            
            dbc.Col(html.H4("Figure 9: Total Deaths")),
            dbc.Tabs(className="nav", children=[
                dbc.Tab(dcc.Graph(id="totdeaths_raw"), label="Raw Data"),
                dbc.Tab(dcc.Graph(id="totdeaths_pc"), label="Per 10,000")
            ]),
        ])
    ], no_gutters=False)
])


# # 3. Run Application

# In[213]:


if __name__ == '__main__':
    #app.run_server(debug=True, use_reloader=False)  # Jupyter
    app.run_server(debug=False,host='0.0.0.0')    # Use this line prior to heroku deployment
    #application.run(debug=False, port=8080) # Use this line for AWS

