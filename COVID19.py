#!/usr/bin/env python
# coding: utf-8

# In[50]:


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

# In[51]:


excess_deaths = pd.read_csv('s3://mncovid19data/excess_deaths.csv',index_col=False)
state_df = pd.read_csv('s3://mncovid19data/state_df.csv',index_col=False)

# Load json file
with open('./Data/geojson-counties-fips.json') as response:  # Loads local file
    counties = json.load(response)    


# In[52]:


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

# In[53]:


# Initialize Dash
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Covid-19 U.S. Dashboard'
server = app.server  # Name Heroku will look for


# ## (Row 2, Col 1) U.S. Excess Deaths

# In[54]:



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

# In[55]:


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


# ##  (Row 3, Col 1) Line Graph:  Positive Cases over Time by State (7-day Rolling Average)

# In[56]:


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


# ## (Row 3, Col 2)  Line Graph: Hospitalizations over Time by State (7-day Rolling Average)

# In[57]:


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


# ## (Row 4, Col 1)  Line Graph: Daily Deaths by State (7-day Rolling Average)

# In[58]:


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


# ## (Row 4, Col 2) Line Graph: Cumulative Deaths by State

# In[59]:


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


# In[60]:


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


# ## Call-backs and Control Utilities

# In[61]:


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
                        'display': 'inline-block',
                        'padding-left' : '10px'})

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

# In[62]:


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


# In[63]:


#---------------------------------------------------------------------------
# DASH App formating
#---------------------------------------------------------------------------
header = html.H1(children="COVID-19 TRENDS (as of " + today + ")")

state_desc = dcc.Markdown(
            f"""
The following graphs depict Covid-19 trends. The graphs are interactive; 
e.g., hover your cursor over a data-series to observe specific values.
Select midwestern states are included by default but you can modify the analysis by choosing a different subset of
states, periods, and/or standardize the statistics by population.
            """   
)
            
# App Layout
app.layout = dbc.Container(fluid=True, children=[
    ## Top
#    navbar, 
#    html.Br(),html.Br(),html.Br(),html.Br(),
    
    ## 
    dbc.Row([
        dbc.Col(width=12, children=[
        state_dropdown, slider,
        html.Br(),html.Br()
        ]),

        ### left plots
        dbc.Col(width=6, children=[   
            dbc.Row(
            children=[html.H4("Observed vs Expected Deaths in:  "),state_dropdown_alt]),
            dbc.Col(dcc.Graph(id="excess_deaths")),
            modal_data,
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
            modal_calc,html.Br(),html.Br(),
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
#    html.Br(),html.Br(),
#    navbar_footer
])


# # 3. Run Application

# In[64]:


if __name__ == '__main__':
    #app.run_server(debug=True, use_reloader=False)  # Jupyter
    app.run_server(debug=False,host='0.0.0.0')    # Use this line prior to heroku deployment
    #application.run(debug=False, port=8080) # Use this line for AWS

