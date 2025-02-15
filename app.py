import os
import pathlib
import numpy as np
import datetime as dt
import dash
from dash import dcc
from dash import html
import pandas as pd
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rayleigh
from plotly.subplots import make_subplots
# from db.api import get_wind_data, get_wind_data_by_id
# http://127.0.0.1:8050/
#https://velvety-alfajores-c34948.netlify.app/dashboard.html#menu3
# Load data
existing_file_path='master_file_cold_stage_22.xlsx'
parent_df = pd.read_excel('AutoDBD2/master_file_cold_stage_222.xlsm', header=[0, 1, 2])
# print(parent_df.columns)
# Flatten the multi-level columns
# parent_df = pd.DataFrame(data)

# Process columns, joining strings and handling non-string elements
new_columns = []
for col in parent_df.columns:
    if all(isinstance(x, str) for x in col):
        new_col = ' '.join(col).strip()
    else:
        # Handle the case where one or more elements in col are not strings
        new_col = ' '.join(str(x) for x in col).strip()  # Convert non-strings to strings
    new_columns.append(new_col)

parent_df.columns = new_columns

# print(parent_df)
# print()
# Rename columns for easier access
parent_df.rename(columns={
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 0_level_1 Unnamed: 0_level_2': 'Sr. No.',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 1_level_1 Unnamed: 1_level_2': 'Date',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 2_level_1 Unnamed: 2_level_2': 'Engine Number',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 3_level_1 Unnamed: 3_level_2': 'Model',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 4_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Engine Speed1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 5_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Compression Pressure1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 6_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Lub Oil Pressure1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 7_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Magneto Current1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 8_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Magneto Voltage1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 9_level_1 STEP 1 (IDLING RPM, 650 RPM)': 'Torque1',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 10_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Engine Speed2',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 11_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Compression Pressure2',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 12_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Lub Oil Pressure2',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 13_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Magneto Current2',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 14_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Magneto Voltage2',
    'COLD TEST STAGE 2 DATA\n(PERFORMANCE TEST) Unnamed: 15_level_1 STEP 2 (CRANKING RPM, 1800 RPM)': 'Torque2'
}, inplace=True)


# df_move_data = pd.read_excel('AutoDBD2/master_file_cold_test_merged.xlsx', header=[0, 1])
# # Append new DataFrame to existing DataFrame
# df = pd.concat([parent_df, df_move_data], ignore_index=True)
# Assume parent_df is already defined somewhere above this code.
df = parent_df

# Print column names to confirm
# print(df.columns)

# If your Date column is multi-indexed or has a complex name, access it appropriately.
# For simplicity, I'm assuming 'Date' is a single-level column name.
df = df.iloc[3:].reset_index(drop=True)
# Convert the 'Date' column to datetime
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%d:%m:%Y %H:%M:%S:%f')
except ValueError as e:
    print(f"Date conversion error: {e}")
    # You might need to investigate the exact format or errors in your 'Date' column.

# Find the minimum and maximum dates
min_date = df['Date'].min()
max_date = df['Date'].max()

# Adjust start_date and end_date to include 5 days before and after
df_start_date = (min_date - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
df_end_date = (max_date + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

# print('df_start_date:', df_start_date)
# print('df_end_date:', df_end_date)



range_df_original = pd.read_excel('AutoDBD2/RangeDir.xlsx', header=[0, 1])
range_df=range_df_original
# print(range_df.columns)
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

def get_model_ranges(model):
    # print("range_df_Check : ",range_df)
    # Filter the DataFrame for the selected model
    model_data = range_df[range_df[('Model', 'Unnamed: 2_level_1')] == model]
    # print("model is : ",model,"model range : ",model_data)
    # Extract ranges for the selected model
    compression_pressure = model_data[('Compression Pressure', 'Min')].values[0], model_data[('Compression Pressure', 'Max')].values[0]
    lub_oil_pressure = model_data[('Lub Oil Pressure', 'Min')].values[0], model_data[('Lub Oil Pressure', 'Max')].values[0]
    magneto_voltage = model_data[('Magneto Voltage ', 'Min')].values[0], model_data[('Magneto Voltage ', 'Max')].values[0]
    magneto_current = model_data[('Magneto Current ', 'Min')].values[0], model_data[('Magneto Current ', 'Max')].values[0]
    torque = model_data[('Torque', 'Min')].values[0], model_data[('Torque', 'Max')].values[0]
    
    return {
        'compression_pressure': compression_pressure,
        'lub_oil_pressure': lub_oil_pressure,
        'magneto_voltage': magneto_voltage,
        'magneto_current': magneto_current,
        'torque': torque
    }
import numpy as np
import pandas as pd

# def check_status_for_selected_model(df, model, model_ranges):
#     # Initialize status with NA values
#     df['Status'] = 'NA'
    
#     # Filter rows for the selected model
#     mask = df['Model'] == model
#     df_selected_model = df[mask]
    
#     # If there are no rows for the selected model, return the DataFrame with all NA
#     if df_selected_model.empty:
#         return df
    
#     # Extract ranges for the selected model
#     if model not in model_ranges:
#         raise ValueError(f"Model {model} is not in the model ranges.")
    
#     ranges = model_ranges[model]
#     comp_min, comp_max = ranges['compression_pressure']
#     lub_min, lub_max = ranges['lub_oil_pressure']
#     mag_volt_min, mag_volt_max = ranges['magneto_voltage']
#     mag_curr_min, mag_curr_max = ranges['magneto_current']
#     torque_min, torque_max = ranges['torque']
    
#     # Vectorized checks
#     is_comp_valid = df_selected_model['Compression Pressure1'].between(comp_min, comp_max)
#     is_lub_valid = df_selected_model['Lub Oil Pressure1'].between(lub_min, lub_max)
#     is_mag_volt_valid = df_selected_model['Magneto Voltage1'].between(mag_volt_min, mag_volt_max)
#     is_mag_curr_valid = df_selected_model['Magneto Current1'].between(mag_curr_min, mag_curr_max)
#     is_torque_valid = df_selected_model['Torque1'].between(torque_min, torque_max)
    
#     # Combine all checks
#     is_valid = is_comp_valid & is_lub_valid & is_mag_volt_valid & is_mag_curr_valid & is_torque_valid
    
#     # Assign 'Pass' or 'Fail' to the selected model rows
#     df.loc[mask, 'Status'] = np.where(is_valid, 'Pass', 'Fail')
    
#     return df

def check_status_vectorized(df, model_ranges,selected_stage):
    # Initialize status with NA values
    status = pd.Series(index=df.index, dtype='object').fillna('NA')
    
    # Iterate over each model and apply vectorized checks
    for model, ranges in model_ranges.items():
        mask = df['Model'] == model
        # print("mask sample : ",mask)
        # If no rows match the current model, continue to the next one
        if not mask.any():
            continue
        
        # Extract ranges for the current model
        comp_min, comp_max = ranges['compression_pressure']
        lub_min, lub_max = ranges['lub_oil_pressure']
        mag_volt_min, mag_volt_max = ranges['magneto_voltage']
        mag_curr_min, mag_curr_max = ranges['magneto_current']
        torque_min, torque_max = ranges['torque']
        
        # Vectorized checks
        is_comp_valid = df.loc[mask, 'Compression Pressure'+str(selected_stage)].between(comp_min, comp_max)
        is_lub_valid = df.loc[mask, 'Lub Oil Pressure'+str(selected_stage)].between(lub_min, lub_max)
        is_mag_volt_valid = df.loc[mask, 'Magneto Voltage'+str(selected_stage)].between(mag_volt_min, mag_volt_max)
        is_mag_curr_valid = df.loc[mask, 'Magneto Current'+str(selected_stage)].between(mag_curr_min, mag_curr_max)
        is_torque_valid = df.loc[mask, 'Torque'+str(selected_stage)].between(torque_min, torque_max)
        
        # Combine all checks
        is_valid = is_comp_valid & is_lub_valid & is_mag_volt_valid & is_mag_curr_valid & is_torque_valid
        
        # Assign 'Pass' or 'Fail' to the status Series for rows matching the current model
        status[mask] = np.where(is_valid, 'Pass', 'Fail')
    
    return status


# def check_status(row):
#     model = row['Model']
    
#     # Get range values for the model
#     ranges = get_model_ranges(model)
    
#     # Extract ranges
#     comp_min, comp_max = ranges['compression_pressure']
#     lub_min, lub_max = ranges['lub_oil_pressure']
#     mag_volt_min, mag_volt_max = ranges['magneto_voltage']
#     mag_curr_min, mag_curr_max = ranges['magneto_current']
#     torque_min, torque_max = ranges['torque']
    
#     # Check if the values fall within the specified ranges
#     status = 'Pass'
#     if not (comp_min <= row['Compression Pressure1'] <= comp_max):
#         status = 'Fail'
#     if not (lub_min <= row['Lub Oil Pressure1'] <= lub_max):
#         status = 'Fail'
#     if not (mag_volt_min <= row['Magneto Voltage1'] <= mag_volt_max):
#         status = 'Fail'
#     if not (mag_curr_min <= row['Magneto Current1'] <= mag_curr_max):
#         status = 'Fail'
#     if not (torque_min <= row['Torque1'] <= torque_max):
#         status = 'Fail'
    
#     return status




models = df['Model'].dropna().unique()
dates = df['Date'].dropna().unique()
dates = sorted(dates)  # Ensure dates are sorted
dates_range = [dates[0], dates[-1]]  # Define the range for the slider


# GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Cold Test Analysis"

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

app.layout = html.Div(
    [
        html.Div([
            # [    dcc.Store(id='selected-parameter-store', data={'parameter': 'Engine Speed'}),
                # Select Date Range
                html.Div(
                    [
                        html.H6('Select Date Range:'),
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            start_date=df_start_date,
                            end_date=df_end_date,
                            display_format='YYYY-MM-DD',
                            style={
                                'width': '200px',  # Fixed width
                                'height': '50px',
                                'font-family': 'Poppins, sans-serif',
                                'font-size': '14px',
                                'border-radius': '8px',
                                'padding': '0px',  # Ensure padding is zero to prevent size issues
                                'box-shadow': '0 2px 6px rgba(0,0,0,0.1)',
                                'border': '1px solid #ddd',
                                'background-color': '#ffffff'
                            }
                        )
                    ],
                    style={
                        'flex': '1',  # Adjust space taken by the DatePickerRange
                        'margin-right': '2px',  # Space between DatePicker and Dropdown
                    }
                ),

                # Select Engine Model
                html.Div(
                    [
                        html.H6('Select Engine Model:'),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[{'label': model, 'value': model} for model in models],
                            value=models[0],
                            clearable=False,
                            optionHeight=40,
                            style={
                                'width': '200px',  # Fixed width
                                'font-family': 'Poppins, sans-serif',
                                'font-size': '14px',
                                'border-radius': '8px',
                                'border': '1px solid #ddd',
                                'box-shadow': '0 2px 6px rgba(0,0,0,0.1)',
                                'margin-bottom': '5px'
                                # 'padding': '5px',
                            }
                        ),
                        dcc.Dropdown(
                            id='stage-dropdown',
                            options=[
                                {'label': 'Stage-I', 'value': 1},
                                {'label': 'Stage-II', 'value': 2}
                            ],
                            value=1,  # Default value set to Stage-I (1)
                            clearable=True,
                            optionHeight=40,
                            style={'width': '100%'}  # Ensure it takes full width of the container
                        )
                    ],
                    style={
                        'flex': '1',  # Adjust space taken by the Dropdown
                        'margin-right': '10px',  # Space between Dropdown and Buttons
                    }
                ),

                # Parameter Buttons
                html.Div(
                    [
                        dcc.RadioItems(
                            id='parameter-radio',
                            options=[
                                {'label': 'Engine Speed', 'value': 'Engine Speed'},
                                {'label': 'Compression Pressure', 'value': 'Compression Pressure'},
                                {'label': 'Lub Oil Pressure', 'value': 'Lub Oil Pressure'},
                                {'label': 'Magneto Voltage', 'value': 'Magneto Voltage '},
                                {'label': 'Magneto Current', 'value': 'Magneto Current '},
                                {'label': 'Torque', 'value': 'Torque'}
                            ],
                            value='Engine Speed',  # Default value
                            className='dash-radio-items'
                        ),
                    ],
                    style={
                        'flex': '4',  # Adjust space taken by the Buttons
                        'display': 'block',
                        'flex-wrap': 'wrap',
                        'justify-content': 'space-between',
                        'align-items': 'center',
                        'gap': '5px'  # Space between buttons
                    }
                ),
                # dcc.Store(id='selected-parameter')  # Store for the selected parameter
            ],
            style={
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'space-between',
                'flex-wrap': 'wrap',  # Allow wrapping for responsiveness
                'padding': '5px',
            }
        ),
        
    

            html.Div(
                [
                    # wind speed
                    html.Div(
                        [
 html.Div(
    [
        # Button section
        html.Div(
            [
                html.Button(id='model-pass-count', className='count-button'),
                html.Button(id='model-fail-count', className='count-button'),
                html.Button(id='total-pass-count', className='count-button'),
                html.Button(id='total-fail-count', className='count-button'),
                dcc.Graph(
                    id='pass-fail-pie-chart',
                    figure=dict(
                        layout=dict(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                        )
                    ),
                    style={
                        'position':'absolute',
                        'top': '-70px',
                        'right': '120px',
                        'width': '260px',
                        'height': '260px',
                        'margin-right': '5px',  # Reduced space between charts
                        'margin': '0',
                        'padding': '0',  # Ensure no extra padding
                    }
                ),
                dcc.Graph(
                    id='total-pass-fail-pie-chart',
                    figure=dict(
                        layout=dict(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                        )
                    ),
                    style={
                        'position':'absolute',
                        'top': '-70px',
                        'right': '-80px',
                        'width': '260px',
                        'height': '260px',
                        'margin': '0',
                        'padding': '0',  # Ensure no extra padding
                    }
                ),
            ],
            className="button-container"  # Flexbox container for buttons
        ),
        
       # Pie charts section
        # html.Div(
        #     [
                
        #     ],
        #     className="pie-charts-container"  # Flexbox container for pie charts
        # ),
    ],
    className="header-container",  # Flexbox container for buttons and pie charts
    style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'space-between',
        'flex-wrap': 'wrap',
        'padding': '5px',  # Fixed padding
    }
),
                            dcc.Graph(
                                id="scatter-plots",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor=app_color["graph_bg"],
                                        paper_bgcolor=app_color["graph_bg"],
                                    )
                                ),
                            ),
                            
                        #         html.Div(
                        #     [
                                
                        #     ],
                        #     style={'position': 'relative', 'width': '100%', 'height': '500px'}
                        # ),
                           

                        ],
                        className="scatter__window",
                        style={'position': 'relative'}
                    ),
                    html.Div(
                        [
                            # histogram
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H6(
                                                "ENGINE PARAMETER HISTOGRAM",
                                                className="graph__title",
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dcc.RangeSlider(
                                                id="param-slider",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0,100],
                                                updatemode="drag",
                                                marks={i: {'label': str(i)} for i in range(1, 101, 10)}
                                                
                                            )
                                        ],
                                        className="slider",
                                    ),

                                    dcc.Graph(
                                        id='normal-plots',
                                              figure=dict(
                                            layout=dict(
                                                plot_bgcolor=app_color["graph_bg"],
                                                paper_bgcolor=app_color["graph_bg"],
                                            )
                                        ),),
                                
                               
                               
                                    
                                    # dcc.Graph(
                                    #     id="param-histogram",
                                    #     figure=dict(
                                    #         layout=dict(
                                    #             plot_bgcolor=app_color["graph_bg"],
                                    #             paper_bgcolor=app_color["graph_bg"],
                                    #         )
                                    #     ),
                                    # ),
                                ],
                                className="graph__container first",
                            ),
                            # wind direction
                            html.Div(
                                [
                                 # Pie Charts
                        # html.Div(
                        #     [
                        #         html.Div(
                        #             [
                        #                 # html.H6("Pass vs Fail Counts", className="graph__title"),
                        #                 dcc.Graph(id="pass-fail-pie-chart"),
                        #             ],
                        #             className="graph__container",
                        #             style={'flex': '1', 'margin-right': '10px'}
                        #         ),
                        #         html.Div(
                        #             [
                        #                 # html.H6("Total Pass vs Total Fail Counts", className="graph__title"),
                        #                 dcc.Graph(id="total-pass-fail-pie-chart"),
                        #             ],
                        #             className="graph__container",
                        #             style={'flex': '1'}
                        #         ),
                        #     ],
                        #     className="pie-charts-container",
                        #     style={
                        #         'display': 'flex',
                        #         'justify-content': 'space-between',
                        #         'gap': '10px'
                        #     }
                        # ),
                                ],
                                className="graph__container second",
                            ),
                        ],
                        className="one-third column histogram__direction",
                    ),
                ],
                className="app__content",
            ),
        ],
        className="app__container",
    )




# def get_current_time():
#     """ Helper function to get the current time in seconds. """

#     now = dt.datetime.now()
#     total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)
#     return total_time



import numpy as np

@app.callback(
    [Output('param-slider', 'min'),
     Output('param-slider', 'max'),
     Output('param-slider', 'value'),
     Output('param-slider', 'marks')],
    [Input('model-dropdown', 'value'),
     Input('parameter-radio', 'value'),
     Input('stage-dropdown', 'value'),]
)
def update_sliders(selected_model, selected_parameter,selected_stage):
    # print("selected_parameter:", selected_parameter)  # Debugging info
    
    # Filter the DataFrame for the selected model
    model_data = range_df[range_df[('Model', 'Unnamed: 2_level_1')] == selected_model]
    
    # Check if the selected parameter exists in model_data
    if selected_parameter in ['Compression Pressure', 'Lub Oil Pressure', 'Magneto Voltage ', 'Magneto Current ', 'Torque']:
        try:
            # Get min and max values for the selected parameter
            param_min = model_data[(selected_parameter, 'Min')].values[0]
            param_max = model_data[(selected_parameter, 'Max')].values[0]
            
            # Convert numpy types to standard Python types
            param_min = float(param_min)  # Ensure param_min is a float
            param_max = float(param_max)  # Ensure param_max is a float

            default_value = [param_min, param_max]
            # print("default_value : ", default_value)
            
            # Calculate step size for marks
            param_range = param_max - param_min
            param_step = param_range / 10
            
            # Create marks dictionary using np.arange for floating-point ranges
            default_marks = {round(float(i), 2): {'label': f"{round(float(i), 2):.2f}"} for i in np.arange(param_min-4*param_step, param_max + 4*param_step, param_step)}
            
            return param_min-4*param_step, param_max+4*param_step, default_value, default_marks
        
        except KeyError as e:
            print(f"KeyError: {e}")  # Handle missing keys in model_data
            default_min = 0
            default_max = 100
            default_value = [default_min, default_max]
            
            # Create a default marks dictionary with integer keys
            default_marks = {i: {'label': f"{i:.2f}"} for i in range(default_min, default_max + 1, 10)}
            
            return default_min, default_max, default_value, default_marks
    else:
            # print(f"Parameter '{selected_parameter}' not found in model data.")
            if selected_stage==1:
                default_min = 600
                default_max = 700
                default_value = [default_min, default_max]
            else:
                default_min = 1750
                default_max = 1900
                default_value = [default_min, default_max]
                
            # Create a default marks dictionary with integer keys
            default_marks = {i: {'label': f"{i:.2f}"} for i in range(default_min, default_max + 1, 10)}
            
            return default_min, default_max, default_value, default_marks  # Default fallback values

color_map = {
    'Engine Speed': '#ADD8E6',  # Light Blue
    'Compression Pressure': '#90EE90',  # Light Green
    'Lub Oil Pressure': '#F08080',  # Light Coral
    'Magneto Voltage': '#D8BFD8',  # Thistle (a light purple shade)
    'Magneto Current': '#FFDAB9',  # Peach Puff (a light orange shade)
    'Torque': '#D2B48C'  # Tan (a light brown shade)
}




@app.callback(
    Output('scatter-plots', 'figure'),
    [
        Input('model-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('stage-dropdown', 'value'),
        Input('parameter-radio', 'value'),
        Input('param-slider', 'min'),
        Input('param-slider', 'max'),
        Input('param-slider', 'value')
    ]
)
def update_scatter_plot(selected_model, start_date, end_date, selected_stage, selected_parameter, slider_min, slider_max, slider_value):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    parameter_with_stage = f"{selected_parameter.strip()}{selected_stage}"

    filtered_df = df[
        (df['Date'] >= start_date) & 
        (df['Date'] <= end_date) & 
        (df['Model'] == selected_model) &
        (df[selected_parameter.strip() + str(selected_stage)] >= slider_value[0]) & 
        (df[selected_parameter.strip() + str(selected_stage)] <= slider_value[1])
    ]

    if parameter_with_stage not in filtered_df.columns:
        raise ValueError(f"Selected parameter '{parameter_with_stage}' not found in DataFrame columns.")
    
    mean_value = filtered_df[parameter_with_stage].mean()
    
    filtered_df = filtered_df.sort_values(by='Date')

    fig = go.Figure()

    color = color_map.get(selected_parameter.strip(), 'black')

    trace = go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df[parameter_with_stage],
        mode='lines+markers',
        name=f'{parameter_with_stage} Line',
        line=dict(color=color),
        marker=dict(color=color, size=8)
    )
    fig.add_trace(trace)

    fig.update_layout(
        title=f'Scatter Plot for {selected_model} - {parameter_with_stage[:-1]}',
        xaxis_title='Date',
        yaxis_title=parameter_with_stage,
        height=800,
        width=1000,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white')
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white')
    )

    return fig



app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

from scipy.stats import norm

# def update_histogram(start_date, end_date, selected_model, selected_stage, selected_parameter):
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)
#     parameter_with_stage = f"{selected_parameter.strip()}{selected_stage}"

#     # Filter DataFrame
#     filtered_df = df[
#         (df['Date'] >= start_date) & 
#         (df['Date'] <= end_date) & 
#         (df['Model'] == selected_model)
#     ]

#     # Compute mean, standard deviation, and variance
#     mean_value = filtered_df[parameter_with_stage].mean()
#     std_dev = filtered_df[parameter_with_stage].std()
#     variance_value = std_dev ** 2

#     # Create histogram trace
#     histogram_trace = go.Histogram(
#         x=filtered_df[parameter_with_stage],
#         nbinsx=20,  # Number of bins
#         marker=dict(color='rgba(200, 200, 250, 0.8)')
#     )

#     # Generate x values for the normal distribution curve
#     x_values = np.linspace(filtered_df[parameter_with_stage].min(), filtered_df[parameter_with_stage].max(), 100)
#     # Generate y values for the normal distribution curve
#     y_values = norm.pdf(x_values, mean_value, std_dev) * len(filtered_df[parameter_with_stage]) * (filtered_df[parameter_with_stage].max() - filtered_df[parameter_with_stage].min()) / 20

#     # Create the normal distribution curve trace
#     curve_trace = go.Scatter(
#         x=x_values,
#         y=y_values,
#         mode='lines',
#         line=dict(color='blue ', width=2),
#         showlegend=False  # Hide the legend for this trace
#     )

#     # Create the figure
#     # fig = go.Figure(data=[histogram_trace, curve_trace])
#     fig = go.Figure(data=[histogram_trace])

#     # Update layout with custom background color and vertical lines
#     fig.update_layout(
#         title=f'Overall Histogram of {selected_parameter} for {selected_model}',
#         xaxis_title=selected_parameter,
#         yaxis_title='Count',
#         plot_bgcolor=app_color["graph_bg"],
#         paper_bgcolor=app_color["graph_bg"],
#         font={"color": "#fff"}
#     )

# # Add vertical lines for mean and median
#     fig.add_vline(
#         x=mean_value,
#         line=dict(color="cyan", width=2, dash="dash")
#     )
#     fig.add_vline(
#         x=filtered_df[parameter_with_stage].median(),
#         line=dict(color="magenta", width=2, dash="dash")
#     )

#     # Add annotations with arrows to simulate arrowheads
#     fig.add_annotation(
#         x=mean_value,
#         y=filtered_df[parameter_with_stage].max() * 0.9,  # Adjust the y position to be within the plot area
#         text=f"Avg: {mean_value:.2f}",
#         showarrow=True,
#         arrowhead=2,
#         arrowcolor="cyan",
#         font=dict(size=12, color='cyan'),
#         ax=-60,  # Horizontal offset of the arrow's end
#         ay=-50  # Vertical offset of the arrow's end
#     )
#     fig.add_annotation(
#         x=filtered_df[parameter_with_stage].median(),
#         y=filtered_df[parameter_with_stage].max() * 0.9,  # Adjust the y position to be within the plot area
#         text=f"Median: {filtered_df[parameter_with_stage].median():.2f}",
#         showarrow=True,
#         arrowhead=2,
#         arrowcolor="magenta",
#         font=dict(size=12, color='magenta'),
#         ax=60,  # Horizontal offset of the arrow's end
#         ay=-50  # Vertical offset of the arrow's end
#     )

#     # Update x and y axes
#     fig.update_xaxes(
#         title=selected_parameter,
#         gridcolor='rgba(255,255,255,0.2)',
#         zerolinecolor='rgba(255,255,255,0.2)',
#         linecolor='rgba(255,255,255,0.2)',
#         tickfont=dict(color='white')
#     )
#     fig.update_yaxes(
#         title='Count',
#         gridcolor='rgba(255,255,255,0.2)',
#         zerolinecolor='rgba(255,255,255,0.2)',
#         linecolor='rgba(255,255,255,0.2)',
#         tickfont=dict(color='white')
#     )

#     return fig


# # Example usage in a callback
# @app.callback(
#     Output('histogram-plot', 'figure'),
#     [Input('date-picker-range', 'start_date'),
#      Input('date-picker-range', 'end_date'),
#      Input('model-dropdown', 'value'),
#      Input('stage-dropdown', 'value'),
#      Input('parameter-radio', 'value')]
# )
# def update_histogram_callback(start_date, end_date, selected_model, selected_stage, selected_parameter):
#     return update_histogram(start_date, end_date, selected_model, selected_stage, selected_parameter)

@app.callback(
    Output('normal-plots', 'figure'),
    [
        Input('model-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('stage-dropdown', 'value'),
        Input('parameter-radio', 'value'),
        Input('param-slider', 'min'),
        Input('param-slider', 'max'),
        Input('param-slider', 'value')
    ]
)
def update_normal_curve(model, start_date, end_date, selected_stage, selected_parameter, min_param, max_param, slider_value):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # parameter_with_stage = f"{selected_parameter.strip()}{selected_stage}"

    # # Filter DataFrame
    # filtered_df = df[
    #     (df['Date'] >= start_date) & 
    #     (df['Date'] <= end_date) & 
    #     (df['Model'] == selected_model)
    # ]

    # # Compute mean, standard deviation, and variance
    # mean_value = filtered_df[parameter_with_stage].mean()
    # std_dev = filtered_df[parameter_with_stage].std()
    # variance_value = std_dev ** 2

    # # Create histogram trace
    # histogram_trace = go.Histogram(
    #     x=filtered_df[parameter_with_stage],
    #     nbinsx=20,  # Number of bins
    #     marker=dict(color='rgba(200, 200, 250, 0.8)')
    # )
    # Filter the DataFrame based on inputs
    filtered_df = df[
        (df['Date'] >= start_date) & 
        (df['Date'] <= end_date) & 
        (df['Model'] == model) &
        (df[selected_parameter.strip() + str(selected_stage)] >= slider_value[0]) & 
        (df[selected_parameter.strip() + str(selected_stage)] <= slider_value[1])
    ]
    
    parameter_with_stage = f"{selected_parameter.strip() + str(selected_stage)}"
    
    # Compute statistics
    mean_value = filtered_df[parameter_with_stage].mean()
    std_dev = filtered_df[parameter_with_stage].std()
    data_min = filtered_df[parameter_with_stage].min()
    data_max = filtered_df[parameter_with_stage].max()
    
    # Generate x values for the normal distribution curve
    x_values = np.linspace(filtered_df[parameter_with_stage].min(), filtered_df[parameter_with_stage].max(), 100)
    # Generate y values for the normal distribution curve
    y_values = norm.pdf(x_values, mean_value, std_dev)
    
    # Create histogram trace
    histogram_trace = go.Histogram(
        x=filtered_df[parameter_with_stage],
        nbinsx=20,
        marker=dict(color='rgba(200, 200, 250, 0.8)')
    )
    
    # Create normal distribution curve trace
    curve_trace = go.Scatter(
        x=x_values,
        y=y_values * len(filtered_df[parameter_with_stage]) * (data_max - data_min) / 20,  # Scale to match histogram
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False
    )
    
    # Create shaded area under the curve
    fill_trace = go.Scatter(
        x=np.concatenate([x_values, x_values[::-1]]),
        y=np.concatenate([y_values * len(filtered_df[parameter_with_stage]) * (data_max - data_min) / 20, np.zeros_like(y_values)]),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Area under the Curve'
    )
    
    # Create the figure
    fig = go.Figure(data=[histogram_trace, curve_trace, fill_trace])
    
    # Update layout with custom background color
    fig.update_layout(
        title=f'Normal Distribution of {selected_parameter} for {model}',
        xaxis_title=selected_parameter,
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={"color": "#fff"},
        showlegend=False
    )
    # Add vertical lines for mean and median
    fig.add_vline(
        x=mean_value,
        line=dict(color="cyan", width=2, dash="dash")
    )
    fig.add_vline(
        x=filtered_df[parameter_with_stage].median(),
        line=dict(color="magenta", width=2, dash="dash")
    )

    # Add annotations with arrows to simulate arrowheads
    fig.add_annotation(
        x=mean_value,
        y=filtered_df[parameter_with_stage].max() * 0.9,  # Adjust the y position to be within the plot area
        text=f"Avg: {mean_value:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="cyan",
        font=dict(size=12, color='cyan'),
        ax=-60,  # Horizontal offset of the arrow's end
        ay=-50  # Vertical offset of the arrow's end
    )
    fig.add_annotation(
        x=filtered_df[parameter_with_stage].median(),
        y=filtered_df[parameter_with_stage].max() * 0.9,  # Adjust the y position to be within the plot area
        text=f"Median: {filtered_df[parameter_with_stage].median():.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="magenta",
        font=dict(size=12, color='magenta'),
        ax=60,  # Horizontal offset of the arrow's end
        ay=-50  # Vertical offset of the arrow's end
    )
    # Update x and y axes to ensure grid lines and ticks are visible
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white')
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.2)',
        zerolinecolor='rgba(255,255,255,0.2)',
        linecolor='rgba(255,255,255,0.2)',
        tickfont=dict(color='white')
    )
    
    return fig
# @app.callback(
#     [Output('model-pass-count', 'children'),
#      Output('model-fail-count', 'children'),
#      Output('total-pass-count', 'children'),
#      Output('total-fail-count', 'children')],
# [       Input('model-dropdown', 'value'),
#         Input('date-picker-range', 'start_date'),
#         Input('date-picker-range', 'end_date'),
#         Input('stage-dropdown', 'value'),
#         Input('parameter-radio', 'value'),
#         Input('param-slider', 'min'),
#         Input('param-slider', 'max'),
#         Input('param-slider', 'value')]
# )
# def update_counts(selected_model, start_date, end_date, selected_stage, selected_parameter, slider_min, slider_max, slider_value):
#     # Get pass and fail counts based on slider range
#     # counts = get_pass_fail_counts(slider_range)
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)

#     # Adjust column names based on the selected stage
#     parameter_with_stage = f"{selected_parameter.strip()}{selected_stage}"

#     # Filter DataFrame based on the inputs
#     filtered_df = df[
#         (df['Model'] == selected_model) & 
#         (df['Date'] >= start_date) & 
#         (df['Date'] <= end_date)
#     ]

#     # Check if the selected parameter with stage is in the columns of filtered_df
#     if parameter_with_stage not in filtered_df.columns:
#         raise ValueError(f"Selected parameter '{parameter_with_stage}' not found in DataFrame columns.")
    
#     # # Sort DataFrame by 'Date' to ensure sequential plotting
#     # filtered_df = filtered_df.sort_values(by='Date')

#     # mean_value = filtered_df[parameter_with_stage].mean()

#     if selected_parameter in ['Compression Pressure', 'Lub Oil Pressure', 'Magneto Voltage ', 'Magneto Current ', 'Torque']:
#         try:
#             # Update the min and max values in the model_data DataFrame
#             range_df.loc[(range_df[('Model', 'Unnamed: 2_level_1')] == selected_model) , 
#                             (selected_parameter, 'Min')] = slider_value[0]
#             range_df.loc[(range_df[('Model', 'Unnamed: 2_level_1')] == selected_model) , 
#                             (selected_parameter, 'Max')] = slider_value[1]
#             print("updated range_df : ",range_df)
#             # Store the updated DataFrame in a Dash Store
#             # return range_df.to_dict('records')
            
#         except KeyError as e:
#             print(f"KeyError: {e}")  # Handle missing keys in model_data
#             # return range_df.to_dict('records')
#     # Precompute ranges for all models
#     model_ranges = {model: get_model_ranges(model) for model in df['Model'].unique()}

    
#     # Apply the function to each row
#     # df['Status'] = df.apply(check_status, axis=1)
#     df['Status'] = check_status_vectorized(df, model_ranges,selected_stage)
#     # df_updated = check_status_for_selected_model(df, 'JU', model_ranges)
#     status_counts = df['Status'].value_counts(dropna=False) 
#     total_pass_count = status_counts.get('Pass', 0)
#     total_fail_count = status_counts.get('Fail', 0)
#     mask = df['Model'] == selected_model
#     status_counts = df[mask]['Status'].value_counts(dropna=False) 

#     # Get counts of 'Pass', 'Fail', and NA
#     pass_count = status_counts.get('Pass', 0)
#     fail_count = status_counts.get('Fail', 0)
#     na_count = status_counts.get('NA', 0)  # NA might be represented as NaN or 'NA'

#     # Print the counts
#     print(f"Pass count: {pass_count}")
#     print(f"Fail count: {fail_count}")
#     print(f"NA count: {na_count}")

#     # Filter the DataFrame to include only rows where the Status is 'Pass'
#     pass_rows = df[df['Status'] == 'Pass']

#     # Print all rows where the Status is 'Pass'
#     print(pass_rows)

    
#     # Return formatted strings to be displayed on the buttons
#     return f"{selected_model} Pass: {pass_count}", f"{selected_model} Fail: {fail_count}",f"total Pass: {total_pass_count}",f"total Fail: {total_fail_count}"


@app.callback(
    [Output('model-pass-count', 'children'),
     Output('model-fail-count', 'children'),
     Output('total-pass-count', 'children'),
     Output('total-fail-count', 'children'),
     Output('pass-fail-pie-chart', 'figure'),
     Output('total-pass-fail-pie-chart', 'figure')],
    [Input('model-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('stage-dropdown', 'value'),
     Input('parameter-radio', 'value'),
     Input('param-slider', 'min'),
     Input('param-slider', 'max'),
     Input('param-slider', 'value')]
)
def update_counts(selected_model, start_date, end_date, selected_stage, selected_parameter, slider_min, slider_max, slider_value):
    # Existing logic...
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Adjust column names based on the selected stage
    parameter_with_stage = f"{selected_parameter.strip()}{selected_stage}"

    # Filter DataFrame based on the inputs
    filtered_df = df[
        (df['Model'] == selected_model) & 
        (df['Date'] >= start_date) & 
        (df['Date'] <= end_date)
    ]

    # Check if the selected parameter with stage is in the columns of filtered_df
    if parameter_with_stage not in filtered_df.columns:
        raise ValueError(f"Selected parameter '{parameter_with_stage}' not found in DataFrame columns.")
    
    # # Sort DataFrame by 'Date' to ensure sequential plotting
    # filtered_df = filtered_df.sort_values(by='Date')

    # mean_value = filtered_df[parameter_with_stage].mean()

    if selected_parameter in ['Compression Pressure', 'Lub Oil Pressure', 'Magneto Voltage ', 'Magneto Current ', 'Torque']:
        try:
            # Update the min and max values in the model_data DataFrame
            range_df.loc[(range_df[('Model', 'Unnamed: 2_level_1')] == selected_model) , 
                            (selected_parameter, 'Min')] = slider_value[0]
            range_df.loc[(range_df[('Model', 'Unnamed: 2_level_1')] == selected_model) , 
                            (selected_parameter, 'Max')] = slider_value[1]
            # print("updated range_df : ",range_df)
            # Store the updated DataFrame in a Dash Store
            # return range_df.to_dict('records')
            
        except KeyError as e:
            print(f"KeyError: {e}")  # Handle missing keys in model_data
            # return range_df.to_dict('records')
    # Precompute ranges for all models
    model_ranges = {model: get_model_ranges(model) for model in df['Model'].unique()}

    
    # Apply the function to each row
    # df['Status'] = df.apply(check_status, axis=1)
    df['Status'] = check_status_vectorized(df, model_ranges,selected_stage)
    # df_updated = check_status_for_selected_model(df, 'JU', model_ranges)
    status_counts = df['Status'].value_counts(dropna=False) 
    total_pass_count = status_counts.get('Pass', 0)
    total_fail_count = status_counts.get('Fail', 0)
    mask = df['Model'] == selected_model
    status_counts = df[mask]['Status'].value_counts(dropna=False) 

    # Get counts of 'Pass', 'Fail', and NA
    pass_count = status_counts.get('Pass', 0)
    fail_count = status_counts.get('Fail', 0)
    na_count = status_counts.get('NA', 0) 

        # Calculate total and percentage
    total = pass_count + fail_count
    pass_percentage = pass_count / total * 100 if total > 0 else 0
    fail_percentage = fail_count / total * 100 if total > 0 else 0
     # Calculate total and percentage for total pass and fail
    total_pass_percentage = total_pass_count / (total_pass_count + total_fail_count) * 100 if (total_pass_count + total_fail_count) > 0 else 0
    total_fail_percentage = total_fail_count / (total_pass_count + total_fail_count) * 100 if (total_pass_count + total_fail_count) > 0 else 0

    # Data for Pass vs Fail Pie Chart
    pass_fail_pie_chart = {
        'data': [
            {
                'labels': ['Pass', 'Fail'],
                'values': [pass_count, fail_count],
                'type': 'pie',
                'hole': 0.4,  # Add this line to make it a donut chart
                'textinfo': 'none',  # Hide text inside the pie chart
                'showlegend': False,  # Hide default legend
                'marker': {'colors': ['#1f77b4', '#ff7f0e']},  # Colors for the segments
            }
        ],
        'layout': {
            # 'title': 'Pass vs Fail Counts',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#fff'},
            'annotations': [
                {
                    'text': f"Fail: {fail_count} ({fail_percentage:.1f}%)",
                    'x': 1.0,  # Position outside the chart, adjust for clear visibility
                    'y': -0.15,
                    'font': {'size': 14, 'color': '#fff'},
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -50,  # Arrow length, adjust as needed
                    'ay': 0,
                },
                {
                    'text': f"Pass: {pass_count} ({pass_percentage:.1f}%)",
                    'x': -0.2,  # Position outside the chart, adjust for clear visibility
                    'y': 1.2,
                    'font': {'size': 14, 'color': '#fff'},
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 50,  # Arrow length, adjust as needed
                    'ay': 0,
                }
            ]
        }
    }

    # Data for Total Pass vs Total Fail Pie Chart
    total_pass_fail_pie_chart = {
        'data': [
            {
                'labels': ['Total Pass', 'Total Fail'],
                'values': [total_pass_count, total_fail_count],
                'type': 'pie',
                'hole': 0.4,  # Add this line to make it a donut chart
                'textinfo': 'none',  # Hide text inside the pie chart
                'showlegend': False,  # Hide default legend
                'marker': {'colors': ['#2ca02c', '#d62728']},  # Colors for the segments
            }
        ],
        'layout': {
            # 'title': 'Total Pass vs Total Fail Counts',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            # 'plot_bgcolor': app_color["graph_bg"],
            # 'paper_bgcolor': app_color["graph_bg"],
            'font': {'color': '#fff'},
            'annotations': [
                {   'text': f"Total Fail: {total_fail_count} ({total_fail_percentage:.1f}%)",
                    
                    'x': 0.6,  # Position outside the chart, adjust for clear visibility
                    'y': -0.15,
                    'font': {'size': 14, 'color': '#fff'},
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -50,  # Arrow length, adjust as needed
                    'ay': 0,
                },
                {
                    'text': f"Total Pass: {total_pass_count} ({total_pass_percentage:.1f}%)",
                    'x': -0.5,  # Position outside the chart, adjust for clear visibility
                    'y': 1.2,
                    'font': {'size': 14, 'color': '#fff'},
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 50,  # Arrow length, adjust as needed
                    'ay': 0,
                }
            ]
        }
    }


    # Return formatted strings and pie chart figures
    return (
        f"{selected_model} Pass: {pass_count}",
        f"{selected_model} Fail: {fail_count}",
        f"total Pass: {total_pass_count}",
        f"total Fail: {total_fail_count}",
        pass_fail_pie_chart,
        total_pass_fail_pie_chart
    )



if __name__ == "__main__":
    app.run_server(debug=True)



