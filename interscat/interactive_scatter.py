import os
import re
import glob
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from skimage.io import imread
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output


opj = os.path.join
SCATTER_MARKER_SIZE = 10


def interactive_scatter(umap, path_images, path_additional_data=None, path_save=""):
    """
    Creates an interactive UMAP visualisation plot with option to view image associated to the selected data point.

    Args:
        umap (str|pd.DataFrame): path to the UMAP embedding or pd.Dataframe with c1, c2, filename,
            class, origin, subset columns,
        path_images (str): path to where images are stored,
        path_additional_data (str): path to the csv file containing additional data which will be joined by 'filename',
        path_save (str): path for optional saving of the selected points.
    """
    # read umap
    if type(umap) is pd.DataFrame:
        data = umap
    else:
        if umap.endswith('.csv'):
            data = pd.read_csv(umap)
        else:
            data = pd.read_excel(umap)

    # join additional data and prepare paths to the images
    if path_additional_data is not None:
        data_additional = pd.read_csv(path_additional_data)
        data = pd.merge(data, data_additional, on="filename", suffixes=(None, "_add"))
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')].copy()
    data['path'] = data['filename'].apply(lambda x: opj(path_images, str(os.path.splitext(x)[0])))

    # remove Nones and Nans
    data = data.replace('None', 0)
    data.fillna(0, inplace=True)

    #
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data.select_dtypes(include=[np.number]).fillna(0)
    obj_cols = data.select_dtypes(include=['object']).columns
    data[obj_cols] = data.select_dtypes(include=['object']).fillna('NaN')

    # get ranges
    c1_min, c1_max = np.min(data['c1']), np.max(data['c1'])
    c2_min, c2_max = np.min(data['c2']), np.max(data['c2'])

    # preprocessing
    data_categorical = data.select_dtypes(exclude=["number"])
    custom_data = ['path', 'class', 'filename']
    custom_data_short = ['path', 'class', 'c1', 'c2', 'filename']
    additional_data = [col_name for col_name in data.columns.to_list() if col_name not in custom_data]
    custom_data.extend(additional_data)

    # prepare scene
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    fig = px.scatter(data, x='c1', y='c2', color='class', custom_data=custom_data_short,
                     color_discrete_map=handle_discrete_color_scale(data.copy(), 'class'))
    fig.update_traces(marker=dict(size=SCATTER_MARKER_SIZE))
    fig = update_scatter_fig_settings(fig)

    fig_initial = px.imshow(np.zeros((1024, 1024)), color_continuous_scale='gray')
    fig.update_layout(clickmode='event+select')

    app.layout = html.Div(children=[

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='choose_hue',
                    options=[{'label': i, 'value': i} for i in data.columns.to_list()],
                    value='class'
                )
            ], style={'margin': '1px'}),
        ]),

        html.Div([
            html.Div(
                dcc.Dropdown(
                    id='dropdown-filtration-col',
                    options=[{'label': i, 'value': i} for i in data_categorical.columns.to_list()],
                ), style={'width': '49%', 'display': 'inline-block', 'margin': '1px'}
            ),
            html.Div(
                dcc.Dropdown(
                    id='dropdown-filtration-val',
                    options=[{'label': i, 'value': i} for i in ["Select"]],
                ), id='div-dropdown-filtration-val',
                style={'width': '49%', 'display': 'inline-block', 'margin': '1px'}
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.RangeSlider(
                id='colorbar-range-slider',
                min=0,
                max=1,
                step=0.1,
                value=[0, 1]
            )
        ], id='div-colorbar_range-slider', style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Div(
                [dcc.Graph(id="umap", figure=fig, style={"height": "100%"})],
                style={"width": '49%', "margin": 0, 'display': 'inline-block', "height": "90vh", 'float': 'left'}
            ),

            html.Div([
                html.Div(
                    [dcc.Graph(id="image", figure=fig_initial, style={'height': "100%"})],
                    style={'height': "74%", "margin": 0, 'display': 'inline-block'}
                ),
                html.Button("Export",
                            id='save-dataset-button',
                            n_clicks=0,
                            style={'display': 'inline-block', 'float': 'left'}
                            ),
                html.Div(
                    id='tab1',
                    style={'height': "14%", "width": "45vw", "margin": 0,
                           'display': 'inline-block', 'overflow': 'scroll'}
                )
            ], style={"width": '49%', "margin": 0, 'display': 'inline-block', "height": "90vh", 'float': 'right'}
            ),

            ],
            className="row", style={"height": "90vh"})
    ], style={"width": "100vw", "height": "90vh"})

    states_table = {'clickData': None, 'selectedData': None}
    states_figure = {'clickData': None, 'selectedData': None}
    state_hue_range = {'hue_range_exists': False}

    @app.callback(
        Output('div-colorbar_range-slider', 'children'),
        Input('choose_hue', 'value'),
        Input('dropdown-filtration-col', 'value'),
        Input('dropdown-filtration-val', 'value')
    )
    def update_colorbar_range_slider(hue_choice, filtration_col, filtration_val):
        local_data = data.copy()
        if filtration_col is not None and filtration_val is not None:
            local_data = local_data.loc[local_data[filtration_col] == filtration_val]
        local_data, continuous_color_map = handle_continuous_color_scale(local_data.copy(), hue_choice)
        if continuous_color_map is not None:
            #marks = {np.log(j): str(j) for j in np.arange(np.min(local_data[hue_choice]), np.max(local_data[hue_choice]),
            #                                              step=0.1 * (np.max(local_data[hue_choice]) - np.min(local_data[hue_choice])))}
            # _local_data = np.log(local_data[hue_choice])
            _local_data = local_data.copy()
            # _local_data = np.clip(_local_data, 0, np.max(_local_data))
            marks = {j: str(np.round(np.exp(j), 3)) for j in
                     np.arange(np.min(_local_data), np.max(_local_data),
                               step=0.1 * (np.max(_local_data) - np.min(_local_data)))}
            range_slider = dcc.RangeSlider(
                id='colorbar-range-slider',
                min=np.min(_local_data),
                max=np.max(_local_data),
                step=0.1,
                marks = marks,
                value=[np.min(_local_data), np.max(_local_data)]
            )
            state_hue_range['hue_range_exists'] = True
        else:
            range_slider = dcc.RangeSlider(
                id='colorbar-range-slider'
            )
            state_hue_range['hue_range_exists'] = False
        return range_slider


    @app.callback(
        Output('image', 'figure'),
        Input('umap', 'clickData'),
        Input('umap', 'selectedData'))
    def display_click_data(clickData, selectedData):
        if clickData is not None and clickData != states_figure['clickData']:
            fig = make_subplots(rows=1, cols=1)
            # path = clickData['points'][0]['customdata'][0]
            filename = clickData['points'][0]['customdata'][4]
            path = glob.glob(opj(path_images, str(os.path.splitext(filename)[0]) + '.*'))[0]
            image = imread(path)
            image = np.rot90(np.rot90(image))
            fig.add_trace(px.imshow(image, color_continuous_scale='gray').data[0], row=1, col=1)
            layout = px.imshow(np.arange(0, 1, 0.1).reshape(2, 5), color_continuous_scale='gray').layout
            fig.layout.coloraxis = layout.coloraxis
            states_figure['clickData'] = clickData
        elif selectedData is not None and selectedData != states_figure['selectedData']:
            shapes = 3
            fig = make_subplots(rows=shapes, cols=shapes,
                    horizontal_spacing=0.01,
                    shared_yaxes=True)
            for enum, point in enumerate(selectedData['points']):
                # path = point['customdata'][0]
                filename = point['customdata'][4]
                path = glob.glob(opj(path_images, str(os.path.splitext(filename)[0]) + '.*'))[0]
                image = imread(path)
                image = np.rot90(np.rot90(image))
                rows = int(np.floor(enum / shapes))
                cols = int(enum - rows * shapes)
                # fig_image = px.imshow(image, color_continuous_scale='gray')
                fig.add_trace(px.imshow(image).data[0], row=rows + 1, col=cols + 1)
                if enum >= int(shapes ** 2 - 1):
                    break
            layout = px.imshow(np.arange(0, 1, 0.1).reshape(2,5), color_continuous_scale='gray').layout
            fig.layout.coloraxis = layout.coloraxis
            # fig.update_xaxes(**layout.xaxis.to_plotly_json())
            # fig.update_yaxes(**layout.yaxis.to_plotly_json())
            states_figure['selectedData'] = selectedData
        else:
            # image = np.zeros((512, 512))
            fig = make_subplots(1, 1)
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

    @app.callback(
        Output('tab1', 'children'),
        Input('umap', 'clickData'),
        Input('umap', 'selectedData')
    )
    def update_table(clickData, selectedData):
        if clickData is not None and clickData != states_table['clickData']:
            values = clickData['points'][0]['customdata']
            dict_table = [{custom_data_short[j]: values[j] for j in range(len(custom_data_short))}]
            states_table['clickData'] = clickData
        elif selectedData is not None and selectedData != states_table['selectedData']:
            values = selectedData['points']
            dict_table = []
            for value in values:
                value_customdata = value['customdata']
                dict_table.append({custom_data_short[j]: value_customdata[j] for j in range(len(custom_data_short))})
            states_table['selectedData'] = selectedData
        else:
            dict_table = [{custom_data_short[j]: "NAN" for j in range(len(custom_data_short))}]
        columns = [{"name": i, "id": i, } for i in (custom_data_short)]
        return dash_table.DataTable(data=dict_table, columns=columns, style_cell={
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            })

    @app.callback(
        Output('div-dropdown-filtration-val', 'children'),
        Input('dropdown-filtration-col', 'value'),
    )
    def update_dropdown_filtration_val(filtration_col):
        if filtration_col is not None:
            data_filtration_col = data[filtration_col]
            filtration_categories = list(np.unique(data_filtration_col))
            dcc_object = dcc.Dropdown(
                id='dropdown-filtration-val',
                options=[{'label': str(i), 'value': str(i)} for i in filtration_categories]
            )
        else:
            dcc_object = dcc.Dropdown(
                    id='dropdown-filtration-val',
                )
        return dcc_object

    @app.callback(
        Output('umap', 'figure'),
        Input('choose_hue', 'value'),
        Input('dropdown-filtration-col', 'value'),
        Input('dropdown-filtration-val', 'value'),
        Input('colorbar-range-slider', 'value')
    )
    def update_scatter_plot(hue_choice, filtration_col, filtration_val, hue_range):
        local_data = data.copy()
        if filtration_col is not None and filtration_val is not None:
            local_data = local_data.loc[local_data[filtration_col] == filtration_val]
        if state_hue_range['hue_range_exists'] and hue_range is not None:
            lower_range = np.exp(hue_range[0])
            upper_range = np.exp(hue_range[1])
            local_data = local_data.loc[(local_data[hue_choice] >= lower_range) & (local_data[hue_choice] <= upper_range)]
        # color scales initialization
        local_data, continuous_color_map = handle_continuous_color_scale(local_data, hue_choice)
        discrete_color_scale = handle_discrete_color_scale(local_data, hue_choice)
        # figure creation
        # fig = px.scatter(local_data, x="c1", y="c2", color=hue_choice, custom_data=custom_data_short)
        if continuous_color_map is None:
            fig = px.scatter(local_data, x="c1", y="c2", color=hue_choice,
                             hover_data=['class', 'filename'], custom_data=custom_data_short,
                             color_discrete_map=discrete_color_scale)
        else:
            fig = px.scatter(local_data, x="c1", y="c2", color=hue_choice,
                             hover_data=['class', 'filename'], custom_data=custom_data_short,
                             color_continuous_scale=continuous_color_map)
        fig.update_traces(marker=dict(size=SCATTER_MARKER_SIZE))
        fig = update_scatter_fig_settings(fig)
        fig.update_xaxes(range=[c1_min - 1, c1_max + 1])
        fig.update_yaxes(range=[c2_min - 1, c2_max + 1])
        if continuous_color_map is not None:
            tickvals = np.percentile(local_data[hue_choice].to_numpy(), [0, 40, 100])
            ticktext = np.percentile(local_data[hue_choice].to_numpy(), [0, 40, 100])
            ticktext = [str(t) for t in ticktext]
            fig.update_layout(coloraxis_colorbar=dict(
                title=hue_choice,
                tickvals=tickvals,
                ticktext=ticktext
            ))
        else:
            fig.for_each_trace(
                lambda trace:
                trace.update(name='<br>'.join(re.findall('.{1,10}', trace.name)))
                if len(trace.name) > 10 else trace.update(name=trace.name))
        # Update trace to enable point highlighting on click
        # Set clickmode to 'event+select' to enable point highlighting on click
        fig.update_layout(clickmode='event+select')
        return fig

    @app.callback(
        dash.dependencies.Output('save-dataset-button', 'value'),
        [dash.dependencies.Input('save-dataset-button', 'n_clicks')],
        [dash.dependencies.State('tab1', 'children')])
    def save_selection(n_clicks, children):
        if children is not None:
            df_new = pd.DataFrame()
            for row in children['props']['data']:
                idx = len(df_new)
                for key, value in row.items():
                    if type(value) is list:
                        df_new.loc[idx, key] = value[0]
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
            local_path_save = os.path.join(path_save, dt_string + "_dataset.csv")
            df_new.to_csv(local_path_save, index=False)
        return 'Event'

    app.run_server(debug=False)


def get_text(names, values):
    text = ""
    for name, value in zip(names, values):
        text += str(name) + " : " + str(value) + "\n"
    return text


def handle_discrete_color_scale(local_data, hue_choice):
    locs = local_data[hue_choice].to_numpy()
    classes = np.unique(locs)
    if len(classes) > 5:
        return None
    else:
        return dict(zip(classes, ['#1A356A', '#04E5C6', '#1986FD', '#7D83E6', '#D184D0'][::-1]))


def has_fractional_part(value):
    # Check if the value is a string and contains exactly one dot
    value = str(value)
    if value.count('.') == 1:
        # Attempt to convert the string to a float
        try:
            float(value)
            return True
        except ValueError:
            return False
    else:
        return False


def handle_continuous_color_scale(local_data, hue_choice):
    locs = local_data[hue_choice].to_numpy()
    # if len(np.unique(locs)) >= 100 or len(str(float(locs[0])).split('.')[-1]) >= 3:
    if has_fractional_part(locs[0]): # str(locs[0]).replace('.', '', 1).isdigit():
        # local_data.loc[:, hue_choice] = np.log10(local_data.loc[:, hue_choice])
        color_scale = 'Picnic'
    else:
        color_scale = None
    return local_data, color_scale


def update_scatter_fig_settings(fig):
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv(
        r"D:\Marek\Documents\!Programming\Python\InteractiveScatter\experiments\2024-10-14-michalHE\data\UMAP_MDLv1_MSE_KL 1.txt",
        index_col=[0], sep="\t")
    path_data = r"Z:\PROCESSED_DATA\HE\Previews_all_data"
    df = df.reset_index().rename(columns={'index': 'filename', "UMAP1": "c1", "UMAP2": "c2"})
    df['class'] = 'unknown'
    df['filename'] = df['filename'].apply(lambda x: x + '_he')
    interactive_scatter(df, path_data)
