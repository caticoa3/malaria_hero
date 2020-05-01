#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:36:06 2018
@author: Carlos A Ariza
"""

import base64
import os
from urllib.parse import quote as urlquote
import dash
from flask import Flask, send_from_directory
from web_img_class_API import web_img_class
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd

UPLOAD_FOLDER = '../flask/uploads'
for setup_dir in [UPLOAD_FOLDER, '../results/']:
    if not os.path.exists(setup_dir):
        os.makedirs(setup_dir)

server = Flask(__name__)
# Sets the maximum allowable upload size
server.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
server.config['SECRET_KEY'] = '$PZ5v3vXTGc3'
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True


def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


@server.route('/download/<path:path>')
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_FOLDER, path, as_attachment=True)


# Load example results
pred_df = pd.read_csv('../primed_results/init_table.csv', index_col=0)


DF_SIMPLE = pd.DataFrame({
    'x': ['A', 'B', 'C', 'D', 'E', 'F'],
    'y': [4, 3, 1, 2, 3, 6],
    'z': ['a', 'b', 'c', 'a', 'b', 'c']
})

ROWS = [
    {'a': 'AA', 'b': 1},
    {'a': 'AB', 'b': 2},
    {'a': 'BB', 'b': 3},
    {'a': 'BC', 'b': 4},
    {'a': 'CC', 'b': 5},
    {'a': 'CD', 'b': 6}
]

app.layout = html.Div([
    html.H4('Malaria Hero'),
    #    https://github.com/plotly/dash-docs/blob/master/tutorial/examples/core_components/upload-image.py
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': {
                    'b': '10px'}
        },
        # Allow multiple files to be uploaded
        multiple=True,
        ),
    #        html.H2('File List'),
    #        html.Ul(id='file-list'),
    html.Button(id='demo-button', n_clicks=0, children='Demo',
                style={
                        'margin': '10px',
                        'fontSize': 14
                        },),
    html.Button(id='reset-button', n_clicks=0, children='Reset',
                style={
                        'margin': '10px',
                        'fontSize': 14
                        },),
    #        html.Div(id='output-image-upload'),

    dt.DataTable(
        data=pred_df.to_dict('records'),

        # optional - sets the order of columns
        columns=[{'name': i, 'id': i} for i in pred_df.columns],

        filter_action='native',
        sort_action='native',
        id='summary-table'
    ),
    #    html.Div(id='selected-indexes'),
    #    dcc.Graph(
    #        id='graph-gapminder'
    #    ),
    #    html.H1('UMAP'),
    #    html.Div(id='bokeh_script',
    #             children = 'placeholder for plot')
 ], className='container')


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode('utf8').split(b';base64,')[1]
    with open(os.path.join(UPLOAD_FOLDER, name), 'wb') as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = '/download/{}'.format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output('summary-table', 'data'),
    [Input('upload-data', 'filename'), Input('upload-data', 'contents'),
     Input('demo-button', 'n_clicks')],
)
def update_output(uploaded_filenames, uploaded_file_contents, button_clicks,
                  pred_df=pred_df):

    if button_clicks > 0:
        image_dir = '../flask/demo_images'
    else:
        image_dir = UPLOAD_FOLDER

    '''Clear folders before saving new content'''
    for folder in [UPLOAD_FOLDER, '../results/']:
        clear_folder(folder)
#    pd.DataFrame().to_csv('../results/prod_test.csv')

    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    print(f'demo button at {button_clicks} clicks')

    files = uploaded_files()
    print('files in upload folder', len(files))
    # load example results when page is first loaded
    if (len(files) == 0) and (button_clicks == 0):
        pred_df = pd.read_csv('../primed_results/init_table.csv', index_col=0)
        return pred_df.to_dict(orient='records')
#        return [html.Li('No files yet!')]
    else:
        action_df = web_img_class(
                             image_dir=image_dir,
                             prediction_csv='malaria.csv',
                             trained_model='../models/trained_log_reg.sav',
                             features_file1='../results/prod_test_feat.csv',
                             min_samples1=0,
                             training1=False)

        return action_df.to_dict(orient='records')


@app.callback(
    Output('reset-button', 'style'),
    [Input('demo-button', 'n_clicks')],
    )
def color_demo_button(clicks):
    if clicks > 0:
        return {
                'margin': '10px',
                'backgroundColor': '#F45555',
                'fontSize': 14
                }
    else:
        return {
                'margin': '10px',
                'fontSize': 14
                }


# reset "demo button" after "reset button" is clicked
@app.callback(
    Output('demo-button', 'n_clicks'),
    [Input('reset-button', 'n_clicks')],
    state=[State('demo-button', 'value')]
    )
def reset_demo_button(n_clicks, input_value):
    for folder in ['../flask/uploads', '../results/']:
        clear_folder(folder)
    print('upload and result folders cleared')
    return 0


@app.callback(
    [Output('upload-data', 'filename'),
     Output('upload-data', 'contents')],
    [Input('reset-button', 'n_clicks')],
    state=[State('demo-button', 'value')]
    )
def clear_upload_filename(n_clicks, input_value):
    print(f'reset button clicked')
    return None, None


#    [html.Li(file_download_link(filename)) for filename in files]
# -- bokeh plot update
# @app.callback(
#    Output('bokeh_script', 'children'),
#    [Input('summary-table', 'rows')],
# )
# def bokeh_update(rows):
#        bn_df = pd.read_csv('../results/prod_test_feat.csv', index_col=0)
#        pred_df = pd.DataFrame(rows)
#        if pred_df.shape[0] > 3:
#            #http://biobits.org/bokeh-flask.html
#
#            html = umap_bokeh(bn_feat = bn_df,
#                            pred_df = pred_df,
#                            image_folder =UPLOAD_FOLDER)
#        else:
#            html = 'Plotting error: At least 4 cells are need for plots.'
# #            div = ''
#        return html #script
#
# -- interactive table and graph creation


@app.callback(
    Output('summary-table', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('summary-table', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices


# @app.callback(
#    Output('graph-gapminder', 'figure'),
#    [Input('summary-table', 'rows'),
#     Input('summary-table', 'selected_row_indices')])
# def update_figure(rows, selected_row_indices):
#    dff = pd.DataFrame(rows)
#    fig = plotly.tools.make_subplots(
#        rows=2, cols=1, #rows=3
#        subplot_titles=('Counts',''),
#        shared_xaxes=True)
#
#    marker_parasite = {'color': ['#3399ff']*len(dff)}
#    marker_uninfected = {'color': ['#ff9933']*len(dff)}
#
#    for i in (selected_row_indices or []):
#        marker_parasite['color'][i] = '#93bf2a'
#        marker_uninfected['color'][i] = '#93bf2a'
#
#    mask = dff['Predicted_label']=='Parasitized'
#
#    c = list(dff.loc[mask,'Parasitized_probability'].values)
#    d = list(dff.loc[~mask,'Parasitized_probability'].values)
#    fig.append_trace({'x': c,
#                      'type': 'histogram',
#                      'opacity':0.75,
#                      'marker': marker_parasite,
#                      'name': 'Parasitized'
#                      }, 1, 1)
#    fig.append_trace({'x': d,
#                      'type': 'histogram',
#                      'opacity':0.75,
#                      'marker': marker_uninfected,
#                      'name': 'Uninfected'
#                      }, 1, 1)
#    fig.layout.update(go.Layout(barmode = 'overlay'))
#
#    fig['layout']['showlegend'] = True
#    fig['layout']['height'] = 800
#    fig['layout']['margin'] = {
#        'l': 40,
#        'r': 10,
#        't': 60,
#        'b': 200
#    }
# plotly.offline.plot(fig)
# #    fig['layout']['yaxis3']['type'] = 'log'
#    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=5000, host='0.0.0.0')
