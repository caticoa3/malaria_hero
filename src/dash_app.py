#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:36:06 2018
@author: Carlos A Ariza
"""

import base64
import numpy as np
import os
from urllib.parse import quote as urlquote
import dash
from PIL import Image, ImageOps
from flask import Flask, send_from_directory
from tflite_pred import tflite_img_class
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd

UPLOAD_FOLDER = '../flask/uploads/unknown/'
for setup_dir in [UPLOAD_FOLDER, '../../results/']:
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
def bar_plot(pred_df):
    pred_df = pred_df.sort_values('% Infected Cells')
    pred_df['Patient'] = pred_df['Patient'].astype(str)
    fig = px.bar(pred_df, y='Patient', x='% Infected Cells', orientation='h')
    fig.update_layout(yaxis_type='category')
    return fig


def resize_image(img: Image, desired_square_size=80):
    old_size = img.size
    ratio = float(desired_square_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img.thumbnail(new_size, Image.ANTIALIAS) #in-place operation
    return img


def pad_image(img: Image, desired_square_size=80):
    img_size = img.size
    delta_w = desired_square_size - img_size[0]
    delta_h = desired_square_size - img_size[1]
    padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
    img = ImageOps.expand(img, padding)
    return img


def image_montage(image_paths):
    im_array = []
    print('image_paths', image_paths)
    for p in image_paths:
        print(p)
        img = Image.open(p)
        img = resize_image(img)
        img = pad_image(img)
        # img = img.convert('RGB')
        img = np.array(img)
        print('converted to np array\n', np.sum(img, axis=(0,1)))
        print(img.shape)
        im_array.append(img)

    # image have different sizes, displaying with px.imshow only works when
    # images are the same size.
    im_array = np.stack(im_array)
    print('im_array.shape', im_array.shape)
    montage = px.imshow(im_array, facet_col=0, facet_col_wrap = 7)
    montage.for_each_annotation(lambda a: a.update(text=''))
    # hide subplot y-axis titles and x-axis titles
    for axis in montage.layout:
        if type(montage.layout[axis]) == go.layout.YAxis:
            montage.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')
        if type(montage.layout[axis]) == go.layout.XAxis:
            montage.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')

    return montage

pred_df = pd.read_csv('../primed_results/init_table.gz',
                      compression='gzip')
fig = bar_plot(pred_df)
mon = image_montage(['../images/malaria_hero.jpg'])

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
    html.P('''image file names should contain
           P<patient number>C<area of interest>cell_<cell number>.png'''),
    html.P('e.g. P143C4cell_8 or C5P320cell_90'),
    # https://github.com/plotly/dash-docs/blob/master/tutorial/examples/core_components/upload-image.py
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
    dcc.Graph(figure=fig, id='bar-plot'),
    html.Div(style={'height': '30%'},
             children = dcc.Graph(figure=mon, id='montage',
                                  style= {'height': 'inherit'})
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


def uploaded_files(directory=UPLOAD_FOLDER):
    """List the files in the upload directory."""
    file_paths = []
    for entry in os.scandir(directory):
        if os.path.isfile(entry.path):
            file_paths.append(entry.path)
    return file_paths


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = '/download/{}'.format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    [Output('summary-table', 'data'), Output('bar-plot', 'figure'), Output('montage', 'figure')],
    [Input('upload-data', 'filename'), Input('upload-data', 'contents'),
     Input('demo-button', 'n_clicks')],
)
def update_output(uploaded_filenames, uploaded_file_contents,
                  demo_button_clicks, pred_df=pred_df):

    if (demo_button_clicks > 0
        and uploaded_filenames is None
        and uploaded_file_contents is None):
        image_dir = '../flask/demo_images/unknown/'
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

    print(f'demo button at {demo_button_clicks} clicks')

    files = uploaded_files()
    print('files in upload folder', len(files))
    # load example results when page is first loaded
    if (len(files) == 0) and (demo_button_clicks == 0):
        pred_df = pd.read_csv('../primed_results/init_table.gz',
                              compression='gzip')
        return pred_df.to_dict(orient='records'), bar_plot(pred_df), image_montage(['../images/malaria_hero.jpg'])
#        return [html.Li('No files yet!')]
    else:
        files = uploaded_files(image_dir)
        action_df = tflite_img_class(
                             image_dir=image_dir,
                             prediction_csv='malaria.csv',
                             trained_model='../models/model.tflite',
                             )

        return action_df.to_dict(orient='records'), bar_plot(action_df), image_montage(files)


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
    print('reset button clicked')
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


# @app.callback(
    # Output('summary-table', 'selected_row_indices'),
    # [Input('graph-gapminder', 'clickData')],
    # [State('summary-table', 'selected_row_indices')])
# def update_selected_row_indices(clickData, selected_row_indices):
    # if clickData:
        # for point in clickData['points']:
            # if point['pointNumber'] in selected_row_indices:
                # selected_row_indices.remove(point['pointNumber'])
            # else:
                # selected_row_indices.append(point['pointNumber'])
    # return selected_row_indices


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


app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        <!-- Google Tag Manager -->
        <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
        new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
        j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
        'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
        })(window,document,'script','dataLayer','GTM-NTTNB9F');</script>
        <!-- End Google Tag Manager -->

        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script>(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
        (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
        m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
        })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
        ga('create', 'UA-165365115-1', 'auto');
        ga('send', 'pageview');
        </script>
        <!-- End Global Google Analytics -->

        {%metas%}
        <title>Malaria Hero</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <!-- Google Tag Manager (noscript) -->
        <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-NTTNB9F"
        height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
        <!-- End Google Tag Manager (noscript) -->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, port=5000, host='0.0.0.0')
