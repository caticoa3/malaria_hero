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
from plotly.subplots import make_subplots
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
    fig = px.bar(pred_df, y='Patient', x='% Infected Cells',
                 orientation='h')
    fig.update_layout(yaxis_type='category')
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='gray')
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    # print(fig)
    return fig


def resize_image(img: Image, desired_square_size=60):
    old_size = img.size
    ratio = float(desired_square_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img.thumbnail(new_size, Image.ANTIALIAS) #in-place operation
    return img


def pad_image(img: Image, desired_square_size=60, prediction=0):
    img_size = img.size
    delta_w = desired_square_size - img_size[0]
    delta_h = desired_square_size - img_size[1]
    padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
    if prediction == 0: # infected
        border_color = '#3399ff'
    elif prediction == 1: # uninfected
        border_color = '#ff9933'
    img = ImageOps.expand(img, padding, border_color)
    return img


def image_montage(image_dir, details, summary):
    patients = list(details.Patient.unique())
    print(f'{patients=}')
    max_image_counts = int(details.groupby('Patient').agg('nunique')['fn'].max())
    print(f'{max_image_counts=}')
    number_of_patients = len(patients)
    montage = make_subplots(number_of_patients, max_image_counts,
                            vertical_spacing=0.05)

    summary = summary.set_index('Patient')
    for p_i, patient in enumerate(patients):
        patient_filter = (details.Patient == patient)
        patient_df = details.loc[patient_filter, ['fn','Predicted_label']]
        infection_rate = summary.loc[patient, '% Infected Cells']
        montage.add_annotation(text=f"Patient {patient}: {infection_rate} % infected",
                               xref="paper",
                               yref="paper", x=0,
                               y=1.05 - p_i*(1/number_of_patients + 0.05), #include verticle spacing
                               showarrow=False)
        for i_i, row in patient_df.reset_index().iterrows():
            im_p = row['fn']
            pred = row['Predicted_label']
            img_i = Image.open(image_dir + im_p.split('/')[-1]).copy()
            img_i = img_i.convert('RGB')
            img_i = resize_image(img_i, 50)
            img_i = pad_image(img_i, 55, pred)
            img_i = np.array(img_i)
            montage.add_trace(go.Image(z=img_i), p_i + 1, i_i + 1)

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
           P<patient number>C<area on microscope slide>cell_<cell number>.png'''),
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
    html.H5('Color-Coded Classified Cells: Parasitzed cells framed in blue',
            id='montage_heading'),
    dcc.Graph(id= 'montage')
 ]
 , className='container')


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


def empty_image_montage():
    montage = make_subplots(1,1, print_grid=False)
    montage.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    montage.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    montage.update_yaxes(showgrid=False, zeroline=False)
    montage.update_xaxes(showgrid=False, zeroline=False)
    for axis in ['xaxis', 'yaxis']:
        montage.layout[axis].tickfont = dict(color = 'rgba(0,0,0,0)')
    return montage


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = '/download/{}'.format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    [Output('summary-table', 'data'), Output('bar-plot', 'figure'),
     Output('montage', 'figure'), Output('montage_heading','style')],
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
        pred_df = pd.read_csv('../primed_results/init_table.gz', compression='gzip')
        return (pred_df.to_dict(orient='records'), bar_plot(pred_df),
                empty_image_montage(), {'display':'none'})

    else:
        files = uploaded_files(image_dir)
        action_df, details = tflite_img_class(
                                        image_dir=image_dir,
                                        prediction_csv='malaria.csv',
                                        trained_model='../models/model.tflite'
                                        )

        return (action_df.to_dict(orient='records'), bar_plot(action_df),
                image_montage(image_dir, details, action_df),
                {'dipslay':'block'})


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
