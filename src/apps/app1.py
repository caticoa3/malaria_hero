#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:36:06 2018

@author: Carlos A Ariza, PhD
"""

import base64
import os
from urllib.parse import quote as urlquote
import dash
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_wtf import FlaskForm
from wtforms import IntegerField, RadioField
from werkzeug.utils import secure_filename
from web_img_class_API import web_img_class, make_tree
from umap_plots import umap_bokeh
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import datetime

from app import app, server

UPLOAD_DIRECTORY = '../flask/uploads'
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

#server = Flask(__name__)

#app = dash.Dash(server=server)

#app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

#http://docs.sherlockml.com/user-guide/apps/examples/dash_file_upload_download.html
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


layout = html.Div([
    html.H4('Parasite Alert Data Upload'),
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
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True,
        ),
        html.H2("File List"),
        html.Ul(id="file-list"),
     ])

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))
        
def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    '''Clear folders before saving new content'''
    for folder in [UPLOAD_DIRECTORY, '../results/']:
        clear_folder(folder)
    
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        classify, pred_df, bn_df = web_img_class(image_dir = UPLOAD_DIRECTORY,\
                                 prediction_csv = 'malaria.csv',\
                                 trained_model = '../models/trained_log_reg.sav',\
                                 features_file1= '../results/prod_test_feat.csv',\
                                 min_samples1 = 0,\
                                 training1= False)
        
        return [html.Li(file_download_link(filename)) for filename in files]

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

#if __name__ == '__main__':
#    app.run_server(debug=True, port=8888)