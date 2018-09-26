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

app = dash.Dash()

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True
# app.css.config.serve_locally = True

UPLOAD_DIRECTORY = './tmp'
#if not os.path.exists(UPLOAD_DIRECTORY):
#    os.makedirs(UPLOAD_DIRECTORY)

#DF_GAPMINDER = pd.DataFrame()
DF_GAPMINDER = pd.read_csv('../results/predicted_malaria.csv', index_col =0)
print(DF_GAPMINDER.shape)

#def load_file():
#    DF_GAPMINDER = pd.read_csv('../results/predicted_malaria.csv', index_col =0)
#    print(DF_GAPMINDER.shape)

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
    html.H4('Parasite Alert Results'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image Files')
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
        multiple=True),

    #  html.Div(id='output-data-upload'),

    dt.DataTable(
        rows=DF_GAPMINDER.to_dict('records'),

        # optional - sets the order of columns
        columns=sorted(DF_GAPMINDER.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-gapminder'
    ),
    html.Div(id='selected-indexes'),
    dcc.Graph(
        id='graph-gapminder'
    ),
 ], className="container")

def parse_contents(contents, filename, date):
    print(type(contents), filename, date)
    file = contents
        #  if allowed_file(file.filename):
            #  filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_DIRECTORY, filename))
    return html.Div([
        html.H5(filename),
        #  html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
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
        if filename == '.keep':
            continue
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

@app.callback(
    #  Output("output-data-upload", "children"),
    Output("datatable-gapminder", "rows"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        print(files)
        #  return [html.Li(filename) for filename in files]
        return [{'Parasitized_probability': e, 'Predicted_label': 'moo',
                 'fn': f} for e, f in enumerate(files)]

# -- interactive table and graph creation
@app.callback(
    Output('datatable-gapminder', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('datatable-gapminder', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices


@app.callback(
    Output('graph-gapminder', 'figure'),
    [Input('datatable-gapminder', 'rows'),
     Input('datatable-gapminder', 'selected_row_indices')])
def update_figure(rows, selected_row_indices):
    dff = pd.DataFrame(rows)
    fig = plotly.tools.make_subplots(
        rows=2, cols=1, #rows=3
        subplot_titles=('Counts',''),
        shared_xaxes=True)
#    marker = {'color': ['#0074D9']*len(dff)}
    marker_parasite = {'color': ['#3399ff']*len(dff)}
    marker_uninfected = {'color': ['#ff9933']*len(dff)}
    for i in (selected_row_indices or []):
        marker_parasite['color'][i] = '#93bf2a'
        marker_uninfected['color'][i] = '#93bf2a'
#    trace1 = [go.Histogram(x = list((DF_GAPMINDER['Predicted_label']=='Parasitized')*1), opacity=0.75)]
    mask = DF_GAPMINDER['Predicted_label']=='Parasitized'
#    https://stackoverflow.com/questions/46750462/subplot-with-plotly-with-multiple-traces
#    a = DF_GAPMINDER.loc[mask,['Parasitized_probability']].values.round(3)
#    b = DF_GAPMINDER.loc[~mask,['Parasitized_probability']].values.round(3)
#    print(a.shape)

#    fig.append_trace(go.Histogram(x = a,
#                                  opacity = 0.75, name = 'Parasitized'),1,1)
#                                # histfunc='count',marker=marker, visible=True
#    fig.append_trace(go.Histogram(x = b,
#                                  opacity = 0.75, name = 'Uninfected'),1,1)

    c = list(DF_GAPMINDER.loc[mask,'Parasitized_probability'].values)
    d = list(DF_GAPMINDER.loc[~mask,'Parasitized_probability'].values)
    fig.append_trace({'x': c,
                      'type': 'histogram',
                      'opacity':0.75,
                      'marker': marker_parasite,
                      'name': 'Parasitized'
                      }, 1, 1)
    fig.append_trace({'x': d,
                      'type': 'histogram',
                      'opacity':0.75,
                      'marker': marker_uninfected,
                      'name': 'Uninfected'
                      }, 1, 1)
    fig.layout.update(go.Layout(barmode = 'overlay'))

    fig['layout']['showlegend'] = True
    fig['layout']['height'] = 800
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
    }
#    plotly.offline.plot(fig)
#    fig['layout']['yaxis3']['type'] = 'log'
    return fig




app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
