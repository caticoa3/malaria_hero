#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:53:35 2018

@author: Carlos A Ariza, PhD
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app1, dash_table_1


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
         return app1.layout
    elif pathname == '/results':
         return dash_table_1.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)