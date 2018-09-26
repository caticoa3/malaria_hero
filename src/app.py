#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:50:30 2018

@author: Carlos A Ariza, PhD
"""

import dash
from flask import Flask

server = Flask(__name__)

app = dash.Dash(server=server)

app.config.suppress_callback_exceptions = True