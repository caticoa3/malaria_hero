#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:25:29 2018

@author: Carlos Atico Ariza, PhD
"""

def print_directory_contents(sPath):
    print('sPath', sPath)
    import os                                       
    for sChild in os.listdir(sPath):                
        sChildPath = os.path.join(sPath,sChild)
        if os.path.isdir(sChildPath):
            print('directory')
            print_directory_contents(sChildPath)
        else:
            print(sChildPath)

print_directory_contents('..')
