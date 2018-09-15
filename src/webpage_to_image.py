#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:04:19 2017

@author: carlos
"""
#Modified code from https://stackoverflow.com/questions/1197172/how-can-i-take-a-screenshot-image-of-a-website-using-python

#Trying to use the TOR network
#https://stackoverflow.com/questions/40208051/selenium-using-python-geckodriver-executable-needs-to-be-in-path

import os, errno
import numpy as np
from subprocess import Popen, PIPE
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import pandas as pd
from PIL import Image
from pathlib import Path
import time
from openpyxl import load_workbook
import signal
import threading
# 5 minutes from now

#Set maximum image size to convert to avoid DecompressionBombWarning when 
#images are larger than the default 89478485 pixels
Image.MAX_IMAGE_PIXELS = 10000000000 

abspath = lambda *p: os.path.abspath(os.path.join(*p))
ROOT = abspath(os.path.dirname(__file__))

def handler(signum, frame):
    print('Moving onto next URL')
    raise Exception('Timed out waiting for webpage capture')
    

def execute_command(command):
    result = Popen(command, shell=True, stdout=PIPE).stdout.read()
    if len(result) > 0 and not result.isspace():
        raise Exception(result)

browser = None
def get_browser(binary=None):
    global browser  
    # only one instance of a browser opens, remove global for multiple instances
    if not browser: 
        browser = webdriver.Firefox(firefox_binary=binary)
    return browser

driver = None
def do_screen_capturing(url, screen_path, width, height):
#    print("Capturing screen..")
    global driver
    if not driver:
    #    https://stackoverflow.com/questions/29463603/phantomjs-returning-empty-web-page-python-selenium
        driver = webdriver.PhantomJS(service_args=['--ignore-ssl-errors=true', 
                                                   '--ssl-protocol=TLSv1'])
        
        '''Need geckodriver to use firefox
        https://github.com/mozilla/geckodriver/releases'''
#        driver = webdriver.Firefox()
        # it save service log file in same directory
        # if you want to have log file stored else where
        # initialize the webdriver.PhantomJS() as
        # driver = webdriver.PhantomJS(service_log_path='/var/log/phantomjs/ghostdriver.log')
        '''Trying to use TOR browser
        geckodriver version 18.0 worked, versions 19.0, 19.1, and 20.0 did not
        get a WebDriverException: Reached error page: 
        about:neterror?e=proxyConnectFailure'''
    #    binary = '/Applications/TorBrowser.app/Contents/MacOS/firefox'
    #    firefox_binary = FirefoxBinary(binary)
    #    profile=webdriver.FirefoxProfile()
    #    profile.set_preference('network.proxy.type', 1)
    #    profile.set_preference('network.proxy.socks', '127.0.0.1')
    #    profile.set_preference('network.proxy.socks_port', 9150)
    #    driver = webdriver.Firefox(profile,binary)
    #    driver.get('http:\\www.yahoo.com')
    
    driver.set_script_timeout(60*5)
    if width and height:
        driver.set_window_size(width, height)
    driver.get(url)
    driver.save_screenshot(screen_path)


def do_crop(params):
    print("Croping captured image..")
    command = [
        'convert',
        params['screen_path'],
        '-crop', '%sx%s+0+0' % (params['width'], params['height']),
        params['crop_path']
    ]
    execute_command(' '.join(command))


def do_thumbnail(params):
    print("Generating thumbnail from croped captured image..")
    command = [
        'convert',
        params['crop_path'],
        '-filter', 'Lanczos',
        '-thumbnail', '%sx%s' % (params['width'], params['height']),
        params['thumbnail_path']
    ]
    execute_command(' '.join(command))


def get_screen_shot(**kwargs):
    url = kwargs['url']
    width = int(kwargs.get('width', 1024)) # screen width to capture
    height = int(kwargs.get('height', 768)) # screen height to capture
    filename = kwargs.get('filename', 'screen.png') # file name e.g. screen.png
    path = kwargs.get('path', ROOT) # directory path to store screen

    crop = kwargs.get('crop', False) # crop the captured screen
    crop_width = int(kwargs.get('crop_width', width)) # the width of crop screen
    crop_height = int(kwargs.get('crop_height', height)) # the height of crop screen
    crop_replace = kwargs.get('crop_replace', False) # does crop image replace original screen capture?

    thumbnail = kwargs.get('thumbnail', False) # generate thumbnail from screen, requires crop=True
    thumbnail_width = int(kwargs.get('thumbnail_width', width)) # the width of thumbnail
    thumbnail_height = int(kwargs.get('thumbnail_height', height)) # the height of thumbnail
    thumbnail_replace = kwargs.get('thumbnail_replace', False) # does thumbnail image replace crop image?

    screen_path = abspath(path, filename)
    crop_path = thumbnail_path = screen_path

    if thumbnail and not crop:
        raise Exception('Thumnail generation requires crop image, set crop=True')

    do_screen_capturing(url, screen_path, width, height)

    if crop:
        if not crop_replace:
            crop_path = abspath(path, 'crop_'+filename)
        params = {
            'width': crop_width, 'height': crop_height,
            'crop_path': crop_path, 'screen_path': screen_path}
        do_crop(params)

        if thumbnail:
            if not thumbnail_replace:
                thumbnail_path = abspath(path, 'thumbnail_'+filename)
            params = {
                'width': thumbnail_width, 'height': thumbnail_height,
                'thumbnail_path': thumbnail_path, 'crop_path': crop_path}
            do_thumbnail(params)
    return screen_path, crop_path, thumbnail_path

###############################################################################

def webcapture_process(df, key, writer, training1, dir_name, jpg_files):
    import datetime
    #create empty dataframe with the columns that will be added to record file
    df_out_0 = pd.DataFrame(columns=['captured','image_location'])
    df1 = df.copy()
    #add the columns 'captured' and 'image_location" to urlFile and concat
    df_out = pd.concat([df.copy(),df_out_0], axis=1)
    if isinstance(df1, pd.Series):
        df1 = df1.to_frame()
    for row in df1.itertuples(index=True, name='Pandas'):
        #Set random sleep periods to help prevent being blocked by website
#        time.sleep(np.random.uniform(0, 0.5)) 
        #'row' is a tupple containing all values in a row of the dataframe
        # get the label and the URL from each row in the data frame
        now = datetime.datetime.now()
        i = getattr(row, "Index")
        
        URL = getattr(row, "URL")
#        if training1:
        label = getattr(row, "label")
        print('key',key)
        print('label', label)

        #Prevent timeout and save to excel file every 30 rows or iterations
        if (i % 40 == 0) or ((i > df.shape[0] - 40) and (i%5==0)):
            print('Capture https://duckduckgo.com/ to prevent websocket timeout')
            get_screen_shot(
                url='https://duckduckgo.com/', filename= 'webcapture/drop.png',
                crop=False, crop_replace=False,
                thumbnail=False, thumbnail_replace=False,
                thumbnail_width=200, thumbnail_height=150)
            s = 'rm -p drop.png'
            os.system(s)
            df_out.to_excel(writer, sheet_name = key, index = False)
            writer.save()
        if pd.isnull(df1.loc[i,'URL']) or df1.loc[i,'URL'] == '':
            print('No URL in row number',str(i+2),'in worksheet named:',key
                  ,sep=' ')
            print('Moving on to next row')
            continue
        elif (pd.isnull(df1.loc[i,'label']) or df1.loc[i,'label'] == ''):
            label = 'unknown'
            df_out.loc[i,'label'] = 'unknown'
            print(URL, 'not labeled/classified; it has been labeled as unknown') 
        
        #Check if file exists. If not then it is created.
        if key!='predict':
            filename0 = os.path.join(dir_name, label,'_'.join([key,label,
                                                               str(i+2),'.png']))
            fn_jpg = filename0[:-3]+'jpg'
            dir_label = os.path.join(dir_name, label)
        else:
            filename0 = os.path.join(dir_name, '_'.join([label,str(i+2)]),'.png')
            fn_jpg = filename0[:-3]+'jpg'
            dir_label = dir_name
#            label = None
        
        if (fn_jpg not in jpg_files):
        #To determine if the webpage was previously captured we check for a jpg
            print('capturing image of', URL)
            
            if not os.path.exists(dir_label):
                try:
                    print('Made new directory:', dir_label)
                    os.makedirs(dir_label)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            
            signal.alarm(60*3) #sets/resets timeout in seconds every iteration
            try:
                #Plug in values into module
                screen_path, crop_path, thumbnail_path = get_screen_shot(
                url=URL, filename= filename0,
                crop=False, crop_replace=False,
                thumbnail=False, thumbnail_replace=False,
                thumbnail_width=200, thumbnail_height=150)
               
                #convert the image from .png to .jpg    
                file_size = os.path.getsize(filename0)
          
                im = Image.open(filename0)
                if file_size > 7000000:
                    signal.alarm(60*3)
                    print('Captured webpage image size larger than 7 MB.',
                          'Deleting bottom 3/4 of image.')
                    #https://stackoverflow.com/questions/5723400/how-to-crop-from-one-image-and-paste-into-another-with-pil
                    im = im.crop((0,0,im.size[0],im.size[1]/3))
                    im.load()
                
                rgb_im = im.convert('RGB')      
                rgb_im.save(fn_jpg)
                #delete the png version of the image
                os.remove(filename0)
                print(fn_jpg, 'web snapshot created')
                df_out.loc[i,'captured'] = now.strftime("%Y-%m-%d %H:%M:%S")
                df_out.loc[i,'image_location'] = fn_jpg
                time.sleep(np.random.uniform(0.4, 1.4))
                if Path(fn_jpg).is_file():
                    print('.jpg of webpage created.')
                    continue
            #if capturing the webpage timesout an exception is caught, and 
            #the next URL is processed
            except Exception as exc:
                print(exc)
                continue

        else:
            print(URL, 'was previously captured and saved in',dir_name)
            #a screenshot of the webpage was found; 
            #get the time the file was last modified
            timestamp = os.path.getmtime(fn_jpg)
            mod_time = datetime.datetime.fromtimestamp(timestamp).strftime(
                                                        "%Y-%m-%d %H:%M:%S")
        
            df_out.loc[i,'captured'] = mod_time
            df_out.loc[i,'image_location'] = fn_jpg

            print('The time of creation and save location of the .jpg',
                  'created from',URL, 'were added to the records.',sep=' ')
        
        if os.path.getsize(fn_jpg) == 0:
            #if image grab process was interupted empty files can be created
            #delete empty files with file name fn_jpg
            s = 'rm -p {}'.format(fn_jpg)
            os.system(s)
            print(fn_jpg, 'was deleted because it was empty.')
        if i > (df.shape[0] - 5):
#            print('df',df)
#            print('i', i)
            print(str(df.shape[0]-i-1), 'URLs remain.')
#            if i%10==0 and ((df.shape[0]-i-1)!=0):
#                print('Capture duckduckgo.com to prevent websocket timeout')
#                get_screen_shot(
#                    url='https://duckduckgo.com/', filename= 'webcapture/drop.png',
#                    crop=False, crop_replace=False,
#                    thumbnail=False, thumbnail_replace=False,
#                    thumbnail_width=200, thumbnail_height=150)
#                s = 'rm -p drop.png'
#                os.system(s)
            #save last batch of rows individualy; so that all data is saved
#            print('df_out',df_out)
            df_out.to_excel(writer, sheet_name = key, index = False)
            writer.save()
    
            #df_out.to_csv(csv_file, index = False)
#    print('full page path', screen_path,
#          'crop path', crop_path,
#          'thumbnail_path', thumbnail_path)

def jpg_of_webpage(csv_file = [], xlsFile = [], sheets= ['Sun','Sean','Peter'], 
    training1=False):
    # Based on above code the following creates an image of a full webpage
#    xlsFile =  [] #'../data/urlFile.xlsx' # 
#    csv_file = '../data/test_pipe.csv' #  [] # 
#    training1 = False # True
    #create data frames
    print('current_thread', threading.current_thread())
    assert threading.current_thread() == threading.main_thread()
    signal.signal(signal.SIGALRM, handler)
    
    if csv_file and xlsFile:
        print("No input file")
        exit
    
    #Create a directory to store results if it does not exist
    Path('../results').mkdir(parents=True, exist_ok=True)
    
    if training1:
        if not xlsFile:
            print('The spreadsheet containing labled URLs for algorithm training is needed.'
                  'Please input the excel formated spreadsheet containing labeled URLs.')
        else:
            dir_name = '../webcapture/dataset/'
            records_file = '../results/webcaptured.xlsx'
            #Import list of URL address that were manually input and labeled in a google sheet
            urlFile = pd.read_excel(xlsFile, sheet_name=sheets)
            #List of jpg files in webcaptured directory
            jpg_files = [str(x) for x in (Path(dir_name).glob('**/*.jpg'))]
    
            #load record file of webpages used for training that have been 
            #converted to jpg - if it exists.
    #        records = {}
            if Path(records_file).is_file():
            #https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting-data-using-pandas
                book = load_workbook(records_file)
                writer = pd.ExcelWriter(records_file, engine='openpyxl') 
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)                
    #            records = pd.read_excel(records_file, sheet_name=sheets)
    
            for key, df in urlFile.items():
            #for each worksheet in the urlFile containing urls, capture webpages
                webcapture_process(df, key, writer, training1, dir_name,
                                   jpg_files)
    if not training1:
        if not csv_file:
            print('No .csv file. Please input .csv file with URLs for ML labeling.')
        else:
            dir_name = '../webcapture/predict/'
            #List of jpg files in webcaptured directory
            jpg_files = [str(x) for x in (Path(dir_name).glob('**/*.jpg'))]
            records_file = '../results/predictions.xlsx'
            urlFile = pd.read_csv(csv_file)
            print('urlFile.columns',urlFile.columns)
            if 'label' not in urlFile.columns:
                #create empty column
                urlFile['label'] = None
            writer = pd.ExcelWriter(records_file, engine='openpyxl')
            webcapture_process(df = urlFile, key= 'predictions', writer=writer,
                               training1=training1, dir_name=dir_name, jpg_files=jpg_files)
    return dir_name

#jpg_of_webpage()
if __name__ == '__main__':
    import argparse
    # defined command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--worksheets",  # name on the CLI - drop the `--` for positional/required parameters
      nargs="*",  # 0 or more values expected => creates a list
      type=str,  # any type/callable can be used here
      default=['Sun','Sean','Peter'], # default if nothing is provided
    )
    CLI.add_argument(
    "url_file",  # name on the CLI - drop the `--` for positional/required parameters
      nargs=1,  # 0 or more values expected => creates a list
      type=str,  # any type/callable can be used here
      default='../data/urlFile.xlsx', # default if nothing is provided
    )    
    CLI.add_argument(
    "training",  # name on the CLI - drop the `--` for positional/required parameters
      nargs=1,  # 0 or more values expected => creates a list
      type=bool,  # any type/callable can be used here
      default=False, # default if nothing is provided
    )
    
    # parse the command line
    args = CLI.parse_args()
    # access CLI options
    print("worksheets: %r" % args.worksheets)
    print("url_file: %s" % args.url_file[0])
    print("training: %s" % args.training[0])
    
    jpg_of_webpage(xlsFile = args.url_file[0], sheets = args.worksheets, 
                   training1 = args.training[0])

#example command line use:
#python webpage_to_image.py url_file ../data/urlFile.xlsx training True --worksheets Sun Sean Peter 