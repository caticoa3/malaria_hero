#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 2018

@author: carlos atico ariza
"""

#Webpage interface for image-based webpage classification
#Ask for a directory with png images
import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_wtf import FlaskForm
from wtforms import IntegerField, RadioField
from werkzeug.utils import secure_filename
from web_img_class_API import web_img_class, make_tree
from umap_plots import umap_bokeh

UPLOAD_FOLDER = '../flask/uploads'
ALLOWED_EXTENSIONS = set(['csv','txt','png'])

app =  Flask(__name__)
app.config['SECRET_KEY'] = '$PZ5v3vXTGc3'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #Sets the maximum allowable upload size

class UserInput(FlaskForm): #Instatiates the class UserInput inherited from FlaskForm
    training = RadioField('Training', choices=[('False','Predict'),('True','Train')],default='False')
    sample_min = IntegerField(label='Minimum number of samples each label should have when training',default=0)
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# future work: https://flask-dropzone.readthedocs.io/en/latest/index.html
#using dropzone-js-resources
def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

@app.route('/', methods=['GET', 'POST'])
def img_classfier():
        
    form = UserInput()
    
    if form.validate_on_submit():
        for folder in [UPLOAD_FOLDER, '../results/']:
            clear_folder(folder)

#        if form.training.data: #form.training.data value is True or False
#            print('Training with', form.sample_min.data, 'or greater samples for each class')    

#    if request.method == 'POST':
        files = request.files.getlist('file', None)
        print('Uploaded files', files) #filename)
        # check if the post request has the file part
#        if 'file' not in files:
#            flash('No file part')
#            return redirect(request.url)
        # if user does not select file, browser also
        # submit a empty part without filename
        if files == '':
            flash('No selected file')
            return redirect(request.url)
        for file in files:
            print('\n',file)
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        classify, pred_df, bn_df = web_img_class(image_dir = UPLOAD_FOLDER,\
                                 prediction_csv = 'malaria.csv',\
                                 trained_model = '../models/trained_log_reg.sav',\
                                 features_file1= '../results/prod_test_feat.csv',\
                                 min_samples1 = form.sample_min.data,\
                                 training1= False)
        
        if bn_df.shape[0] > 3:
            #http://biobits.org/bokeh-flask.html
        
            script, div = umap_bokeh(bn_feat = bn_df,
                            pred_df = pred_df,
                            image_folder =UPLOAD_FOLDER)
        else:
            script= 'Plotting error: At least 4 cells are need for plots.'
            div = ''
        return render_template('classify_out.html',\
                               file_loc = url_for('dirtree'),\
                               features_file = '../results/prod_test_feat.csv',\
                               class_output = classify, script=script, div=div)

    return render_template('front_page.html', form=form)
        

@app.route('/uploads/')
def dirtree():
    path = '../flask/uploads/'
    return render_template('dirtree.html', tree=make_tree(path))

@app.route('/uploads/<filename>')
def uploaded_file(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
############################ How to use #######################################

# Make current working directory same as location of web_img_class_API.py
# cd src
# In terminal enter:
# python flask_app.py
# In another terminal run the following example:
# curl -F "file=@../data/generated_data.csv" http://0.0.0.0:5000

#If socket already in use error raised. Find process ID to kill socket.
# Run in terminal:
# ps -fA | grep python
# or another option to find process id:
# sudo lsof -i:5000
#then kill <process id>
# if that doesnt kill it try
# kill -s KILL <pid> or kill -9 <pid>

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')    
    
