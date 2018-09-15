#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:41:44 2017

@author: carlos
"""

#Webpage interface for image-based webpage classification
#Ask for a file with URLs
import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_wtf import FlaskForm
from wtforms import IntegerField, RadioField
from werkzeug.utils import secure_filename
from web_img_class_API import web_img_class

UPLOAD_FOLDER = '../flask/uploads'
ALLOWED_EXTENSIONS = set(['csv','txt'])

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

@app.route('/', methods=['GET', 'POST'])
def img_classfier():
    
    form = UserInput()
    
    if form.validate_on_submit():
        if form.training.data: #form.training.data value is True or False
            print('Training with', form.sample_min.data, 'or greater samples for each class')    

    if request.method == 'POST':
        print('Uploaded file', request.files['file'].filename)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            feat_filename = ''.join([filename.split('.')[0],'_feat.csv'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            classify = web_img_class(csv_file1 = filename,\
                                     trained_model = 'trained_RF.sav',\
                                     features_file1= feat_filename,\
                                     image_dir1 = '../webcapture/prod_test/',
                                     min_samples1 = form.sample_min.data,\
                                     training1=form.training.data)
            return render_template('classify_out.html',\
                                   file_loc = url_for('uploaded_file',filename=filename),\
                                   features_file = feat_filename,\
                                   class_output = classify)

    return render_template('front_page.html', form=form)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

############################ How to use #######################################

# Make current working directory same as location of anom_api.py
# cd run
# In terminal enter:
# python anom_api.py
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
    
