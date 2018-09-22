########################load libraries#########################################
import time
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import applications
from keras.optimizers import SGD
from load_data import load_resized_training_data, load_resized_validation_data
from sklearn.metrics import log_loss
import numpy as np
from densenet121_models import densenet121_model 
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from evaluation import plot_confusion_matrix
from sklearn.metrics import average_precision_score
#########################image characteristics#################################
img_rows=100 #dimensions of image
img_cols=100
channel = 3 #RGB
num_classes = 2 
batch_size = 1 #vary depending on the GPU
num_epoch = 60
###############################################################################
''' This code uses VGG-16 as a feature extractor'''

# create the base pre-trained model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

''' you can use the rest of the models like:
feature_model = applications.ResNet50((weights='imagenet', include_top=False, 
                                       input_shape=(224,224,3)) 
feature_model = applications.Xception((weights='imagenet', include_top=False,
                                       input_shape=(100,100,3))
For DenseNet, the main file densenet121_model is included to this repository.
The model can be used as :
feature_model = densenet121_model(img_rows=img_rows, img_cols=img_cols, 
                                  color_type=channel, num_classes=num_classes)
'''
#extract feature from an intermediate layer
base_model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output) 

''' you can use the rest of the models like this:
feature_model = Model(input=feature_model.input, output=feature_model.get_layer('res5c_branch2c').output) #for ResNet50
feature_model = Model(input=feature_model.input, output=feature_model.get_layer('block14_sepconv1').output) #for Xception'''

#get the model summary
base_model.summary()
###############################################################################
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer 
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
###############################################################################
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers to prevent large gradient updates 
# wrecking the learned weights
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
#fix the optimizer
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True) 
#compile the gpu model
model.compile(optimizer=sgd,
              loss='mse',
              metrics=['accuracy'])
###############################################################################
#load data for training
X_train, Y_train = load_resized_training_data(img_rows, img_cols)
X_valid, Y_valid = load_resized_validation_data(img_rows, img_cols)
#print the shape of the data
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
###############################################################################
t=time.time() #make a note of the time
#start training
print('-'*30)
print('Start Training the model...')
print('-'*30)
hist = model.fit(X_train, Y_train,
      batch_size=batch_size,
      epochs=num_epoch,
      shuffle=True,
      validation_data=None,
      verbose=1)

#print the history of the trained model
print(hist.history)

#compute the training time
print('Training time: %s' % (time.time()-t))
###############################################################################
# Make predictions on validation data
print('-'*30)
print('Predicting on validation data...')
print('-'*30)
y_pred = model.predict(X_valid, batch_size=batch_size, verbose=1)
###############################################################################
#compute the ROC-AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot ROC curves
plt.figure(figsize=(15,10), dpi=300)
lw = 1 
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.show()

# computhe the cross-entropy loss score
score = log_loss(Y_valid,y_pred)
print(score)

# compute the average precision score
prec_score = average_precision_score(Y_valid,y_pred)  
print(prec_score)

# compute the accuracy on validation data
Test_accuracy = accuracy_score(Y_valid.argmax(axis=-1),y_pred.argmax(axis=-1))
print("Test_Accuracy = ",Test_accuracy)

#declare target names
target_names = ['class 0(abnormal)', 'class 1(normal)'] #it should be normal 
# and abnormal for linux machines

#print classification report
print(classification_report(Y_valid.argmax(axis=-1),y_pred.argmax(axis=-1),
                            target_names=target_names))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_valid.argmax(axis=-1),y_pred.argmax(axis=-1))
np.set_printoptions(precision=4)

# Plot non-normalized confusion matrix
plt.figure(figsize=(15,10), dpi=300)
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
###############################################################################
# transfer it back
y_pred = np.argmax(y_pred, axis=1)
Y_valid = np.argmax(Y_valid, axis=1)
print(y_pred)
print(Y_valid)

#save the predicted and ground-truth labels.
np.savetxt('malaria_y_pred.csv',y_pred,fmt='%i',delimiter = ",")
np.savetxt('malaria_Y_test.csv',Y_valid,fmt='%i',delimiter = ",")
###############################################################################
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(15,10), dpi=300)
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(15,10), dpi=300)
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
###############################################################################