
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:02:39 2022

@author: acous
"""
import math
import sys
import os
import timeit
import gc
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import set_session
from sklearn.model_selection import KFold


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.is_built_with_cuda()

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

#Defaults
CV_fold = 5
label = 'phenotype_value'
model_nn = False
out_dir = 'deepl_project'
max_node = 80
max_layers = 2
learning = True

for i in range (1,len(sys.argv),2):
    if sys.argv[i] == "-x":
        X_file = sys.argv[i+1]
    elif sys.argv[i] == "-y":
        Y_file = sys.argv[i+1]
    elif sys.argv[i] == "-cv":
        CV_fold = int(sys.argv[i+1])
    elif sys.argv[i] == "-label":
        label = sys.argv[i+1]
    elif sys.argv[i] == "-o":
        out_dir = int(sys.argv[i+1])
    elif sys.argv[i] == "-m":
         model_nn = bool(sys.argv[i+1])
    elif sys.argv[i] == "mf":
        model_file = sys.argv[i+1]
    else:
        print('unknown option ' + str(sys.argv[i]))
        quit()

X = pd.read_csv(X_file,index_col=0)
pheno = pd.read_csv(Y_file,index_col=0)

y = pheno[[label]]


base = os.path.basename(X_file)
pheno_name = os.path.splitext(base)[0]

start_time = timeit.default_timer()

test = np.asarray(y)
if np.isin(test, [0, 1]).all() == True:
    print('Assuming Binary classification')
    method1 = 'Binary_classification'
    loss_type = 'binary_crossentropy'
    metric_type = 'accuracy'
else:
    print('Assuming regression problem')
    method1 = 'Regression'
    loss_type = 'mean_absolute_error'
    metric_type = 'mean_absolute_error'
    if round(float(y.mean()),3)<0:
        y=y*100

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=0)

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, max_layers)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=30,
                                            max_value=max_node,
                                            step=20),
                               activation='relu'))
    #if hp.Boolean("dropout"):
    #   model.add(layers.Dropout(rate=0.25))
    if method1 == 'Regression':
        model.add(layers.Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3])),
            loss=loss_type,
            metrics=[metric_type])
    else:
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3])),
            loss=loss_type,
            metrics=[metric_type])       
    return model


def cross_validation(hyps, X_train, y_train, X_test, y_test):
    model1 = tuner.hypermodel.build(hyps)
    
    model1.fit(X_train, y_train, epochs=best_epoch,validation_data=(X_test, y_test), verbose=0)
    bla = model1.predict(X_test)
    bla = bla.flatten()
    y_sub=np.asarray(y_test)
    y_sub=y_sub.flatten()
    z = pd.Series(bla)
    w = pd.Series(y_sub)
    corr = z.corr(w)
    del model1, hyps, X_train, X_test, y_train, y_test, bla, y_sub, z, w
    gc.collect()
    
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)
    
    
    return corr

        

def layer_count(something):
    n= len(something)
    b= 1
    layer_count = []
    for layer in something:
        if b <= n-1:
            layer_count.append(layer.output_shape[1])
        b+=1
    print('The final architechture is: '+str(layer_count))
    return layer_count

    
if model_nn == False:
    tuner = kt.Hyperband(build_model,
                            objective='val_'+metric_type,
                            max_epochs=25,
                            factor=4,
                            directory=out_dir,
                            project_name=pheno_name)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train,
                     epochs=15,
                     validation_data=(X_test, y_test), callbacks=[stop_early], verbose=0)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    lr = best_hps.get('learning_rate')
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)

    if method1 == 'Regression':
        val_acc_per_epoch = history.history['val_'+metric_type]
        best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    else:
        val_acc_per_epoch = history.history['val_'+metric_type]
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    del model, history
    gc.collect()
    
    seed = 7
    np.random.seed(seed)
    cvscores = []
    kfold = KFold(n_splits=CV_fold, shuffle=True, random_state=seed)
    for train, test in kfold.split(X, y):
        pearson_cc = cross_validation(best_hps, X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test])
        cvscores.append(pearson_cc)
    cvscores = [item for item in cvscores if not(math.isnan(item)) == True]
 
    
    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test), verbose=0)
    hypermodel.save(out_dir+'/'+pheno_name+'_model.h5')

    layer_nn = layer_count(hypermodel.layers)
    
    comp_time = int(round(timeit.default_timer() - start_time,0))
    DateTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    outfile_name = 'Outputs/'+pheno_name+'.txt'
    if not os.path.isfile(outfile_name):
        out2 = open(outfile_name,'w')
        out2.write('Phenotype\tDateTime\tCompTime\tCV_fold\tArchit\tModel_goal\tEpochs\tlearning_rate\tAccuracy\tStandard_deviation\n' )
        out2.write('%s\t%s\t%i\t%i\t%s\t%s\t%i\t%0.5f\t%0.5f\t%0.5f\n' % (
            pheno_name, DateTime, comp_time, int(CV_fold), str(layer_nn), method1, int(best_epoch), lr, 
            round(np.mean(cvscores),4), round(np.std(cvscores),4)))
        out2.close()
 
else:
    hypermodel = tf.keras.models.load_model(model_file)
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))


    
print(hypermodel.summary())
print(cvscores)
print(np.mean(cvscores))
print(np.std(cvscores))
print('For phenotype: '+pheno_name)


del hypermodel, X, y, X_train, X_test, y_train, y_test, comp_time, DateTime
gc.collect()
