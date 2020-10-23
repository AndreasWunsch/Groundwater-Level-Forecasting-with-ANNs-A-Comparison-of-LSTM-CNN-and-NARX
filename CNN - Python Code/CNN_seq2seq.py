# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:53:39 2020
updated on on Thu Oct 15 17:15:45 2020
@author: Andreas Wunsch
"""

#reproducability
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs #needed for: load existing optimizer states
import os
import glob
import pandas as pd
import keras as ks
import datetime
from scipy import stats
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')


def load_RM_GW_and_HYRAS_Data(i):
    pathGW = "./GWData"
    pathHYRAS = "./MeteoData"
    pathconnect = "/"
    GWData_list = glob.glob(pathGW+pathconnect+'*.csv');
    
    Well_ID = GWData_list[i]
    Well_ID = Well_ID.replace(pathGW+'\\', '')
    Well_ID = Well_ID.replace('_GWdata.csv', '')
    
    GWData = pd.read_csv(pathGW+pathconnect+Well_ID+'_GWdata.csv', 
                         parse_dates=['Date'],index_col=0, dayfirst = True, 
                         decimal = '.', sep=',')
    HYRASData = pd.read_csv(pathHYRAS+pathconnect+Well_ID+'_HYRASdata.csv',
                            parse_dates=['Date'],index_col=0, dayfirst = True,
                            decimal = '.', sep=',')
    data = pd.merge(GWData, HYRASData, how='inner', left_index = True, right_index = True)
    
    
    return data, Well_ID

def split_data(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["test_start"])] #separate testdata
    
    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset))+1:round(0.9 * len(dataset))]
    StopData_ext = dataset[round(0.8 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:round(0.9 * len(dataset))] #extend data according to dealys/sequence length
    OptData = dataset[round(0.9 * len(dataset))+1:]
    OptData_ext = dataset[round(0.9 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:] #extend data according to dealys/sequence length
    
    TestData = data[(data.index >= GLOBAL_SETTINGS["test_start"]) & (data.index <= GLOBAL_SETTINGS["test_end"])] #Testdaten entsprechend dem angegebenen Testzeitraum
    TestData_ext = pd.concat([dataset.iloc[-GLOBAL_SETTINGS["seq_length"]:], TestData], axis=0) # extend Testdata to be able to fill sequence later                                              

    return TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext

def extract_PI1_testdata(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["test_start"])] #separate testdata
    start = dataset.shape[0]-1
    Testdata_PI1 = data['GWL'][start:-1]
    return Testdata_PI1

# split a multivariate sequence into samples
def split_sequences(data, GLOBAL_SETTINGS):
	X, y = list(), list()
	for i in range(len(data)):
		# find the end of this pattern
		end_ix = i + GLOBAL_SETTINGS["seq_length"]
		out_end_ix = end_ix + GLOBAL_SETTINGS["output_seq_length"]
		# check if we are beyond the dataset
		if out_end_ix > len(data):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = data[i:end_ix, 1:], data[end_ix:out_end_ix, 0]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train,X_stop, Y_stop):
    # define model
    seed(ini)
    tf.random.set_seed(ini)
    model = ks.models.Sequential()
    model.add(ks.layers.convolutional.Conv1D(filters=GLOBAL_SETTINGS["filters"], kernel_size=GLOBAL_SETTINGS["kernel_size"], 
                                                padding='same', activation='relu', 
                                                input_shape=(GLOBAL_SETTINGS["seq_length"], X_train.shape[2])))
    model.add(ks.layers.convolutional.MaxPooling1D(padding='same'))
    
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu'))
    model.add(ks.layers.Dense(GLOBAL_SETTINGS["output_seq_length"], activation='linear'))

    optimizer = ks.optimizers.Adam(lr=GLOBAL_SETTINGS["learning_rate"], epsilon=10E-3, clipnorm=GLOBAL_SETTINGS["clip_norm"], clipvalue=GLOBAL_SETTINGS["clip_value"])
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    # early stopping
    es = ks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    
    # fit network
    model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop), epochs=GLOBAL_SETTINGS["epochs"], verbose=1,
                        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es])
    
    return model

# this is the optimizer function but checks only if paramters are integers and calls real optimizer function
def bayesOpt_function(pp,densesize, seqlength, batchsize, filters, rH, T, Tsin):
    #basically conversion to rectangular function
    densesize_int = int(densesize)
    seqlength_int = int(seqlength)
    batchsize_int = int(batchsize)
    filters_int = int(filters)
    
    pp = int(pp)
    
    rH = int(round(rH))
    T = int(round(T)) 
    Tsin = int(round(Tsin)) 
    
    return bayesOpt_function_with_discrete_params(pp, densesize_int, seqlength_int, batchsize_int, filters_int, rH, T, Tsin)

#this is the real optimizer function
def bayesOpt_function_with_discrete_params(pp,densesize_int, seqlength_int, batchsize_int, filters_int, rH, T, Tsin):
    
    assert type(densesize_int) == int
    assert type(seqlength_int) == int
    assert type(batchsize_int) == int
    assert type(filters_int) == int
    assert type(rH) == int
    assert type(T) == int
    assert type(Tsin) == int
    
    # fixed settings for all experiments
    GLOBAL_SETTINGS = {
        'pp': pp,
        'batch_size': batchsize_int, #16-128
        'kernel_size': 3, #ungerade!
        'dense_size': densesize_int, 
        'filters': filters_int, 
        'seq_length': seqlength_int,
        'output_seq_length': 12,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 30,
        'learning_rate': 1e-3,
        'test_start': pd.to_datetime('02012012', format='%d%m%Y'),
        'test_end': pd.to_datetime('28122015', format='%d%m%Y')
    }

    ## load data
    data, Well_ID = load_RM_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    
    # inputs
    if rH == 0:
        data = data.drop(columns='rH')
    if T == 0:
        data = data.drop(columns='T')
    if Tsin == 0:
        data = data.drop(columns='Tsin')
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = StandardScaler()
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    # #sequence data
    X_train, Y_train = split_sequences(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = split_sequences(StopData_ext_n.values, GLOBAL_SETTINGS)
    X_opt, Y_opt = split_sequences(OptData_ext_n.values, GLOBAL_SETTINGS)
    X_test, Y_test= split_sequences(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with idifferent initializations
    inimax = 5
    forecast_idx = OptData_ext_n.index.day < 8
    forecast_idx = forecast_idx[GLOBAL_SETTINGS["seq_length"]:len(OptData_ext_n)-GLOBAL_SETTINGS["output_seq_length"]+1]
    X_opt_reduced = X_opt[forecast_idx]
    Y_opt_reduced = Y_opt[forecast_idx]
    optresults_members = np.zeros((len(OptData_n), len(X_opt_reduced), inimax))
    optresults_members[:] = np.nan
    for ini in range(inimax):
        print("BayesOpt-Iteration {} - ini-Ensemblemember {}".format(len(optimizer.res)+1, ini+1))
        
        model = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  

        idx = 0
        for i in range(0,len(X_opt_reduced)):
            opt_sim_n = model.predict(X_opt_reduced[i,:,:].reshape(1,X_opt_reduced.shape[1],X_opt_reduced.shape[2]))
            opt_sim = scaler_gwl.inverse_transform(opt_sim_n)
            
            while forecast_idx[idx] == False:
                idx = idx + 1

            optresults_members[idx:idx+GLOBAL_SETTINGS["output_seq_length"], i, ini] = opt_sim.reshape(-1,)
            idx = idx+1

    opt_sim_median = np.nanmedian(optresults_members,axis = 2)

    # get scores
    errors = np.zeros((opt_sim_median.shape[1],6))
    errors[:] = np.nan
    for i in range(0,opt_sim_median.shape[1]):
        sim = np.asarray(opt_sim_median[:,i].reshape(-1,1))
        sim = sim[~np.isnan(sim)].reshape(-1,1)
        obs = np.asarray(scaler_gwl.inverse_transform(Y_opt_reduced[i,:].reshape(-1,1)))
        
        err = sim-obs
        err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
        err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))
        errors[i,0] = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2))) #NSE
        r = stats.linregress(sim[:,0], obs[:,0])
        errors[i,1] = r.rvalue ** 2 #R2
        errors[i,2] =  np.sqrt(np.mean(err ** 2)) #RMSE
        errors[i,3] = np.sqrt(np.mean(err_rel ** 2)) * 100 #rRMSE
        errors[i,4] = np.mean(err) #Bias
        errors[i,5] = np.mean(err_rel) * 100 #rBias
        
    m_error = np.median(errors,axis = 0).reshape(1,-1)
    
    print("total elapsed time = {}".format(datetime.datetime.now()-time1))
    print("(pp) elapsed time = {}".format(datetime.datetime.now()-time_single))

    return m_error[0,0]+m_error[0,1]

def simulate_testset(pp,densesize_int, seqlength_int, batchsize_int, filters_int, rH, T, Tsin):
    
    # fixed settings for all experiments (must be equal to settings in opt-function)
    GLOBAL_SETTINGS = {
        'pp': pp,
        'batch_size': batchsize_int, #16-128
        'kernel_size': 3, #ungerade!
        'dense_size': densesize_int, 
        'filters': filters_int, 
        'seq_length': seqlength_int,
        'output_seq_length': 12,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 30,
        'learning_rate': 1e-3,
        'test_start': pd.to_datetime('02012012', format='%d%m%Y'),
        'test_end': pd.to_datetime('28122015', format='%d%m%Y')
    }
    
    ## load data
    data, Well_ID = load_RM_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    
    # inputs
    if rH == 0:
        data = data.drop(columns='rH')
    if T == 0:
        data = data.drop(columns='T')
    if Tsin == 0:
        data = data.drop(columns='Tsin')
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext, OptData, OptData_ext, TestData, TestData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n, OptData_n, OptData_ext_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    X_train, Y_train = split_sequences(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = split_sequences(StopData_ext_n.values, GLOBAL_SETTINGS)
    X_opt, Y_opt = split_sequences(OptData_ext_n.values, GLOBAL_SETTINGS)
    X_test, Y_test= split_sequences(TestData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with different initializations
    inimax = 10
    forecast_idx = TestData_ext_n.index.day < 8
    forecast_idx = forecast_idx[GLOBAL_SETTINGS["seq_length"]:len(TestData_ext_n)-GLOBAL_SETTINGS["output_seq_length"]+1]
    X_test_reduced = X_test[forecast_idx]
    Y_test_reduced = Y_test[forecast_idx]
    testresults_members = np.zeros((len(TestData_n), len(X_test_reduced), inimax))
    testresults_members[:] = np.nan
    for ini in range(inimax):
        model = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  
        
        idx = 0
        for i in range(0,len(X_test_reduced)):
            test_sim_n = model.predict(X_test_reduced[i,:,:].reshape(1,X_test_reduced.shape[1],X_test_reduced.shape[2]))
            test_sim = scaler_gwl.inverse_transform(test_sim_n)
            
            while forecast_idx[idx] == False:
                idx = idx + 1

            testresults_members[idx:idx+GLOBAL_SETTINGS["output_seq_length"], i, ini] = test_sim.reshape(-1,)
            idx = idx+1

    test_sim_median = np.nanmedian(testresults_members,axis = 2)
    
    # get scores
    errors = np.zeros((test_sim_median.shape[1],7))
    errors[:] = np.nan
    TestData_PI1 = extract_PI1_testdata(data, GLOBAL_SETTINGS)
    TestData_PI1 = TestData_PI1[:-GLOBAL_SETTINGS["output_seq_length"]+1]
    TestData_PI1 = TestData_PI1[forecast_idx]
    for i in range(0,test_sim_median.shape[1]):
        sim = np.asarray(test_sim_median[:,i].reshape(-1,1))
        sim = sim[~np.isnan(sim)].reshape(-1,1)
        obs = np.asarray(scaler_gwl.inverse_transform(Y_test_reduced[i,:].reshape(-1,1)))
        
        err = sim-obs
        err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
        err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))
        err_PI = obs - TestData_PI1[i]
        
        errors[i,0] = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2))) #NSE
        r = stats.linregress(sim[:,0], obs[:,0])
        errors[i,1] = r.rvalue ** 2 #R2
        errors[i,2] =  np.sqrt(np.mean(err ** 2)) #RMSE
        errors[i,3] = np.sqrt(np.mean(err_rel ** 2)) * 100 #rRMSE
        errors[i,4] = np.mean(err) #Bias
        errors[i,5] = np.mean(err_rel) * 100 #rBias
        errors[i,6] = 1 - ((np.sum(err ** 2)) / (np.sum((err_PI) ** 2))) #PIop
        
    m_error = np.median(errors,axis = 0).reshape(1,-1)
    scores = pd.DataFrame(np.array([[m_error[0,0], m_error[0,1], m_error[0,2], m_error[0,3], m_error[0,4], m_error[0,5], m_error[0,6]]]),
                       columns=['NSE','R2','RMSE','rRMSE','Bias','rBias','PI'])
    print(scores)
    
    # Ensemble Member Errors
    errors_members = np.zeros((testresults_members.shape[1],inimax,7))
    errors_members[:] = np.nan
    
    for i in range(0,inimax):
        for ii in range(0,testresults_members.shape[1]):
            sim = np.asarray(testresults_members[:,ii,i].reshape(-1,1))
            sim = sim[~np.isnan(sim)].reshape(-1,1)
            obs = np.asarray(scaler_gwl.inverse_transform(Y_test_reduced[ii,:].reshape(-1,1)))
            
            err = sim-obs
            err_rel = (sim-obs)/(np.max(data['GWL'])-np.min(data['GWL']))
            err_nash = obs - np.mean(np.asarray(data['GWL'][(data.index < GLOBAL_SETTINGS["test_start"])]))
            err_PI = obs - TestData_PI1[ii]
            
            errors_members[ii,i,0] = 1 - ((np.sum(err ** 2)) / (np.sum((err_nash) ** 2))) #NSE
            r = stats.linregress(sim[:,0], obs[:,0])
            errors_members[ii,i,1] = r.rvalue ** 2 #R2
            errors_members[ii,i,2] =  np.sqrt(np.mean(err ** 2)) #RMSE
            errors_members[ii,i,3] = np.sqrt(np.mean(err_rel ** 2)) * 100 #rRMSE
            errors_members[ii,i,4] = np.mean(err) #Bias
            errors_members[ii,i,5] = np.mean(err_rel) * 100 #rBias
            errors_members[ii,i,6] = 1 - ((np.sum(err ** 2)) / (np.sum((err_PI) ** 2))) #PIop
            
            #print ensemble member errors
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_NSE.txt',errors_members[:,:,0].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_r2.txt',errors_members[:,:,1].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_rmse.txt',errors_members[:,:,2].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_rrmse.txt',errors_members[:,:,3].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_bias.txt',errors_members[:,:,4].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_rbias.txt',errors_members[:,:,5].transpose(),delimiter=';', fmt = '%.4f')
            np.savetxt('./ensemble_member_errors_'+Well_ID+'_PIop.txt',errors_members[:,:,6].transpose(),delimiter=';', fmt = '%.4f')

    return scores, TestData, inimax, testresults_members, test_sim_median, Well_ID
    
class newJSONLogger(JSONLogger) :

      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"

"""###########################################################################

above only functions

###########################################################################"""

with tf.device("/cpu:0"):
    
    time1 = datetime.datetime.now()
    basedir = './'
    os.chdir(basedir)
    
    for pp in range(17): #loop over all wells
    
        time_single = datetime.datetime.now()
        seed(1)
        tf.random.set_seed(1)
    
        _, Well_ID = load_RM_GW_and_HYRAS_Data(pp)
        
        # Bounded region of parameter space
        pbounds = {'pp': (pp,pp),
                   'seqlength': (1, 52), 
                   'densesize': (1, 256),
                   'batchsize': (16, 256),
                   'filters': (1, 256),
                   'rH':(0,1),
                   'T':(0,1),
                   'Tsin':(0,1),} #constrained optimization technique, so you must specify the minimum and maximum values that can be probed for each parameter
        
        optimizer = BayesianOptimization(
            f= bayesOpt_function, #function that is optimized
            pbounds=pbounds, #opt.-range of parameters
            random_state=1, 
            verbose = 0 # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent, verbose = 2 prints everything
            )
        
        #load existing optimizer
        log_already_available = 0
        if os.path.isfile("./logs_CNN_seq2seq_"+Well_ID+".json"):
            load_logs(optimizer, logs=["./logs_CNN_seq2seq_"+Well_ID+".json"]);
            print("\nExisting optimizer is already aware of {} points.".format(len(optimizer.space)))
            log_already_available = 1
        
        # Saving progress
        logger = newJSONLogger(path="./logs_CNN_seq2seq_"+Well_ID+".json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        
        # random exploration as a start
        f = open('./timelog_CNN_seq2seq_'+Well_ID+'.txt', "w")
        print("Starttime of first iteration: {}\n".format(datetime.datetime.now()), file = f)#this is not looged in json file
        
        if log_already_available == 0:
            optimizer.maximize(
                    init_points=25, #steps of random exploration (random starting points before bayesopt(?))
                    n_iter=0, # steps of bayesian optimization
                    acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
                    xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
                    )
        
        # optimize while improvement during last 10 steps!
        current_step = len(optimizer.res)
        beststep = False
        step = -1
        while not beststep:
            step = step + 1
            beststep = optimizer.res[step] == optimizer.max #search for best iteration step
    
        while current_step < 50: #below < 50 iterations, no termination
                current_step = len(optimizer.res)
                beststep = False
                step = -1
                while not beststep:
                    step = step + 1
                    beststep = optimizer.res[step] == optimizer.max
                print("\nbeststep {}, current step {}".format(step+1, current_step+1))
                optimizer.maximize(
                    init_points=0, #steps of random exploration 
                    n_iter=1, # steps of bayesian optimization
                    acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
                    xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
                    )
                
        while (step + 20 > current_step and current_step < 150): # termination after 50 steps or after 10 steps without improvement
                current_step = len(optimizer.res)
                beststep = False
                step = -1
                while not beststep:
                    step = step + 1
                    beststep = optimizer.res[step] == optimizer.max
                    
                print("\nbeststep {}, current step {}".format(step+1, current_step+1))
                optimizer.maximize(
                    init_points=0, #steps of random exploration
                    n_iter=1, # steps of bayesian optimization
                    acq="ei",# ei  = expected improvmenet (probably the most common acquisition function) 
                    xi=0.05  #  Prefer exploitation (xi=0.0) / Prefer exploration (xi=0.1)
                    )
            
        print("\nBEST:\t{}".format(optimizer.max))
        # for i, res in enumerate(optimizer.res):
        #     print("Iteration {}: \t{}".format(i+1, res))
            
    
        
        #get best values from optimizer
        densesize_int = int(optimizer.max.get("params").get("densesize"))
        seqlength_int = int(optimizer.max.get("params").get("seqlength"))
        batchsize_int = int(optimizer.max.get("params").get("batchsize"))
        filters_int = int(optimizer.max.get("params").get("filters"))
        rH = int(round(optimizer.max.get("params").get("rH")))
        T = int(round(optimizer.max.get("params").get("T")))
        Tsin = int(round(optimizer.max.get("params").get("Tsin")))
        
        #run test set simulations
        t1_test = datetime.datetime.now()
        scores, TestData, inimax, testresults_members, test_sim_median, Well_ID = simulate_testset(pp, densesize_int, seqlength_int, batchsize_int, filters_int, rH, T, Tsin)
        t2_test = datetime.datetime.now()
        f = open('./timelog_CNN_seq2seq_'+Well_ID+'.txt', "a")
        print("Time [s] for Test-Eval (10 inis)\n{}\n".format(t2_test-t1_test), file = f)
        
        # plot Test-Section
        
        # pyplot.figure(figsize=(16,4))
        pyplot.figure(figsize=(15,6))
        for i in range(0,testresults_members.shape[1]):
            for ii in range(0,testresults_members.shape[2]):
                pyplot.plot(TestData.index, testresults_members[:,i,ii], 'r', label='_nolegend_', alpha=0.1)
        pyplot.plot(TestData.index, test_sim_median[:, 0], 'r', label='simulated median')
        for i in range(1,test_sim_median.shape[1]):
            pyplot.plot(TestData.index, test_sim_median[:, i], 'r', label='_nolegend_')

        pyplot.plot(TestData.index, TestData['GWL'], 'b', label ="observed")
        pyplot.title("CNN - Forecast 3 Months seq2seq: "+Well_ID, size=15)
        pyplot.ylabel('GWL [m asl]', size=12)
        pyplot.xlabel('Date',size=12)
        pyplot.legend(fontsize=12,bbox_to_anchor=(1.2, 1),loc='upper right')
        pyplot.tight_layout()

        s = """NSE = {:.2f}\nR²  = {:.2f}\nRMSE = {:.2f}\nrRMSE = {:.2f}
Bias = {:.2f}\nrBias = {:.2f}\nPI = {:.2f}\n\nfilters = {:d}\ndense-size = {:d}\nin_seqlength = {:d}
out_seqlength = {:d}\nbatchsize = {:d}\nrH = {:d}\nT = {:d}\nTsin = {:d}""".format(scores.NSE[0],scores.R2[0],
        scores.RMSE[0],scores.rRMSE[0],scores.Bias[0],scores.rBias[0],scores.PI[0],
        filters_int, densesize_int,seqlength_int,12,batchsize_int,rH,T,Tsin)

        # pyplot.figtext(0.865, 0.18, s, bbox=dict(facecolor='white'))
        pyplot.figtext(0.856, 0.4, s, bbox=dict(facecolor='white'))
        pyplot.savefig(Well_ID+'_testset_CNN_seq2seq.png', dpi=300)
        pyplot.show()
        
        # print log summary file
        f = open('./log_summary_CNN_seq2seq_'+Well_ID+'.txt', "w")
        print("\nBEST:\n\n"+s+"\n", file = f)
        print("best iteration = {}".format(step+1), file = f)
        print("max iteration = {}\n".format(len(optimizer.res)), file = f)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \t{}".format(i+1, res), file = f) 
        f.close()
        
        #print sim data
        for i in range(inimax):
            printdf = pd.DataFrame(data=testresults_members[:,:,i],index=TestData.index)
            printdf.to_csv("./ensemble_member"+str(i) +"_values_CNN_"+Well_ID+'.txt',sep=';', float_format = '%.4f')