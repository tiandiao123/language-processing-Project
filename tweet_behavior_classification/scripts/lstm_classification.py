'''
Classify documents with LSTM.  Want to map sequence of tokens/feature
vectors to a single binary class.

Adrian Benton
3/29/2016
'''

import cPickle, os

import numpy as np
import scipy as sp

import nn_classification, ingest, shared

from personal.experiment.expdb import ExperimentFileClient
from personal.fmethods import fopen

from sklearn import cross_validation

from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.callbacks import Callback

# What to do the grid search over
architectures = [(50,), (200,), (500,), (1000,)]
dropouts      = [0.1, 0.3, 0.5]
units         = [LSTM]

#architectures = [(1000,), (1000, 200), (1000, 200, 50)]
#dropouts      = [0.5]
#units = [SimpleRNN, LSTM, GRU]

batch_sizes   = [2048]

# Feature sets we are going to evaluate, each a different job
fsetNames = [('tokenStem_seq',),     # Lowercased + stemmed unigrams
             ('pos_seq',),           # POS unigrams
             ('tokenStem_seq', 'pos_seq'),
             ('lus_framenet_seq',),
             ('lus_framenet_seq', 'pos_seq')
           ]

def buildSparseSequences(featurePath, feature_sets, maxLen_hard=50):
  '''
  Builds a collection of X, sequences, indexed by feature ID for separate
  feature sets.  Each sequence of feature vectors will be fed to parallel
  LSTMs.
  '''
  
  X     = []
  testX = []
  for fset in feature_sets:
    ids, folds, X_fset, Y, testX_fset, testY = nn_classification.buildSparseData(featurePath, (fset,), maxLen_hard)
    X.append(X_fset); testX.append(testX_fset)
  
  return ids, folds, X, Y, testX, testY

class EpochLogger(Callback):
  def __init__(self):
    self.N_EPOCHS   = 0
    self.TRAIN_LOSS = []
    self.DEV_LOSS   = []
  
  def on_epoch_end(self, epoch, logs={}):
    self.N_EPOCHS += 1
    
    if 'loss' in logs:
      self.TRAIN_LOSS.append(logs['loss'])
    if 'val_loss' in logs:
      self.DEV_LOSS.append(logs['val_loss'])

class RNNClassifier(ExperimentFileClient):
  '''
  Recurrent neural network for binary classification of a sequence.
  '''
  
  def __init__(self, dsetName, dsetPath, featureNames, architecture, input_dim, dropout_rate, recurrent_unit_constructor=LSTM, bidirectional=False, epochs=100, n_folds=5):
    
    # Call super constructor.
    ExperimentFileClient.__init__(self, shared.PROJECT_NAME,
                                  'seq-rnn-%s-%s' % (
        recurrent_unit_constructor.__name__, dsetName),
                                  'RNNClassifier-%s-%s-%s-%s-%s-%s-%s' % (
        dsetName, epochs, recurrent_unit_constructor.__name__, bidirectional,
        ':'.join([str(d) for d in dropout_rate]),
        ':'.join([str(a) for a in architecture]),
        ':'.join(featureNames)), 
                                  params={'architecture':architecture,
                                          'dropout':dropout_rate,
                                          'epochs':epochs,
                                          'feature_names':featureNames,
                                          'dataset_path':dsetPath,
                                          'dataset':dsetName,
                                          'n_folds':n_folds,
                                          'unit_type':recurrent_unit_constructor.__name__,
                                          'bidirectional':bidirectional},
                                  addtl={'description':'Recurrent NN with option to switch out hidden unit type.  Consumes a sequence and predicts binary label for the sequence.'})
    
    self.featureNames  = featureNames
    self.dsetName      = dsetName
    self.dsetPath      = dsetPath
    self.epochs        = epochs
    self.dropout       = dropout_rate
    self.architecture  = architecture
    self.n_folds       = n_folds
    self.unit_cons     = recurrent_unit_constructor
    self.bidirectional = bidirectional
    
    self._buildArchitecture(input_dim)
    
    self.start()
  
  def _buildArchitecture(self, input_dim):
    '''
    Put the network together.  To begin with, we'll just support sparse inputs (list of feature indices firing.)
    '''
    
    from keras.models import Sequential
    from keras.layers import Activation, Dense, Dropout, Merge, TimeDistributedMerge
    from keras.optimizers import SGD, Adadelta, Adam
    
    fset_models = []
    
    for fset, idim in zip(self.featureNames, input_dim):
      fset_model = Sequential()
      if not self.bidirectional:
        for i in range(len(self.architecture)):
          if i == 0:
            fset_model.add(self.unit_cons(input_dim=idim, output_dim=self.architecture[i]))
          else:
            fset_model.add(self.unit_cons(output_dim=self.architecture[i]))
          fset_model.add(Dropout(self.dropout[i]))
        
        fset_model.add(Dense(1, activation='sigmoid'))
      else:
        lr_models = []
        
        # Build two LSTM hidden units, one that runs forward, the other backward, and sum them together before making a prediction
        for lToR in [False, True]:
          m = Sequential()
          
          for i in range(len(self.architecture)-1):
            if i == 0:
              m.add(self.unit_cons(input_dim=idim, output_dim=self.architecture[i], go_backwards=lToR))
            else:
              m.add(self.unit_cons(output_dim=self.architecture[i], go_backwards=lToR))
            m.add(Dropout(self.dropout[i]))
          
          lr_models.append(m)
        
        fset_model = Sequential()
        fset_model.add(Merge(lr_models, mode='sum'))
      
      fset_models.append(fset_model)
      
      self.model = Sequential()
      # Concatenate all feature sets' hidden layers and use these to predict class
      if len(fset_models) > 1:
        self.model.add(Merge(fset_models, mode='concat'))
      else:
        self.model = fset_models[0]
        self.model.add(Dense(1, activation='sigmoid'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam()
    adadelta = Adadelta(lr=0.5)
    optimizer = adadelta
    
    self.model.compile(loss='binary_crossentropy',
                       optimizer=optimizer,
                       class_mode='binary')
    self.initWeights = self.model.get_weights()
  
  def train(self, X, Y, testX, testY, fixedIter=False):
    scores = []
    
    from keras.callbacks import Callback, EarlyStopping
    
    def _getAcc(x, y):
      n    = y.shape[0]
      pred = 1.*(self.model.predict(x)>0.5)
      acc  = 1. - np.abs(pred - y.reshape((n, 1))).sum()/n
      return acc
    
    epochLogger  = EpochLogger()
    
    cv = cross_validation.StratifiedKFold(Y, self.n_folds, shuffle=True, random_state=shared.SEED)
    for cvIdx, (trainMask, testMask) in enumerate(cv):
      if fixedIter:
        callbacks = [epochLogger]
        print '== Train Params ==\t%s\t%s\t%s\t%s\t%s' % (self.architecture, self.unit_cons, self.dropout, self.featureNames, cvIdx)
      else:
        earlyStopper = EarlyStopping(monitor='val_loss', patience=10,
                                     mode='min')
        callbacks = [earlyStopper, epochLogger]
      
      trainX = [x[trainMask,:] for x in X] if len(X) > 1 else X[0][trainMask,:]
      devX   = [x[testMask,:] for x in X]  if len(X) > 1 else X[0][testMask,:]
      
      self.model.fit(trainX, Y[trainMask], nb_epoch=self.epochs,
                     validation_data=(devX, Y[testMask]),
                     batch_size=2048, callbacks=callbacks, show_accuracy=True)
      devLoss = self.model.test_on_batch(devX, Y[testMask])
      devAcc = _getAcc(devX, Y[testMask])
      print 'devLoss:', devLoss, devAcc
      
      dev_score = devAcc
      scores.append(dev_score)
      self.model.set_weights(self.initWeights)
      epochLogger.TRAIN_LOSS = []
      epochLogger.DEV_LOSS   = []
    
    # Average number of epochs yielding best dev accuracy.
    # Used to train on all training data.
    totalEpochs = epochLogger.N_EPOCHS
    avgEpochs   = totalEpochs/self.n_folds
    print 'Training on all data for %d epochs' % (avgEpochs)
    
    avg_dev_score = np.mean(scores)
    
    trainX = X if len(X) > 1 else X[0]
    self.model.fit(trainX, Y, nb_epoch=avgEpochs, batch_size=2048)
    
    trainAcc = _getAcc(trainX, Y)
    testAcc  = _getAcc(testX,  testY)
    result = dict([('dev_fold%d' % (fold), s) for fold, s in enumerate(scores)] + [('crossval_dev', avg_dev_score), ('train', trainAcc), ('test', testAcc)])
    
    self.writeRun(result)
    #import pdb; pdb.set_trace()
    self.writeModel((self.params, self.model.get_weights()))
    
    self.log('%s -- %f %f %f' % (self.params, trainAcc, avg_dev_score, testAcc))
    
    return trainAcc, avg_dev_score
  
  def stop(self):
    self.close()

def trainAndEvalClassifier(dsetName, dsetPath, fsetNames, dropouts=dropouts, architectures=architectures):
  '''
  Runs grid search over regularization params, writing out train/dev/test accuracies for each setting.
  '''
  
  import pandas as pd
  
  ids, folds, X, Y, testX, testY = buildSparseSequences(dsetPath, fsetNames)
  
  d_results = []
  
  for epochs in [1000]:
    for unit in units:
      for bidirectional in [False, True]:
        for arch in architectures:
          for d in dropouts:
            classifier = RNNClassifier(dsetName, dsetPath, fsetNames, arch, [x.shape[1] for x in X], [d]*len(arch), unit, bidirectional, epochs=epochs, n_folds=5)
            
            train, dev = classifier.train(X, Y, testX, testY)
            d_results.append({'epochs':epochs, 'dropout':d,
                              'unit':unit.__name__, 'bidirectional':bidirectional,
                              'architecture':'-'.join([str(v) for v in arch]),
                              'train_acc':train, 'dev_acc':dev})
            
            classifier.stop()
  
  df = pd.DataFrame(d_results)
  
  bestDevIdx = np.argmax(df['dev_acc'])
  print 'Class split', Y.sum(), (1-Y).sum(), float(Y.sum())/(Y.sum() + (1-Y).sum())
  
  print df[bestDevIdx:(bestDevIdx+1)]

def evalAllFsets(dsetName, dsetPath, dropouts=[0.5], arch=[(1000,)]):
  for names in fsetNames:
    print 'Dset: %s, Fset: %s' % (dsetName, '-'.join(names))
    trainAndEvalClassifier(dsetName, dsetPath, names, dropouts, architectures)

if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("dset",
                      help="name of dataset (flu, cessation, vaccine_neutral, vaccine_posneg)")
  #parser.add_argument("fsetNames", help="semicolon-delimited feature sets to build input vectors with")
  #parser.add_argument("architecture", help="semicolon-delimited layer widths to build MLP")
  #parser.add_argument("dropout", type=float, help="amount of dropout regularization to apply to input layer")
  args = parser.parse_args()
  
  #fname = tuple(args.fsetNames.split(';'))
  dsetName = args.dset
  dsetFeatures = os.path.join(shared.FEATURE_DIR, '%s_features.npz' % (dsetName))
  
  for fname in fsetNames:
    trainAndEvalClassifier(dsetName, dsetFeatures, fname, dropouts, architectures)
