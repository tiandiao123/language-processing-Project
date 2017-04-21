'''
Classify documents with a deep net, looking at BOW-style representation of
tweet.  Uses dropout regularization.

Adrian Benton
3/15/2016
'''

import cPickle

import numpy as np
import scipy as sp

import batch_classification, ingest, shared

from personal.experiment.expdb import ExperimentFileClient
from personal.fmethods import fopen

from sklearn import cross_validation

from keras.callbacks import Callback, EarlyStopping
from keras.layers.normalization import BatchNormalization

# What to do the grid search over
#architectures = [(5,)]
architectures = [(5,), (10,), (20,), (50,), (200,), (500,)]
dropouts      = [0.5, 0.3, 0.1]

#architectures = [(1000,), (1000, 200), (1000, 200, 50)]
#dropouts      = [0.5]

batch_sizes   = [32]

# Feature sets we are going to evaluate, each a different job
'''
fsetNames = [('token_seq',),         # Raw unigrams
             ('tokenLower_seq',),    # Lowercased unigrams
             ('tokenStem_seq',),     # Lowercased + stemmed unigrams
             ('pos_seq',),           # POS unigrams
             ('lus_framenet_seq',),
             ('tokenStem_seq', 'pos_seq'),
             ('lus_framenet_seq', 'pos_seq')
           ]

# Evaluating features preserving some sense of ordering
fsetNames = [('bigramStem_seq',),
             ('tokenStem_seq', 'bigramStem_seq'),
             ('tokenStem_seq', 'bigramStem_seq', 'trigramStem_seq'),
             ('pos_seq', 'bigramPos_seq', 'trigramPos_seq'),
             ('tokenStem_seq', 'bigramStem_seq', 'trigramStem_seq',
              'pos_seq', 'bigramPos_seq', 'trigramPos_seq')
           ]
'''

# Evaluating features preserving some sense of ordering
fsetNames = [('bigramStem_seq',),
             ('tokenStem_seq', 'bigramStem_seq'),
             ('tokenStem_seq', 'bigramStem_seq', 'trigramStem_seq'),
             ('tokenStem_seq', 'bigramStem_seq', 'pos_seq', 'bigramPos_seq', 'trigramPos_seq')
           ]

'''
Cessation:
Baseline: 0.916
Unigram, only embedding: 0.962

Flu:
Baseline: 0.516
Unigram, embedding+direct: 0.808
Unigram + POS, only embedding: 0.813

Vaccine (current, Sentiment neutral/nonneutral)
Baseline: 0.685175 381 0.314748201439
Unigram (only lower, only embedding): 0.750
Unigram+POS (only embedding): 0.725

Vaccine (current, Sentiment negative/positive)
Baseline: 0.618
Unigram (only embedding): 0.764
Unigram+POS (only embedding): 0.779
'''

def buildSparseData(featurePath, feature_sets, maxLen_hard=50):
  '''
  Builds feature vectors for each set of features and concatenates them.
  Features are all binary, represented by their feature indices.
  Merges train and dev sets together since we'll pick parameters based on
  cross-fold validation.
  '''
  
  f = np.load(featurePath)
  
  ids   = f['id']
  y     = f['y']
  folds = f['fold']
  is_test = folds == 2
  
  featureVecs = []
  
  maxCIdx = 0
  for fset in feature_sets:
    data = f[fset]
    if len(data.shape) > 1 and data.dtype=='O':
      data = data[:,0]
    
    data = [[r+maxCIdx for r in row] for row in data]
    
    fvec = []
    for rIdx, row in enumerate(data):
      newRow = []
      for cIdx in set(row):
        if cIdx >= 0:
          newRow.append(cIdx)
          maxCIdx = maxCIdx if cIdx < maxCIdx else cIdx
      fvec.append(newRow)
    
    featureVecs.append(fvec)
  
  dummyFeature = maxCIdx+1 # Used to pad short sequences to the right.
  
  Y     = y[~is_test]
  testY = y[is_test]
  
  x = reduce(lambda x,y: x+y, featureVecs)
  maxLen = min(maxLen_hard, max([len(r) for r in x]))
  
  def _pad(row):
    ''' Pad rows that are too short. '''
    if len(row) >= maxLen:
      return row[:maxLen]
    else:
      return row + [dummyFeature for i in range(maxLen-len(row))]
  
  x = np.asarray([_pad(row) for row in x])
  
  X     = x[~is_test]
  testX = x[is_test]
  
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

class NNClassifier(ExperimentFileClient):
  def __init__(self, dsetName, dsetPath, featureNames, architecture, input_dim, dropout_rate, epochs=100, n_folds=5, is_sparse=True, input_to_output=False):
    
    # Call super constructor.  
    ExperimentFileClient.__init__(self, shared.PROJECT_NAME,
                                  'batch-nn-%s' % (dsetName),
                                  'NNClassifier-%s-%s-%s-%s-%s-%s' % (dsetName, epochs, ':'.join([str(d) for d in dropout_rate]),
                                                                   ':'.join([str(a) for a in architecture]), ':'.join(featureNames), input_to_output), 
                                  params={'architecture':architecture,
                                          'dropout':dropout_rate,
                                          'epochs':epochs,
                                          'feature_names':featureNames,
                                          'dataset_path':dsetPath,
                                          'dataset':dsetName,
                                          'n_folds':n_folds,
                                          'direct_input_to_output':input_to_output},
                                  addtl={'description':'Deep FF NN classifier.  Uses ReLUs and dropout, Each tweet is a static feature vector.'})
    
    self.featureNames = featureNames
    self.dsetName     = dsetName
    self.dsetPath     = dsetPath
    self.epochs       = epochs
    self.dropout      = dropout_rate
    self.architecture = architecture
    self.n_folds      = n_folds
    self.is_sparse    = is_sparse
    self.input_to_output = input_to_output # Connect the input layer directly to the output
    
    self._buildArchitecture(input_dim, is_sparse)
    
    self.start()
    
    #import pdb; pdb.set_trace()
  
  def _buildArchitecture(self, input_dim, input_is_sparse=True):
    '''
    Put the network together.
    '''
    
    from keras.models import Graph, Sequential
    from keras.layers import Activation, Dense, Dropout, Embedding
    from keras.layers.core import TimeDistributedMerge
    from keras.optimizers import SGD, Adadelta, Adam
    
    self.model = Graph()
    
    if input_is_sparse:
      self.model.add_input(name='input', input_shape=(input_dim,))
      self.model.add_node(Dropout(self.dropout[0]), name='d0', input='input')
      self.model.add_node(Embedding(input_dim, self.architecture[0], init='uniform', name='input_embedding'), input='d0', name='input_embedding')
      self.model.add_node(TimeDistributedMerge(), input='input_embedding', name='l0')
    else:
      self.model.add_input(name='input', input_shape=(input_dim,))
      self.model.add_node(Dense(self.architecture[0], input_dim=input_dim, init='uniform', name='input_layer', activation='relu'), name='l0', input='input')
      self.model.add_node(Dropout(self.dropout[0]), name='d0', input='l0')
      #self.model.add_node(BatchNormalization(), name='b0', input='d0')
    
    for i in range(1, len(self.architecture)): 
     self.model.add_node(Dense(self.architecture[i], init='uniform', activation='relu'), name='l%d' % (i), input='d%d' % (i-1))
     self.model.add_node(Dropout(self.dropout[i]), name='d%d' % (i), input='l%d' % (i))
     #self.model.add_node(BatchNormalization(), name='b%d' % (i), input='d%d' % (i))
    
    if self.input_to_output:
      self.model.add_node(Dense(1, activation='sigmoid'), name='out_layer', inputs=['d%d' % (len(self.architecture)-1), 'input'], merge_mode='concat')
    else:
      self.model.add_node(Dense(1, activation='sigmoid'), name='out_layer', input='d%d' % (len(self.architecture)-1))
    
    self.model.add_output(name='output', input='out_layer')
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam()
    adadelta = Adadelta(lr=0.1)
    optimizer = adadelta
    
    self.model.compile(loss={'output':'binary_crossentropy'},
                       optimizer=optimizer)
    
    # Vanilla, sequential MLP
    '''
    self.model = Sequential()
    
    self.model.add(Dense(self.architecture[0], input_dim=input_dim, init='uniform', name='input_layer', activation='relu'))
    self.model.add(Dropout(self.dropout[0]))
    
    for i in range(1, len(self.architecture)):
      self.model.add(Dense(self.architecture[i], init='uniform', activation='relu'))
      self.model.add(Dropout(self.dropout[i]))
    
    if self.input_to_output:
      self.model.add(Dense(1, activation='sigmoid'))
    else:
      self.model.add(Dense(1, activation='sigmoid'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam()
    adadelta = Adadelta(lr=0.5)
    optimizer = adadelta
    
    print self.model.summary()
    
    self.model.compile(loss='binary_crossentropy',
                       optimizer=optimizer,
                       class_mode='binary')
    '''
    
    self.initWeights = self.model.get_weights()
  
  def train(self, X, Y, testX, testY, fixedIter=False):
    scores = []
    
    def _getAcc(x, y):
      n    = y.shape[0]
      pred = 1.*(self.model.predict({'input':x})['output']>0.5)
      acc  = 1. - np.abs(pred - y.reshape((n, 1))).sum()/n
      return acc
    
    def _getAccSeq(x, y):
      n    = y.shape[0]
      pred = 1.*(self.model.predict(x)>0.5)
      acc  = 1. - np.abs(pred - y.reshape((n, 1))).sum()/n
      return acc
    
    epochLogger  = EpochLogger()
    
    cv = cross_validation.StratifiedKFold(Y, self.n_folds, shuffle=True, random_state=shared.SEED)
    for cvIdx, (trainMask, testMask) in enumerate(cv):
      if fixedIter:
        callbacks = [epochLogger]
        print '== Train Params ==\t%s\t%s\t%s\t%s' % (self.architecture, self.dropout, self.featureNames, cvIdx)
      else:
        earlyStopper = EarlyStopping(monitor='val_loss', patience=4,
                                 mode='min')
        callbacks = [earlyStopper, epochLogger]
      
      #for i in range(self.epochs):
      self.model.fit({'input':X[trainMask,:], 'output':Y[trainMask]}, validation_data={'input':X[testMask,:], 'output':Y[testMask]}, nb_epoch=self.epochs, batch_size=batch_sizes[0], callbacks=callbacks)
      #  devAcc   = _getAcc(X[testMask,:], Y[testMask])
      #  trainAcc = _getAcc(X[trainMask,:], Y[trainMask])
      #  print devAcc, trainAcc
      
      #self.model.fit(X[trainMask,:], Y[trainMask], nb_epoch=self.epochs, validation_data=(X[testMask,:], Y[testMask]), batch_size=batch_sizes[0], callbacks=callbacks, show_accuracy=True)
      
      #devLoss = self.model.test_on_batch(X[testMask,:], Y[testMask])
      devLoss = self.model.test_on_batch({'input':X[testMask,:], 'output':Y[testMask]})
      devAcc = _getAcc(X[testMask,:], Y[testMask])
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
    
    #self.model.fit({X, Y}, nb_epoch=avgEpochs, batch_size=batch_sizes[0])
    self.model.fit({'input':X, 'output':Y}, nb_epoch=avgEpochs, batch_size=batch_sizes[0])
    
    trainAcc = _getAcc(X, Y)
    testAcc  = _getAcc(testX, testY)
    result = dict([('dev_fold%d' % (fold), s) for fold, s in enumerate(scores)] + [('crossval_dev', avg_dev_score), ('train', trainAcc), ('test', testAcc)])
    
    self.writeRun(result)
    #import pdb; pdb.set_trace()
    self.writeModel((self.params, self.model.get_weights()))
    
    self.log('%s -- %f %f %f' % (self.params, trainAcc, avg_dev_score, testAcc))
    
    return trainAcc, avg_dev_score
  
  def stop(self):
    self.close()

def trainAndEvalClassifier(dsetName, dsetPath, fsetNames, dropouts=dropouts, architectures=architectures, is_sparse=True, input_to_output=False):
  '''
  Runs grid search over regularization params, writing out train/dev/test accuracies for each setting.
  '''
  
  import pandas as pd
  
  if is_sparse:
    ids, folds, X, Y, testX, testY = buildSparseData(dsetPath, fsetNames)
  else:
    ids, folds, X, Y, testX, testY = batch_classification.buildData(dsetPath, fsetNames)
    X, testX = X.toarray(), testX.toarray()
  
  d_results = []
  
  for epochs in [1000]:
    for arch in architectures:
      for d in dropouts:
        if is_sparse:
          classifier = NNClassifier(dsetName, dsetPath, fsetNames, arch, np.max(X)+1, [d]*len(arch), epochs=epochs, n_folds=5, is_sparse=is_sparse, input_to_output=input_to_output)
        else:
          classifier = NNClassifier(dsetName, dsetPath, fsetNames, arch, X.shape[1], [d]*len(arch), epochs=epochs, n_folds=5, is_sparse=is_sparse, input_to_output=input_to_output)
        
        train, dev = classifier.train(X, Y, testX, testY, False)
        d_results.append({'epochs':epochs, 'dropout':d, 'architecture':'-'.join([str(v) for v in arch]), 'train_acc':train, 'dev_acc':dev})
      
      classifier.stop()
  
  df = pd.DataFrame(d_results)
  
  bestDevIdx = np.argmax(df['dev_acc'])
  print 'Class split', Y.sum(), (1-Y).sum(), float(Y.sum())/(Y.sum() + (1-Y).sum())
  
  print df[bestDevIdx:(bestDevIdx+1)]

def evalAllFsets(dsetName, dsetPath, dropouts=[0.5], arch=[(1000,)]):
  for names in fsetNames:
    print 'Dset: %s, Fset: %s' % (dsetName, '-'.join(names))
    trainAndEvalClassifier(dsetName, dsetPath, names, dropouts, architectures)

def testEmbeddingLayer():
  X = [[np.random.randint(10) for i in range(np.random.randint(2, 10))] for j in range(1000)]
  W = 1. - 2*np.random.random(10)
  #Y = np.asarray([[W[x].sum()] for x in X])
  #Y = Y > 0.
  
  X_padded = np.asarray([x + [1000 for j in range(len(x), 10)] for x in X])
  
  values = [1.0  for j, r in enumerate(X) for cIdx in r]
  rows   = [j    for j, r in enumerate(X) for cIdx in r]
  cols   = [cIdx for j, r in enumerate(X) for cIdx in r]
  
  denseX = sp.sparse.csr_matrix((values, (rows, cols)), shape=(max(rows) + 1, 10))
  denseX = denseX.toarray()
  Y = denseX.dot(W)
  #Y = denseX.dot(W) > 0.
  
  from keras.models import Sequential
  from keras.layers import Activation, Dense, Dropout, Embedding
  from keras.layers.core import Flatten, TimeDistributedMerge
  from keras.optimizers import SGD
  
  model = Sequential()
  
  #Dense(1) is the first hidden layer.
  # in the first layer, you must specify the expected input data shape.
  
  #model.add(Embedding(1001, 1, init='uniform', input_length=10))
  #model.add(TimeDistributedMerge())
  
  model.add(Dense(1, input_dim=10, init='uniform'))
  model.add(Dropout(0.0))
  #model.add(Activation('sigmoid'))
  
  '''
  model = Sequential()
  
  # Dense(self.architecture[0]) is the first hidden layer.
  # in the first layer, you must specify the expected input data shape.
  
  model.add(Embedding(1000, 128, init='uniform', input_length=10))
  model.add(TimeDistributedMerge())
  model.add(Activation('sigmoid'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  '''
  
  sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mse', optimizer=sgd)
  
  #model.fit(X_padded, Y, nb_epoch=1000, validation_split=0.2, batch_size=1024)
  model.fit(denseX, Y, nb_epoch=1000, validation_split=0.2, batch_size=1024, show_accuracy=True)
  
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  import argparse
  
  parser = argparse.ArgumentParser()
  parser.add_argument("dset",
                      help="name of dataset (flu, cessation, vaccine_neutral, vaccine_posneg)")
  #parser.add_argument("fsetNames", help="semicolon-delimited feature sets to build input vectors with")
  #parser.add_argument("architecture", help="semicolon-delimited layer widths to build MLP")
  #parser.add_argument("dropout", type=float, help="amount of dropout regularization to apply to input layer")
  parser.add_argument("--intoout", action='store_true', help='connect input layer directly to output, in addition to the feedforward layers')
  args = parser.parse_args()
  
  #fname = tuple(args.fsetNames.split(';'))
  dsetName = args.dset
  dsetFeatures = '/export/projects/abenton/tweet_behavior_classification/features/%s_features.npz' % (dsetName)
  
  input_to_output  = args.intoout
  
  for fname in fsetNames:
    trainAndEvalClassifier(dsetName, dsetFeatures, fname, dropouts, architectures, False, input_to_output)
