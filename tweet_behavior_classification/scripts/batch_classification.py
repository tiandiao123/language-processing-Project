'''
Classify documents based on aggregated features over entire tweet.  Just use
SGDClassifier in sklearn with L1 regularization.

Adrian Benton
3/15/2016
'''

import cPickle

import numpy as np
import scipy as sp

import ingest, shared

from personal.experiment.expdb import ExperimentFileClient
from personal.fmethods import fopen

from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation

# What to do the grid search over
alphas = [b * 10.**e for b in [1.] for e in range(-5, 2, 1)]
ratios = [0.0, 0.15, 0.25]

# Feature sets we are going to evaluate, each a different job
fsetNames = [('token_avgWord2vec',), # Average word2vec embeddings for lowercased tokens
             ('token_seq',),         # Raw unigrams
             ('tokenLower_seq',),    # Lowercased unigrams
             ('tokenStem_seq',),     # Lowercased + stemmed unigrams
             ('pos_seq',),           # POS unigrams
             ('lus_framenet_seq',),  # Lexical units from framenet
             ('tokenStem_seq', 'bigramStem_seq'), # Stemmed unigrams + bigrams
             ('pos_seq', 'bigramPos_seq'),        # POS unigrams + bigrams
             ('tokenStem_seq', 'bigramStem_seq', 'trigramStem_seq'), # Stemmed unigrams + bigrams + trigrams
             ('pos_seq', 'bigramPos_seq', 'trigramPos_seq'),         # POS unigrams + bigrams + trigrams
             ('lus_framenet_seq', 'pos_seq', 'bigramStem_seq'),      # Lexical units from framenet + POS + stemmed bigrams
             ('tokenStem_seq', 'bigramStem_seq', 'pos_seq', 'bigramPos_seq') # Stemmed/POS unigrams + bigrams
           ]

# Mapping from name of a feature set to the where its alphabet is stored.  If
# the 
fsetNameToAlpha = {}

'''
Cessation:
Baseline: 0.916
Unigram: 0.964
Uni+Bi+Trigram: 0.972

Flu:
Baseline: 0.516
Uni/Bi/Trigram: 0.816

Vaccine (old, before relabeling, Who):
Baseline: 0.916
Uni/Bi/Trigram: 0.929
Word2vec: 0.936

Vaccine (current, Sentiment neutral/nonneutral)
Baseline: 0.685175 381 0.314748201439
Unigram: 0.730
Uni/Bigram: 0.721
Uni/Bi/Trigram: 0.728

Vaccine (current, Sentiment negative/positive)
Baseline: 0.618
Unigram: 0.774
Uni/Bigram: 0.779
Uni/Bi/Trigram: 0.801
'''

class BatchSGDClassifier(ExperimentFileClient):
  def __init__(self, dsetName, dsetPath, featureNames, alpha, l1_ratio, epochs=100, n_folds=5):
    
    # Call super constructor.  
    ExperimentFileClient.__init__(self, shared.PROJECT_NAME,
                                  'batch-%s' % (dsetName),
                                  'SGDHingeClassifier-%s-%s-%s-%s-%s' % (dsetName, epochs, alpha, l1_ratio, ':'.join(featureNames)), 
                                  params={'l1l2_reg':alpha,
                                          'l1_ratio':l1_ratio,
                                          'epochs':epochs,
                                          'feature_names':featureNames,
                                          'dataset_path':dsetPath,
                                          'dataset':dsetName,
                                          'n_folds':n_folds},
                                  addtl={'description':'SGD classifier with hinge loss + l1/l2 regularization.  Each tweet is a static feature vector.'})
    
    self.featureNames = featureNames
    self.dsetName = dsetName
    self.classifier = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, n_iter=epochs, n_jobs=4)
    self.dsetPath   = dsetPath
    self.epochs     = epochs
    self.n_folds    = n_folds
    
    self.start()
  
  def setAlpha(self, alpha, l1_ratio):
    self.classifier.set_params(alpha=alpha, l1_ratio=l1_ratio)
    
    self.params['l1l2_reg'] = alpha
    self.params['l1_ratio'] = l1_ratio
    
    print 'Set alpha:%s, l1_ratio:%s' % (alpha, l1_ratio)
  
  def train(self, X, Y, testX, testY):
    scores = cross_validation.cross_val_score(self.classifier, X, Y,
         cv=cross_validation.StratifiedKFold(Y, self.n_folds, shuffle=True,
                                             random_state=shared.SEED), scoring='accuracy')
    
    avg_dev_score = np.mean(scores)
    
    self.classifier.fit(X, Y)
    
    trainAcc = self.classifier.score(X, Y)
    testAcc  = self.classifier.score(testX, testY)
    result = dict([('dev_fold%d' % (fold), s) for fold, s in enumerate(scores)] + [('crossval_dev', avg_dev_score), ('train', trainAcc), ('test', testAcc)])
    
    self.writeRun(result)
    self.writeModel((self.params, self.classifier))
    
    self.log('%s -- %f %f %f' % (self.params, trainAcc, avg_dev_score, testAcc))
    
    return trainAcc, avg_dev_score
  
  def stop(self):
    self.close()

def buildData(featurePath, feature_sets):
  '''
  Builds feature vectors for each set of features and concatenates them.
  Merges train and dev sets together since we'll pick parameters based on
  cross-fold validation.
  '''
  
  f = np.load(featurePath)
  
  ids   = f['id']
  y     = f['y']
  folds = f['fold']
  is_test = folds == 2
  
  featureVecs = []
  
  for fset in feature_sets:
    if fset == 'token_avgWord2vec': # Evaluate avg word2vec embeddings
      alphaF = fopen('/export/projects/abenton/tweet_behavior_classification/features/alphabets/token_lower.alphabet.pickle.gz')
      alpha = cPickle.load(alphaF)
      vocabulary = set(alpha._wToI.keys())
      alphaF.close()
      
      wToE = ingest.ldWord2vecEmbeddings(vocabulary)
      
      data = f['tokenLower_seq']
      
      if len(data.shape) > 1 and data.dtype=='O':
        data = data[:,0]
      
      avg_embeddings = []
      for row in data:
        total_embedding = np.asarray([0. for i in range(500)])
        inserted = 0.
        for i in row:
          w = alpha.get(i)
          if w in wToE:
            total_embedding += wToE[w]
            inserted += 1
        
        if inserted > 0:
          total_embedding /= inserted
        
        avg_embeddings.append(total_embedding)
      
      featureVecs.append(np.asarray(avg_embeddings))
    else:
      data = f[fset]
      if len(data.shape) > 1 and data.dtype=='O':
        data = data[:,0]
      
      values, rows, cols = [], [], []
      maxCIdx = 0
      maxRIdx = 0
      for rIdx, row in enumerate(data):
        for cIdx in set(row):
          if cIdx >= 0:
            values.append(1.)
            rows.append(rIdx)
            cols.append(cIdx)
            maxCIdx = maxCIdx if cIdx < maxCIdx else cIdx
      maxRIdx = rIdx
      
      featureVec = sp.sparse.csr_matrix((values, (rows, cols)), shape=(maxRIdx+1, maxCIdx+1))
      featureVecs.append(featureVec)
  
  Y     = y[~is_test]
  testY = y[is_test]
  
  if any([fset == 'token_avgWord2vec' for fset in feature_sets]): # Concatenate dense matrices
    newFVecs = []
    for ft in featureVecs:
      if type(ft) == np.ndarray:
        newFVecs.append(sp.sparse.csr_matrix(ft))
      else:
        newFVecs.append(ft)
    featureVecs = newFVecs
  
  x = sp.sparse.hstack(featureVecs).tocsr()
  
  X     = x[~is_test,:]
  testX = x[is_test,:]
  
  return ids, folds, X, Y, testX, testY

def trainAndEvalClassifier(dsetName, dsetPath, fsetNames):
  '''
  Runs grid search over regularization params, writing out train/dev/test accuracies for each setting.
  '''
  
  import pandas as pd
  
  ids, folds, X, Y, testX, testY = buildData(dsetPath, fsetNames)
  
  alphaRatioPairs = [(a, r) for a in alphas for r in ratios]
  alpha, l1_ratio = alphaRatioPairs[0]
  
  classifier = BatchSGDClassifier(dsetName, dsetPath, fsetNames, alpha, l1_ratio, epochs=100, n_folds=5)
  
  d = []
  
  for alpha, l1_ratio in alphaRatioPairs:
    classifier.setAlpha(alpha, l1_ratio)
    train, dev = classifier.train(X, Y, testX, testY)
    d.append({'alpha':alpha, 'l1_ratio':l1_ratio, 'train_acc':train, 'dev_acc':dev})
  classifier.stop()
  
  df = pd.DataFrame(d)
  
  bestDevIdx = np.argmax(df['dev_acc'])
  print 'Class split', Y.sum(), (1-Y).sum(), float(Y.sum())/(Y.sum() + (1-Y).sum())
  
  print df[bestDevIdx:(bestDevIdx+1)]

def evalAllFsets(dsetName, dsetPath):
  for names in fsetNames:
    print 'Dset: %s, Fset: %s' % (dsetName, '-'.join(names))
    trainAndEvalClassifier(dsetName, dsetPath, names)

if __name__ == '__main__':
  pass
