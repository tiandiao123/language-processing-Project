'''
Ingest datasets into format that is easy to train classifiers on.

Adrian Benton
3/9/2016
'''

import cPickle, gzip, json, os, random, sys

import numpy as np
import nltk
from gensim.models import word2vec

import shared

from personal.fmethods  import fopen
from personal.textutils import Alphabet, NGramAlphabetProcessor

import keras

# Feature format: {'IDs':, 'Text':, 'BOW-sequence', 'BOW-stemmed-sequence', 'POS-sequence', 'dependency_ordering':, 'closed_vocab_%s':, , 'Y':, 'Folds':, 'VOCAB_alphabet':, 'POS_alphabet':, 'Y_alphabet':}

# Meaning of tweet annotations/labels
Y_alphabets = {'flu':{0:'flu_awareness', 1:'flu_infection'},
               #'vaccine_who':{0:'vaccine_mention', 1:'vaccinated'},
               'vaccine_isneutral':{0:'neutral', 1:'nonneutral'},
               'vaccine_posneg':{0:'negative', 1:'postive'},
               'cessation':{0:'smoking_mention', 1:'smoking_cessation'}}

# For preprocessing/extracting features
#MAX_TYPES = 100000
#STOPWORDS = set(nltk.corpus.stopwords.words('english')) | set(['rt'])
STEMMER   = nltk.stem.PorterStemmer()

def ldWord2vecEmbeddings(vocab):
  ''' Get embeddings for each word.  This was the model that achieved highest heldout perplexity on a random sample of 2015 tweets/profiles. '''
  
  w2vPath = os.path.join(shared.RESOURCE_DIR, 'tweetsAndProfile_2015-04sampleAndUserEmbeddingData_train.txt.cbow.10.500.50.bin')
  
  model = word2vec.Word2Vec.load_word2vec_format(w2vPath, binary=True)
  embeddings = {}
  for w in vocab:
    if w in model:
      embeddings[w] = model[w]
    
  print 'Loaded embeddings from %s' % (w2vPath)
  
  return embeddings

def runTweetParser(tweetStream):
  ''' Have to run this locally, since COE grid has old library files, then copy to grid. '''
  tmpTxtPath = 'tweet_text_to_parse.txt'
  
  outFile = open(tmpTxtPath, 'w')
  for vs in tweetStream:
    outFile.write(vs[3] + '\n')
  outFile.close()
  
  pwd = os.getcwd()
  
  os.chdir(shared.TWEEBO_HOME)
  os.system('sh run.sh %s' % (os.path.join(pwd, tmpTxtPath)))
  os.chdir(pwd)
  os.system('mv %s.predict %s.predict' % (tmpTxtPath, os.path.join(shared.FEATURE_DIR, tmpTxtPath)))

def tweeboParseToBracketed(trees):
  '''
  Linearizes a TweeboParser dependency parse, converting to bracketed
  notation.
  This is just done by depth-first traversal of the syntax tree.  Tweets may
  have multiple roots in these parses --contain several
  utterances/sentences/ideas.  Get around this by inserting a <ROOT> token
  joining all subtrees.
  
  This is probably not the most efficient way to linearize these graphs,
  but it's an attempt.
  '''
  
  bracketed = []
  
  for tree in trees:
    df_order_list = tree.df_order()
    
    brackets = []
    
    for elt in df_order_list:
      if elt == '<L_BRACKET>':
        brackets.append((elt, elt))
      elif elt == '<R_BRACKET>':
        brackets.append((elt, elt))
      else:
        brackets.append((elt[1], elt[2]))
    
    bracketed.append(brackets)
  
  return bracketed

class Node:
  def __init__(self, value, children=[]):
    self.value      = value
    self.__children = []
    self.is_frozen  = False # If true, then forbid changes to children
  
  def add(self, child):
    if not self.is_frozen:
      self.__children.append(child)
    else:
      raise Exception('Cannot add child, is frozen')
  
  def remove(self, child):
    if not self.is_frozen:
      self.__children.remove(child)
    else:
      raise Exception('Cannot remove child, is frozen')
  
  def pop(self, idx):
    if not self.is_frozen:
      self.__children.pop(idx)
    else:
      raise Exception('Cannot pop child, is frozen')
  
  def numChildren(self):
    return len(self.__children)
  
  def sort(self):
    if not self.is_frozen:
      self.__children.sort(key=lambda n:n.value)
    else:
      raise Exception('Cannot sort children, is frozen')
    
    for child in self.__children:
      child.sort()
  
  def df_order(self, order=[]):
    ''' Return depth-first ordering of node elements '''
    if self.__children:
      order.append('<L_BRACKET>')
    
    for child in self.__children:
      order = child.df_order(order)
    
    if self.__children:
      order.append('<R_BRACKET>')
    
    order.append(self.value)
    
    return order
  
  def to_nested_list(self, parent_list=[]):
    parent_list.append(self.value)
    
    child_list = []
    for child in self.__children:
      child_list = child.to_nested_list(child_list)
    
    if child_list:
      parent_list.append(child_list)
    
    return parent_list
  
  #def __cmp__(self, other):
  #  return self.value.__cmp__

def tweeboParseToTree(inPath):
  '''
  Converts a dependency-parsed tweet (CoNLL format) to a graph.  Since
  a tweet can contain multiple ideas, we have each tweet rooted at a <ROOT>
  token, joining each subtree.
  '''
  
  ROOT = '<ROOT>'
  
  tweets = []
  tweet  = []
  f = open(inPath)
  for ln in f:
    if not ln.strip():
      if tweet:
        tweets.append(tweet)
        tweet = []
      continue
    
    flds  = ln.strip().split()
    idx   = int(flds[0])
    token = flds[1]
    pos   = flds[3]
    head  = int(flds[6])
    
    # Keep tokens not in the parse tree (hashtags) -- they may still be useful
    tweet.append((idx, token, pos, head))
    
  f.close()
  
  if tweet:
    tweets.append(tweet)
  
  print 'Finished reading parses'
  
  trees = []
  
  for tweet in tweets:
    ROOT = Node((0, '<ROOT>', '<ROOT>', -1))
    nodes = dict([(idx, Node((idx, t, pos, head))) for idx, t, pos, head in tweet])
    nodes[0] = ROOT
    
    for idx, t, pos, head in tweet:
      if head == -1: # This node is omitted from parse tree
        continue
      
      nodes[head].add(nodes[idx])
    
    ROOT.sort()
    
    for i, node in nodes.items():
      node.is_frozen = True
    
    trees.append(ROOT)
  
  # Return tweets in sequence and parse trees
  return tweets, trees

def saveSequences(tweets, trees, tweetStream, outPath):
  '''
  Saves the parsed tweets, along with other information we need to train
  classifiers.
  '''
  
  trees_lin = []
  
  def toTokPos(nodeValue):
    if nodeValue == '<R_BRACKET>':
      return ('<R_BRACKET>', '<R_BRACKET>')
    elif nodeValue == '<L_BRACKET>':
      return ('<L_BRACKET>', '<L_BRACKET>')
    else:
      return (nodeValue[1], nodeValue[2])
  
  for tree in trees:
    trees_lin.append(
      map(lambda n: toTokPos(n), tree.df_order([])
        )
    )
  
  IDs = [id for id, y, fold, text in tweetStream]
  Ys = [y for id, y, fold, text in tweetStream]
  folds = [fold for id, y, fold, text in tweetStream]
  
  outFile = fopen(outPath, 'w')
  cPickle.dump({'parsed_tweets':tweets,
                'trees':trees,
                'trees_linearized':trees_lin,
                'y':Ys, 'id':IDs, 'fold':folds}, outFile)
  outFile.close()

def getFluStream(f):
  '''
  May want to retrieve the original status IDs from these data,
  for additional context.
  '''
  
  tweetStream = []
  for ln in f:
    flds = ln.strip().split()
    id = int(flds[0])
    y  = 1 - int(flds[1]) # In the raw data, 1 means awareness, 0 infection, but I want to consider positive cases to be infection
    text = ' '.join(flds[2:])
    
    tweetStream.append((id, y, text))
  
  # Split 75% train/dev, 25% test -- data is assumed to be shuffled
  #random.seed(SEED)
  #random.shuffle(tweetStream)
  splitIdx1 = int(0.5 * len(tweetStream))
  splitIdx2 = int(0.75 * len(tweetStream))
  
  getFoldId = lambda curr_idx: 1*(curr_idx>=splitIdx1) + 1*(curr_idx>=splitIdx2)
  tweetStream = [(id, y, getFoldId(i), text) for i, (id, y, text) in enumerate(tweetStream)]
  
  return tweetStream

def getVaccineWhoStream(f):
  ''' Whether the tweet reports having gotten a vaccination or not. '''
  tweetStream = []
  for ln in f:
    tweet = json.loads(ln)
    
    id = int(tweet['tweet_id'])
    text = tweet['tweet_text']
    #y = 1 if tweet['label'] == 'yes' else 0
    y = 1 if tweet['new_label'] == 'yes' else 0
    
    fold = -1
    if tweet['fold']   == 'train':
      fold = 0
    elif tweet['fold'] == 'dev':
      fold = 1
    elif tweet['fold'] == 'test':
      fold = 2
    
    tweetStream.append((id, y, fold, text))
  
  return tweetStream

def getVaccineIsneutralStream(f):
  ''' Whether the tweet has neutral or non-neutral sentiment regarding vaccines. '''
  
  tweetStream = []
  for ln in f:
    tweet = ln.strip().split('\t')
    
    id = int(tweet[0])
    text = tweet[1]
    #y = 1 if tweet['label'] == 'yes' else 0
    y = 1 if tweet[3] == 'nonneutral' else 0
    
    tweetStream.append((id, y, text))
  
  splitIdx1 = int(0.5 * len(tweetStream))
  splitIdx2 = int(0.75 * len(tweetStream))
  
  getFoldId = lambda curr_idx: 1*(curr_idx>=splitIdx1) + 1*(curr_idx>=splitIdx2)
  tweetStream = [(id, y, getFoldId(i), text) for i, (id, y, text) in enumerate(tweetStream)]
  
  return tweetStream

def getVaccinePosnegStream(f):
  ''' Whether the tweet holds positive or negative sentiment towards vaccines. '''
  
  tweetStream = []
  for ln in f:
    tweet = ln.strip().split('\t')
    
    id = int(tweet[0])
    text = tweet[1]
    #y = 1 if tweet['label'] == 'yes' else 0
    y = 1 if tweet[3] == 'positive' else 0
    
    tweetStream.append((id, y, text))
  
  splitIdx1 = int(0.5 * len(tweetStream))
  splitIdx2 = int(0.75 * len(tweetStream))
  
  getFoldId = lambda curr_idx: 1*(curr_idx>=splitIdx1) + 1*(curr_idx>=splitIdx2)
  tweetStream = [(id, y, getFoldId(i), text) for i, (id, y, text) in enumerate(tweetStream)]
  
  return tweetStream

def getCessationStream(f):
  tweetStream = []
  for id, ln in enumerate(f):
    tweet = json.loads(ln)
    
    y = 1 if tweet['cessation'] else 0
    text = tweet['cleaned_content']
    
    tweetStream.append((id, y, text))
  
  # Split 75% train/dev, 25% test, data is assumed to be shuffled
  #random.seed(SEED)
  #random.shuffle(tweetStream)
  splitIdx1 = int(0.5 * len(tweetStream))
  splitIdx2 = int(0.75 * len(tweetStream))
  
  getFoldId = lambda curr_idx: 1*(curr_idx>=splitIdx1) + 1*(curr_idx>=splitIdx2)
  tweetStream = [(id, y, getFoldId(i), text) for i, (id, y, text) in enumerate(tweetStream)]
  
  return tweetStream

IN_PATHS    = [os.path.join(shared.RAWDATA_DIR, p) for p in ['tobacco_cessation/10k-tob-cess-data-0714-stricter.shuf.json',
                                                             'flu/flu_awarenessVersusInfection.shuf.txt',
                                                             #'vaccines/tweet_majority_labels_who_folds.handlabel.shuf.json' # This problem had too few positive examples
                                                             'vaccines/tweet_majority_labels_sentiment_neutral.shuf.unanim',
                                                             'vaccines/tweet_majority_labels_sentiment_posneg.shuf.unanim']] # Point to the raw data files
TWEET_ITERS = [getCessationStream,
               getFluStream,
               #getVaccineWhoStream,
               getVaccineIsneutralStream,
               getVaccinePosnegStream] # Point to functions that take a file, and extract raw tweets from it
PARSER_PATHS   = [os.path.join(shared.FEATURE_DIR, p) for p
                  in [
                    'tweebo_parser_output/10k-tob-cess-data-0714-stricter.shuf.txt.predict',
                    'tweebo_parser_output/flu_awarenessVersusInfection.shuf.txt.predict',
                    #'tweebo_parser_output/tweet_majority_labels_who_folds.handlabel.shuf.txt.predict' # Too few positive examples for Who
                    'tweebo_parser_output/tweet_majority_labels_sentiment_neutral_folds.shuf.unanim.txt.predict',
                    'tweebo_parser_output/tweet_majority_labels_sentiment_posneg_folds.shuf.unanim.txt.predict']] # Where TweeboParser output is written
SEQ_PATHS = [os.path.join(shared.FEATURE_DIR, 'sequences/%s_sequences.pickle' % (d)) for d in shared.DSET_NAMES] # Where sequences/parse trees are written
OUT_PATHS = [os.path.join(shared.FEATURE_DIR, '%s_features.npz' % (d)) for d in shared.DSET_NAMES] # Where features are written -- sequences/vectors

try:
  GAZETEER_PATHS = [os.path.join(shared.GAZETEER_DIR, p) for p in os.listdir(shared.GAZETEER_DIR) ] # Paths to subsets of tokens we think may be discriminative for our task.
except Exception, ex:
  print 'Cannot find %s: %s' % (shared.GAZETEER_DIR, ex)
  GAZETEER_PATHS = []

SEED = 12345 # For building train/dev/test sets

def ingestAll():
  # Map token/POS to feature index
  tokenAlpha      = Alphabet()
  tokenLowerAlpha = Alphabet()
  
  tokenNGramAlpha = NGramAlphabetProcessor(maxN=3)
  posNGramAlpha   = NGramAlphabetProcessor(maxN=3)
  
  # Initialize gazeteers with closed vocabulary
  gazeteerAlphas  = {}
  for p in GAZETEER_PATHS:
    k = os.path.basename(p).replace('.txt.gz', '')
    
    alpha = Alphabet()
    f = fopen(p)
    for ln in f:
      alpha.put(ln.strip())
    f.close()
    alpha.isFixed = True
    
    gazeteerAlphas[k] = alpha
  
  for inPath, tweet_iter, parsePath, seqPath, outPath in zip(IN_PATHS, TWEET_ITERS, PARSER_PATHS, SEQ_PATHS, OUT_PATHS):
    f = fopen(inPath)
    tweetStream = tweet_iter(f)
    f.close()
    
    # Read and linearize Tweebo parses.
    parsedTweets, trees = tweeboParseToTree(parsePath)
    saveSequences(parsedTweets, trees, tweetStream, seqPath)
    
    print 'Saved sequences for %s' % (inPath)
    
    f = fopen(seqPath)
    seqs = cPickle.load(f)
    f.close()
    
    IDs = np.asarray(seqs['id'])
    Ys  = np.asarray(seqs['y'])
    folds = np.asarray(seqs['fold'])
    
    tweets = seqs['parsed_tweets']
    trees_lin = seqs['trees_linearized']
    
    features = {'id':IDs, 'y':Ys, 'fold':folds}
    
    features['token_seq'] = []
    features['tokenLower_seq'] = []
    features['tokenStem_seq'] = []
    features['bigramStem_seq'] = []
    features['trigramStem_seq'] = []
    features['pos_seq'] = []
    features['bigramPos_seq'] = []
    features['trigramPos_seq'] = []
    
    features['depParseOrder_token_seq'] = []
    features['depParseOrder_tokenLower_seq'] = []
    features['depParseOrder_tokenStem_seq'] = []
    features['depParseOrder_bigramStem_seq'] = []
    features['depParseOrder_trigramStem_seq'] = []
    features['depParseOrder_pos_seq'] = []
    features['depParseOrder_bigramPos_seq'] = []
    features['depParseOrder_trigramPos_seq'] = []
    
    for k in gazeteerAlphas:
      features['%s_seq'% (k)] = []
      features['depParseOrder_%s_seq' % (k)] = []
    
    seqs = ([[(tok, pos) for i, tok, pos, head in tweet] for tweet in parsedTweets], trees_lin)
    
    # Initialize alphabets and save feature sequences (of ints)
    for index_key, seq in zip(['', 'depParseOrder_'], seqs):
      for tIdx, tweet in enumerate(seq):
        stemmed_tweet = []
        
        for t, p in tweet:
          try:
            stemmed_tweet.append(STEMMER.stem(t.lower()))
          except Exception, e:
            stemmed_tweet.append(STEMMER.stem(t.lower().decode('ascii', errors='ignore')))
            print 'Problem with processing tweet:', tIdx, t, p
        
        lowered_tweet   = [t.lower() for t, p in tweet]
        
        tok_ngram_idxes = tokenNGramAlpha.consumeSequence(stemmed_tweet)
        pos_ngram_idxes = posNGramAlpha.consumeSequence([p for t, p in tweet])
        
        features[index_key + 'tokenStem_seq'].append(tok_ngram_idxes[0])
        features[index_key + 'bigramStem_seq'].append(tok_ngram_idxes[1])
        features[index_key + 'trigramStem_seq'].append(tok_ngram_idxes[2])
        features[index_key + 'pos_seq'].append(pos_ngram_idxes[0])
        features[index_key + 'bigramPos_seq'].append(pos_ngram_idxes[1])
        features[index_key + 'trigramPos_seq'].append(pos_ngram_idxes[2])
        
        features[index_key + 'token_seq'].append([[tokenAlpha.put(t) for t, p in tweet]])
        features[index_key + 'tokenLower_seq'].append([tokenLowerAlpha.put(t) for t in lowered_tweet])
        
        for k in gazeteerAlphas:
          gazeteer_seq = []
          for t in lowered_tweet:
            idx = gazeteerAlphas[k].put(t)
            if idx is None:
              gazeteer_seq.append(-1)
            else:
              gazeteer_seq.append(idx)
          
          features[index_key + '%s_seq' % (k)].append(gazeteer_seq)
        
        if not tIdx % 1000:
          print 'Processed tweet %.1fK for sequence types: "%s" ' % (tIdx/1000., index_key)
      
      print 'Populated alphabets and built sequences for "%s"' % (index_key)
    
    np.savez_compressed(outPath, **features)
    
    print 'Saved features to %s' % (outPath)
  
  # Freeze alphabets
  tokenAlpha.isFixed      = True
  tokenLowerAlpha.isFixed = True
  tokenNGramAlpha.setFixed(True)
  posNGramAlpha.setFixed(True)
  
  # Dump alphabets to file
  def _dumpAlphabet(alpha, name):
    outFile = fopen(os.path.join(shared.FEATURE_DIR, 'alphabets', name + '.alphabet.pickle.gz'), 'w')
    cPickle.dump(alpha, outFile)
    outFile.close()
  
  for nm, alpha in ([('token', tokenAlpha), ('token_lower', tokenLowerAlpha), ('stemmed_1to3gram', tokenNGramAlpha), ('pos_1to3gram', posNGramAlpha)] + [(k + '_gazeteer', v) for k, v in gazeteerAlphas.items()]):
    _dumpAlphabet(alpha, nm)
  print 'Saved alphabets'

if __name__ == '__main__':
  ingestAll()
