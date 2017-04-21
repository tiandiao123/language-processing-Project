'''
Load a model to take a look at the features learned.

Adrian Benton
3/17/2016
'''

import cPickle

def inspect(modelPath):
  f = open(modelPath)
  params, model = cPickle.load(f)
  f.close()
  
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  pass
