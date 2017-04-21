'''
Pointers to files used by many scripts.

Adrian Benton
3/9/2016
'''

import os

# Where the Noah's ARK TweeboParser sits.  Points to where it is on local machine -- COE grid library files are too old.
TWEEBO_HOME = '/home/adrianb/Desktop/TweeboParser'

PROJECT_NAME = 'tweet_behavior_classification'

HOME = os.path.join('/export/projects/abenton/', PROJECT_NAME)

RESOURCE_DIR = os.path.join(HOME, 'resources')
RAWDATA_DIR = os.path.join(HOME, 'raw_datasets')
MODEL_DIR = os.path.join(HOME, 'models')
FEATURE_DIR = os.path.join(HOME, 'features')
GAZETEER_DIR = os.path.join(RESOURCE_DIR, 'gazeteers')

DSET_NAMES = ['cessation', # Tobacco cessation data
              'flu',       # Flu data
              #'vaccine'   # Vaccine Who data
              'vaccine_isneutral', # Vaccine sentiment neutral data
              'vaccine_posneg'     # Vaccine sentiment nonneutral, positive vs. negative data
             ]

FEATURE_PATHS = [os.path.join(FEATURE_DIR, '%s_features.npz' % (n)) for n in DSET_NAMES]

# For splitting up data into folds
SEED = 123456789
