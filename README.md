### Twitter Classifier Research Project 
### Location: Department of Computer Science of the Johns Hopkins University


### Description of the research project:

Firstly, there is a data file that contains multiple data sets, and the data sets include text languages about health information and their labels. What we need to do is that designing machine learning algorithms to find the relation between language text information and the corresponding labels. 

We are using Support Vector Machine and Recurrent neural network to solve the problems. It turns out that the language texts of the Twitter have deep relation with health issues such as flu-relavant. More details can be found in the txt file in the src_classifier fold. Those txt file contains information of our training using SVM and GRU algorithms. More models will be added later on!

To test my model:
For example, you can type 
```
python GRU_classifier.py 
```
to train my model. 

Also, make sure to install [keras](https://keras.io/#installation) and [tensorflow](https://www.tensorflow.org/install/) so that you can run my python file!!!

Also , here is an example showing the structure of the GRU:
```
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_10 (Embedding)     (None, 40, 32)            608000    
_________________________________________________________________
dropout_19 (Dropout)         (None, 40, 32)            0         
_________________________________________________________________
gru_10 (GRU)                 (None, 30)                5670      
_________________________________________________________________
dropout_20 (Dropout)         (None, 30)                0         
_________________________________________________________________
dense_10 (Dense)             (None, 2)                 62        
=================================================================


```

Also, here is one of training result of one of data sets:
```
Train on 4311 samples, validate on 1438 samples
Epoch 1/5
4311/4311 [==============================] - 5s - loss: 0.7086 - acc: 0.9188 - val_loss: 0.4077 - val_acc: 0.9249
Epoch 2/5
4311/4311 [==============================] - 4s - loss: 0.3151 - acc: 0.9267 - val_loss: 0.2484 - val_acc: 0.9249
Epoch 3/5
4311/4311 [==============================] - 4s - loss: 0.1736 - acc: 0.9413 - val_loss: 0.1884 - val_acc: 0.9444
Epoch 4/5
4311/4311 [==============================] - 4s - loss: 0.0977 - acc: 0.9712 - val_loss: 0.1721 - val_acc: 0.9590
Epoch 5/5
4311/4311 [==============================] - 4s - loss: 0.0657 - acc: 0.9833 - val_loss: 0.1686 - val_acc: 0.9597
```


### TODO (The following contents were added by Dr. Adrian):

5/6

== Where to find stuff ==

+ Scripts
  - eval*.sh  : run experiments on the grid
  - shared.py : paths and variables shared across multiple scripts (e.g., pointers to datasets)
  - batch_classification.py, nn_classification.py, lstm_classification.py : linear, feedforward net, and RNN classification
  - ingest.py : feature extraction -- Note: I ran TweeboParser ( http://www.cs.cmu.edu/~ark/TweetNLP/ ) by hand on each dataset to generate parse trees.
                This is a manual step for introducing a new dataset.

+ Food for experiments
  - /export/projects/abenton/tweet_behavior_classification/raw_datasets/ : where unprocessed data sit, along with notes and description of each.
  - /export/projects/abenton/tweet_behavior_classification/resources/ : gazeteers, Twitter-trained word2vec embeddings
    = lus_framenet : list of all lexical units from framenet
  - /export/projects/abenton/tweet_behavior_classification/features/  : extracted features, alphabets, etc.
    = *_features.npz : extracted features packaged as compressed numpy arrays.  This is what is read by models/to run experiments
    = tweebo_parser_output : what TweeboParser spits out.  These were stuck here manually before running feature extraction in ingest.py
    = sequences/ : sequences rawer form -- human-readable tokens, both surface and dependency word orders
    = alphabets/ : mappings from feature to index, learned on all datasets

+ Results
  - All under /export/projects/abenton/expfiledb/tweet_behavior_classification/
  - batch-*: linear classifiers
  - batch-nn-*: MLPs, feedforward networks
  - lstm-*: LSTMs (have not dumped these yet)
  - Results from experiments written in each of these directories as experiments/*.run .  Each line in these files contains properties of the experiment along with train, average 5-fold CV, and test accuracy

+ Ignore *-vaccine -- these experiments were on an older dataset whose classes were too imbalanced, replaced with *-vaccine_posneg and *-vaccine_

+ Another approach to try out
  - https://github.com/bdhingra/tweet2vec  -- need to read paper and perhaps need to implement

== Experiment descriptions ==

Explore different feature sets and models to predict opinion or behavior given a tweet.  Hypothesis: these sorts of prediction tasks (Who did what to whom?  What does X think about Y?) can benefit from more linguistic features since we are interested in the relationships between entities.  These features could be syntactic (ordering given by parser), or they could be restricted to specific classes of verbs.  Here I just explore different feature sets and models to feed them into.

Features:
+ n-gram, n \in [1, 3]
+ POS-gram, n \in [1, 3]
+ gazeteer (only considering all lexical units from framenet atm)
+ (average) word2vec embedding

Sequences:
+ Surface word order
+ Dependency parse word order with bracket to denote children

Models/training:
+ Linear classifier trained with hinge loss and elastic-net regularization *DONE*
+ MLP (single hidden layer) trained with minibatch Adadelta sgd.  Dropout regularization on input weights.  *TODO, include additional set of connections from input to output layer*
+ RNN.  Hidden unit (LSTM for now, but can try basic recurrent unit and GRU).  *TODO, wrote code but need to debug and train*

Additional thoughts:
+ Multitask training of MLP/RNN on all datasets at once *TODO*
+ Initialize input weights with, for example, word2vec embeddings *TODO*
+ Try different types of regularization on neural network weights: L2, L1 *TODO*
