import pandas as pd
import numpy as np
import logging
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import optimizers

VERBOSE = 1

PATH = './input/'
TRAIN = 'flowers.csv'

def getLineValue(path, line):
	file = open(path, "r").readlines()
	return int(file[line - 1])

def getVarNames(x):
	idxs = ()
	for i in xrange(0,x):
		idxs += ('X' + str(i + 1),)
	return idxs

NUM_ATTRIBUTES = getLineValue(PATH + TRAIN, 2)
NUM_CLASSES = getLineValue(PATH + TRAIN, 3)
HEADERS = getVarNames(NUM_ATTRIBUTES)
HEADERS += ("Class",)

NFOLDS = 3
MOMENTUM = 0.3
LEARNING_RATE = 0.3
EPOCHS = 30
BATCH = 10

# Set random seed to get the same results
SEED = 1618033
np.random.seed(SEED)

logging.basicConfig(level= logging.DEBUG if VERBOSE == 1 else logging.ERROR);

df = pd.read_csv(PATH + TRAIN, skiprows = 3, names = HEADERS)
dataset = df.values
X = dataset[:, 0:NUM_ATTRIBUTES].astype(float)
Y = dataset[:, NUM_ATTRIBUTES]

# Encoder to get dataset classes
encoder = LabelEncoder()
encoder.fit(Y)
logging.info("Dataset classes:\n" + " ".join(encoder.classes_))

# Get output indexes of the encoder class
encoded_Y = encoder.transform(Y)
# Use one hot vector to indicate the class
onehot_y = np_utils.to_categorical(encoded_Y)

def main_model():
	# Inputs -> [ 2 * NAttributes hidden ] -> [ NAttributes hidden ] -> NClasses outputs
	model = Sequential()

	# Activations
	# selu - Scaled Exponential Linear Unit
	# softplus
	# softsign
	# softmax
	# relu
	# tanh
	# sigmoid
	model.add(Dense(NUM_ATTRIBUTES * 2, input_dim = NUM_ATTRIBUTES, activation = 'tanh'))
	model.add(Dense(NUM_ATTRIBUTES, activation = 'relu'))
	# model.add(Dense(5, activation = 'tanh'))

	# softmax output layer. In range of 0 and 1 may be used as predicted probabilities.
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	# Losses
	# mean_squared_error
	# mean_absolute_error
	# mean_absolute_percentage_error
	# mean_squared_logarithmic_error
	# squared_hinge
	# hinge
	# categorical_hinge
	# logcosh
	# categorical_crossentropy
	# sparse_categorical_crossentropy
	# binary_crossentropy

	# Optimizers
	# sgd
	# rmsProps
	# adagrad
	# adadelta
	# adamax
	# nadam
	# tfoptimizer
	sgd = optimizers.SGD(lr = LEARNING_RATE, decay = 1e-6, momentum = MOMENTUM, nesterov = True)
	model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

	return model

estimator = KerasClassifier(build_fn = main_model, epochs = EPOCHS, batch_size = BATCH, verbose = VERBOSE)

kfold = KFold(n_splits = NFOLDS, shuffle = True, random_state = SEED )

results = cross_val_score(estimator, X, onehot_y, cv = kfold)
print("\nPrecision: %.2f%% \nStd Deviation: %.2f%%" % (results.mean() * 100, results.std() * 100))
