from tensorflow import keras
from operator import add
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Multiply, Add, Lambda, Dot
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
from tensorflow.python.keras.engine import training
from training_utils import *
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import mmap
import random
import argparse
import pandas as pd

from utils import TRAINING_PATH

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bitsize',
                    help='Size of Morgan Fingerprint', required=False,type=int, default=2048)
    ap.add_argument('-a', '--nbatches',
                    help='Number of batches', required=False,type=int, default=10)
    args = vars(ap.parse_args())
    return args

args = get_arguments()

TEST_RATIO = 0.125
BITSIZE = args['bitsize']
N_BATCHES = args['nbatches']
elu_alpha = 0.1
product_input = keras.Input(shape=(1, BITSIZE), name='product')
reactant_input = keras.Input(shape=(1, BITSIZE), name='reactant')
reactant_res = Dense(1024, kernel_initializer='he_normal')(reactant_input)
reactant_res = ELU(alpha=elu_alpha)(reactant_res)

product_res = Dense(1024, kernel_initializer='he_normal')(product_input)
product_res = ELU(alpha=elu_alpha)(product_res)
product_res = highway_layers(product_res, 5)
# print(reactant_res, product_res)
res = Dot(axes=(2, 2), normalize=True)([reactant_res, product_res])
# print(res)
res = Dense(1, activation='sigmoid')(res)
# print(res)
model = keras.Model(
    inputs=[reactant_input, product_input],
    outputs=[res],
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
])

def generate_data(path):
    data = open(TRAINING_PATH + path, 'rb')
    for l in data:
        reactant_f, product_f, label = l.split(b',') 
        reactant_f = normalize(np.array(pickle.loads(reactant_f)))
        reactant_f = reactant_f.reshape((1, 1, BITSIZE))
        product_f = normalize(np.array(pickle.loads(product_f)))
        product_f = product_f.reshape((1, 1, BITSIZE))
        label = np.array([int(label.decode('utf-8'))])
        # print(label)
        yield {'reactant':reactant_f, 'product':product_f}, label

def get_samples(path):
    data = open(TRAINING_PATH + path, 'rb')
    return list(map(lambda x: x.strip().split(b','), data.readlines()))

def get_data(data_arr, batch_size=32):
    # data = open(TRAINING_PATH + path, 'rb')
    # arr = []
    temp_reactant = []
    temp_product = []
    temp_label = []
    for counter, l in enumerate(data_arr):
        reactant_f, product_f, label = l
        reactant_f = normalize(np.array(pickle.loads(reactant_f)))
        temp_reactant.append(reactant_f)
        # reactant_f = reactant_f.reshape((1, 1, BITSIZE))
        product_f = normalize(np.array(pickle.loads(product_f)))
        temp_product.append(product_f)
        # product_f = product_f.reshape((1, 1, BITSIZE))
        label = int(label.decode('utf-8'))
        temp_label.append(label)
        if counter % batch_size == 0:
            yield {'reactant':np.array(temp_reactant), 'product':np.array(temp_reactant)}, np.array(temp_label)
            temp_reactant, temp_product, temp_label = [], [], []
    # return arr

# r1 = normalize(np.random.rand(2048))
# r1 = r1.reshape((1, 1, BITSIZE))
# r2 = normalize(np.random.rand(2048))
# r2 = r2.reshape((1, 1, BITSIZE))
# data_generator = iter ([({'reactant':r1, 'product':r2}, np.asarray([0]))])

# model.fit_generator(data_generator)

for i in range(N_BATCHES - 1):
    # print(training_data)
    net_data = get_samples(f'INSCOPE_DATA{i}')
    random.shuffle(net_data)
    batch_generator = get_data(net_data, batch_size=32)
    model.fit_generator(batch_generator, epochs=1)
    batch_generator = get_data(net_data, batch_size=32)
    model.fit_generator(batch_generator, epochs=1)

test_data = get_samples(f'INSCOPE_DATA{N_BATCHES - 1}')
random.shuffle(test_data)
batch_generator = get_data(test_data, batch_size=32)
model.evaluate_generator(batch_generator)

model.save('inscope_model')