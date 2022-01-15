from re import template
from tensorflow import keras
from operator import add
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Multiply, Add, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
from training_utils import *
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import mmap
import random
import argparse
import csv
import base64
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bitsize',
                    help='Size of Morgan Fingerprint', required=False,type=int, default=2048)
    ap.add_argument('-e', '--epochs',
                    help='Number of epochs', required=False,type=int, default=5)
    ap.add_argument('-a', '--nbatches',
                    help='Number of batches for dataset', required=False,type=int, default=10)
    ap.add_argument('-s', '--simple',
                    help='Use simple templates', action='store_true',required=False, default=False)
    
    args = vars(ap.parse_args())
    return args

args = get_arguments()
BITSIZE = args['bitsize']
TEST_RATIO = 0.125
LIM = len(open(f'TRAINING_DATA/REACTIONS{"_SIMPLE" if args["simple"] else ""}', 'rb').readlines())
elu_alpha = 0.1
inputs = keras.Input(shape=(1, BITSIZE))
#inputs = keras.BatchNormalization()(inputs)
x = Dense(512, kernel_initializer='he_normal')(inputs)
x = ELU(alpha=elu_alpha)(x)
x = Dropout(0.2)(x)
x = highway_layers(x, 5)
outputs = Dense(LIM, activation="softmax")(x)

# print(len(open('TRAINING_DATA/REACTIONS').readlines()) + 1)
def process_set(N):
    # x_test = []
    # y_test = []
    x_train = []
    x_temp = set()
    y_train = []
    with open(f'TRAINING_DATA/NET_SET{N}', 'r') as fp:
        for i, (a, b) in enumerate(tqdm(csv.reader(fp, delimiter='\t'))):
            try:
                # a, b = l.split(b'|')
                arr = pickle.loads(base64.b64decode(a))
                # print(arr)
                # input()
                # if tuple(arr) not in x_temp:
                # if random.random() > TEST_RATIO:
                x_train.append(np.log1p(np.array(arr)).reshape((1, 1, BITSIZE)))
                x_temp.add(tuple(arr))
                y_train.append(np.array([int(b)]))
                # else:
                #     x_test.append(np.log1p(np.array(arr)).reshape((1, 1, BITSIZE)))
                #     x_temp.add(tuple(arr))
                #     y_test.append(int(b.decode('utf-8')))
            except KeyboardInterrupt:
                import sys
                sys.exit(0)
            except:
                print('BAD LINE', a, b)
                continue
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    return random.sample(list(zip(x_train, y_train)), len(x_train))#, x_test, y_test

# with open('TRAINING_DATA/TESTING_SET', 'rb') as fp:
#     for i, l in enumerate(tqdm(fp)):
#         a, b = l.split(b'|')
#         arr = pickle.loads(a)
#         if tuple(arr) not in x_temp and int(b.decode('utf-8')) < LIM:
#             x_test.append(keras.utils.normalize(np.array(arr)))
#             x_temp.add(tuple(arr))
#             y_test.append(int(b.decode('utf-8')))

# print(len(x_train))
def batched_generator(data, batch_size=32):
    temp_data = [[], []]
    for i, point in enumerate(data):
        if i % batch_size == 0 and i != 0:
            temp_data = [np.array(a) for a in temp_data]
            yield [temp_data[0]], temp_data[1]
            temp_data = [[], []]
        for ii in range(2):
            temp_data[ii].append(point[ii][0])

# training_generator = iter(random.sample(list(zip(x_train, y_train)), len(x_train)))
# x_train = np.array(x_train)
# y_train = np.array(y_train)


model = keras.Model(inputs, outputs) 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
EPOCHS = args['epochs']
N_BATCHES = args['nbatches']
for _ in range(EPOCHS):
    for i in range(N_BATCHES - 1):
        training_generator = batched_generator(process_set(i))
        model.fit_generator(training_generator, epochs=1)

# x_test, y_test = zip(*process_set(N_BATCHES-1))
# results = model.evaluate(x_test, y_test, batch_size=32)
# print("test loss, test acc:", results)
model.save(f'model{BITSIZE}{"_SIMPLE" if args["simple"] else ""}')
