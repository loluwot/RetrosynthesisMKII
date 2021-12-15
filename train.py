from tensorflow import keras
from operator import add
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tqdm import tqdm
import mmap
import random

TEST_RATIO = 0.125
LIM = len(open('TRAINING_DATA/REACTIONS', 'rb').readlines())
elu_alpha = 0.1
inputs = keras.Input(shape=(1, 2048))
#inputs = keras.BatchNormalization()(inputs)
x = Dense(512, kernel_initializer='he_normal')(inputs)
x = ELU(alpha=elu_alpha)(x)
x = Dropout(0.2)(x)
# print(len(open('TRAINING_DATA/REACTIONS').readlines()) + 1)
x_test = []
y_test = []
x_train = []
x_temp = set()
y_train = []

with open('TRAINING_DATA/NET_SET', 'rb') as fp:
    for i, l in enumerate(tqdm(fp)):
        a, b = l.split(b'|')
        arr = pickle.loads(a)
        #print(arr)
        if tuple(arr) not in x_temp and int(b.decode('utf-8')) < LIM:
            if random.random() > TEST_RATIO:
                x_train.append(keras.utils.normalize(np.array(arr)))
                x_temp.add(tuple(arr))
                y_train.append(int(b.decode('utf-8')))
            else:
                x_test.append(keras.utils.normalize(np.array(arr)))
                x_temp.add(tuple(arr))
                y_test.append(int(b.decode('utf-8')))

# with open('TRAINING_DATA/TESTING_SET', 'rb') as fp:
#     for i, l in enumerate(tqdm(fp)):
#         a, b = l.split(b'|')
#         arr = pickle.loads(a)
#         if tuple(arr) not in x_temp and int(b.decode('utf-8')) < LIM:
#             x_test.append(keras.utils.normalize(np.array(arr)))
#             x_temp.add(tuple(arr))
#             y_test.append(int(b.decode('utf-8')))

print(len(x_train))

x_train, y_train = zip(*random.sample(list(zip(x_train, y_train)), len(x_train)))   

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


outputs = Dense(LIM, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, verbose=1)

results = model.evaluate(x_test, y_test, batch_size=32)
print("test loss, test acc:", results)

model.save('model')
