import keras
from operator import add
#from rdkit import Chem
#from rdkit.Chem import AllChem
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import Constant
from keras import backend as K
from keras.layers import ELU
from keras.utils import normalize
from keras.models import load_model
import numpy as np
import pickle
#from rdkit import RDLogger
from tqdm import tqdm
import mmap
#RDLogger.DisableLog('rdApp.*')

#fp = list(AllChem.GetMorganFingerprintAsBitVect(m1,2))
#fp2 = list(AllChem.GetMorganFingerprintAsBitVect(m2,2))
#in1 = list(map(add, fp, fp2))
#in1 = list(map(lambda x : x/max(in1), in1))

elu_alpha = 0.1
inputs = keras.Input(shape=(1, 2048))
#inputs = keras.BatchNormalization()(inputs)
x = Dense(512, kernel_initializer='he_normal')(inputs)
x = ELU(alpha=elu_alpha)(x)
x = Dropout(0.2)(x)
outputs = Dense(9913, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

count = 0
#model.summary()
x_test = []
y_test = []
for _ in range(10):
    x_train = []
    x_temp = set()
    y_train = []
    
    with open('../TRAINING/TRAIN_SCRAMBLE', 'rb') as fp:
        mm = mmap.mmap(fp.fileno(), 0, prot=mmap.PROT_READ)
        mm.seek(count)
        for i, l in enumerate(tqdm(iter(mm.readline, b''))):
            if i > 107879:
                break
            a, b = l.split(b'|')
            arr = pickle.loads(a)
            if i > (107879*3)//4:
                x_test.append(keras.utils.normalize(np.array(arr)))
                #x_temp.append(np.array(arr).tostring())
                y_test.append(int(b.decode('utf-8')))
                count += len(l)
                continue
            #print(arr)
            if np.array(arr).tostring() not in x_temp:
                x_train.append(keras.utils.normalize(np.array(arr)))
                x_temp.append(np.array(arr).tostring())
                y_train.append(int(b.decode('utf-8')))
            count += len(l)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    print (len(set(x_temp)))
    model.fit(x_train, y_train, epochs=1, verbose=1)
    #results = model.evaluate(x_test, y_test, batch_size=32)
    #print("test loss, test acc:", results)
    del x_train, y_train#, x_test, y_test
    
#x_test = []
#y_test = []
#with open('../TRAINING/TESTING_PRE', 'rb') as fp:
#    mm = mmap.mmap(fp.fileno(), 0, prot=mmap.PROT_READ)
#    for i, l in enumerate(tqdm(iter(mm.readline, b''))):
#        x_test.append(keras.utils.normalize(np.array(arr)))
#        y_test.append(int(b.decode('utf-8')))
        
        
x_test = np.array(x_test)
y_test = np.array(y_test)

results = model.evaluate(x_test, y_test, batch_size=32)
print("test loss, test acc:", results)

model.save('model')
