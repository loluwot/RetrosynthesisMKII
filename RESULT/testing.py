import keras
from operator import add
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import Constant
from keras import backend as K
from keras.layers import ELU
from keras.utils import normalize
import numpy as np
import pickle
import os
#from rdkit import RDLogger
from tqdm import tqdm
import mmap

reactions = []

with open('../TRAINING/RESULTS2') as f:
    for l in f:
        reactions.append(l)


model = keras.models.load_model('model')
mol = Chem.MolFromSmiles('C1=CC=C(C=C1)C1=CC=CC=C1')
mol.UpdatePropertyCache()
print(Chem.MolToSmiles(mol))
#mol2 = Chem.MolFromSmiles('')
vecs = AllChem.GetMorganFingerprintAsBitVect(mol, 2)    
vecs = normalize(np.array(vecs))
res = model.predict(vecs)
top10 = res.argsort()[0][::-1][:10]
for k, v in enumerate(top10):
    os.mkdir('top{}'.format(k))
    rxn = reactions[v]
    
    # patt = Chem.MolFromSmiles(rxn.split('>>')[0], sanitize=False)
    # products = rxn.split('>>')[1]
    # #patt2.UpdatePropertyCache()
    # patt2 = Chem.RemoveHs(patt, sanitize=False)
    # for a in patt2.GetAtoms():
        # a.SetNumRadicalElectrons(0)
        # a.SetNumExplicitHs(0)
        
    # patt2 = Chem.AddHs(patt2)
    # patt2 = Chem.MolToSmiles(patt2)
    
    # rxn = patt2 + '>>' + products
    
    
    rxn = AllChem.ReactionFromSmarts(rxn)
    #Chem.SanitizeMol(patt)
    #print(mol.HasSubstructMatch(patt2))
    rxn.Initialize()
    #print(AllChem.ReactionToSmiles(rxn))
    #print(rxn.IsMoleculeReactant(mol))
    ps = rxn.RunReactants((mol,))
    #print(ps)
    for i, p in enumerate(ps):
        os.mkdir('top{}/entry{}'.format(k, i))
        for j, pp in enumerate(p):
            try:
                pp = Chem.MolFromSmiles(Chem.MolToSmiles(pp))
                pp.UpdatePropertyCache()
                AllChem.Compute2DCoords(pp)
                Draw.MolToFile(pp, 'top{}/entry{}/{}.png'.format(k, i, j), kekulize=False)
            except KeyboardInterrupt:
                exit()
            except:
                print('ERROR')

print(top10)
