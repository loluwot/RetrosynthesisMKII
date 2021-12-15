from tensorflow import keras
from operator import add
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ELU
from tensorflow.keras.utils import normalize
import numpy as np
import pickle
import os
import mmap
import networkx as nx
from ordered_set import OrderedSet
from rdkit.Chem import rdmolops
import matplotlib.pyplot as plt
from mol_utils import *

reactions = []
model = keras.models.load_model('model')
with open('TRAINING_DATA/REACTIONS') as f:
    for l in f:
        reactions.append(l.strip())

def sanity_check(rxnstr, mol):
    rxn = Reactions.ReactionFromSmarts(rxnstr)
    hashed_mol = HashedMol(Chem.MolToSmiles(mol))
    print('RXNSTR', rxnstr)
    reactants, products = rxnstr.split('>>')
    if products[0] == '(' and products[-1] == ')':
        products = products[1:-1]
    reactants, products = map(lambda x: Chem.MolToSmiles(process_mol(Chem.MolFromSmiles(x, sanitize=False))), [reactants, products])
    # if '.' in reactants:
    #     return []
    temp = [f'{products}>>({reactants})', f'({reactants})>>{products}']
    prxn, prxnrev = map(lambda x: Reactions.ReactionFromSmarts(x, useSmiles=True), temp)
    prxn.Initialize()
    prxnrev.Initialize()
    try:
        split_results = list(prxn.RunReactants((mol,)))
    except:
        # print('ISSUE')
        return []
    # print('SPLIT RESULTS', list(map(lambda x: Chem.MolToSmiles(x[0]), split_results)))
    net_res = set()
    for res in split_results:
        try:
            new_results = prxnrev.RunReactants((res[0],))
            if len(new_results) == 0:
                # print('ISSUE')
                continue
        except:
            continue
        # print('RESULTS', list(map(lambda x: Chem.MolToSmiles(x[0]), new_results)))
        hashed_res = list(map(lambda x: HashedMol(Chem.MolToSmiles(x[0])), new_results))
        if hashed_mol in hashed_res:
            if Chem.MolFromSmiles(Chem.MolToSmiles(res[0])) != None:
                net_res.add(HashedMolSet(Chem.MolToSmiles(res[0])))
            # return True
    # print(net_res)
    net_res = [x.mols for x in net_res]
    # print(net_res)
    # net_res = list(map(lambda x: x.molstr.split('.'), net_res))
    return net_res

def get_top_n(smi, topn=15):
    mol = Chem.MolFromSmiles(smi)
    mol.UpdatePropertyCache()
    vecs = AllChem.GetMorganFingerprintAsBitVect(mol, 2)    
    vecs = normalize(np.array(vecs))
    # print(vecs)
    vecs = np.expand_dims(vecs, axis=0)
    res = model.predict([vecs])
    topres = res.argsort()[0][0][::-1][:topn*5]
    topres = list(map(lambda x: HashedReaction(reactions[x], idx=x), topres))
    topres = OrderedSet(topres)
    ftopres = list(filter(lambda x: len(x[0]) != 0, map(lambda v: (sanity_check(v.real_smarts, mol), v.idx), topres)))[:topn]
    return ftopres

    
# print(reactions[567])
# print(reactions[280])
#HARD
# print(get_top_n('CCC(COC(=O)[C@@H](NP(=O)(Oc1ccccc1)OC[C@H]1O[C@@]([C@@H]([C@@H]1O)O)(C#N)c1ccc2n1ncnc2N)C)CC'))
#EASY
print(get_top_n('FC1=CC=C(C=C1)C1=CC=C(C=O)C=C1'))
# MEDIUM
# print(get_top_n('CCN1C(=O)COc2ccc(CN3CCN(CCOc4cccc5nc(C)ccc45)CC3)cc21'))