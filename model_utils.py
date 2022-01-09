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
from utils import *

reactions = []
additional_info = []
model = keras.models.load_model('model')
inscope_model = keras.models.load_model('inscope_model')
with open('TRAINING_DATA/REACTIONS') as f:
    for l in f:
        reactions.append(l.strip())

filename_set = set()
SMALL_SIZE_FILTER = 3
with open('TRAINING_DATA/REACTIONS_ADDITIONAL') as f:
    for l in f:
        name, index, rxnstr = l.strip().split(',')[:3]
        index = int(index)
        additional_info.append((name, index, rxnstr))
        filename_set.add(name)

# preload = dict()
# for filename in filename_set:
#     preload[filename] = file_to_rxns(filename)


def logit(x):
    return np.log(x/(1-x))

def filter_results(results, rxnid, mol):
    name, index, rxnsmi = additional_info[rxnid]
    # print(results, rxnsmi)
    # rxnsmi = get_reaction_smiles(file_to_rxns(name)[index])
    # rxnsmi = rxnsmi.split(' |')[0]
    product_f = fingerprint([mol])
    reactant_fingerprints = [(i, fingerprint([Chem.MolFromSmiles(x) for x in result])) for i, result in enumerate(results)]
    # agents = get_agents(rxnsmi)
    def is_valid(f1, f2, agents):
        f1s = (np.array(reaction_fingerprint_scratch(f2, f1, agents))).reshape((1, 1, 2048))
        f2s = np.log1p(np.array(f2)).reshape((1, 1, 2048))
        res = inscope_model.predict({'reaction':f1s, 'product':f2s})[0][0][0]
        # print(logit(res))
        return logit(res) > 3, res

    filtered = []
    for idx, rfig in reactant_fingerprints:
        valid, prob = is_valid(rfig, product_f, [])
        if valid:
            filtered.append((idx, prob))
    # filtered = list(filter(lambda x: is_valid(x[1], product_f, []), reactant_fingerprints))
    if len(filtered) == 0:
        return []
    # valid_indices, _ = zip(*filtered)
    return [(list(filter(lambda x: Chem.MolFromSmiles(x).GetNumAtoms() > SMALL_SIZE_FILTER, results[i])), prob) for i, prob in filtered]

def sanity_check(rxnid, mol):
    # print(rxnstr)
    # rxn = Reactions.ReactionFromSmarts(rxnstr)
    rxnstr = reactions[rxnid]
    # print(rxnstr, '---------------------')
    # additional = additional_info[rxnid]
    hashed_mol = HashedMol(Chem.MolToSmiles(mol))
    reactants, products = rxnstr.split('>>')
    # reactants, products = map(lambda x: Chem.MolToSmiles(process_mol(Chem.MolFromSmiles(x, sanitize=False))), [reactants, products])
    prxn, prxnrev = f'{products}>>{reactants}', f'{reactants}>>{products}'
    # prxn, prxnrev = map(lambda x: Reactions.ReactionFromSmarts(x, useSmiles=True), temp)
    # prxn.Initialize()
    # prxnrev.Initialize()
    try:
        # split_results = list(prxn.RunReactants((mol,)))
        split_results = forward_run(prxn, [mol], use_smiles=False, one_product=True)
    except:
        # print('ISSUE1')
        return []
    # print('SPLIT RESULTS', list(map(lambda x: Chem.MolToSmiles(x[0]), split_results)))
    net_res = set()
    for res in split_results:
        # print('CURRENT RES', Chem.MolToSmiles(res[0]))
        try:
            new_results = forward_run(prxnrev, res, use_smiles=False)
            # print('REV RES', list(map(lambda x: Chem.MolToSmiles(x[0]), new_results)))
            # new_results = prxnrev.RunReactants((res[0],))
            if len(new_results) == 0:
                # print('ISSUE2')
                continue
        except:
            continue
        hashed_res = list(map(lambda x: HashedMol(Chem.MolToSmiles(x[0])), new_results))
        # print([hashed.hash for hashed in hashed_res]) 
        # print(hashed_mol.hash)
        if hashed_mol in hashed_res:
            if Chem.MolFromSmiles(Chem.MolToSmiles(res[0])) != None:
                net_res.add(HashedMolSet(Chem.MolToSmiles(res[0])))
    net_res = [x.mols for x in net_res]
    return filter_results(net_res, rxnid, mol)
    # return net_res
    
def get_top_n(smi, topn=15, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    mol.UpdatePropertyCache()
    vecs = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)    
    # vecs = 
    # print(vecs)
    vecs = normalize(np.array(vecs))
    res = model.predict(np.asarray([vecs]))
    topres = res.argsort()[0][0][::-1][:topn*3]
    topres = list(map(lambda x: HashedReaction(smarts_to_smiles(reactions[x]), idx=x), topres))
    # print(topres)
    topres = OrderedSet(topres)
    ftopres = list(filter(lambda x: len(x[0]) != 0, map(lambda v: (sanity_check(v.idx, mol), v.idx, res[0][0][v.idx]), topres)))[:topn]
    ftopres = list(itertools.chain.from_iterable(map(lambda x: [(((x[1], idx), tup[1]*x[2]), tup[0]) for idx, tup in enumerate(x[0])], ftopres)))
    return ftopres

    

# print(reactions[567])
# print(reactions[280])
#HARD
# print(get_top_n('CCC(COC(=O)[C@@H](NP(=O)(Oc1ccccc1)OC[C@H]1O[C@@]([C@@H]([C@@H]1O)O)(C#N)c1ccc2n1ncnc2N)C)CC'))
#EASY
# res = get_top_n('FC1=CC=C(C=C1)C1=CC=C(C=O)C=C1', nbits=2048, topn=30)
# print(res)
# import time
# start_time = time.time()
# res = get_top_n('[O-][N+](=O)C1=CC2=C(OC3(CCC3)CC2=O)C=C1')
# print(res)
# print(time.time() - start_time)
# print(sanity_check('Br-[c;H0;D3;+0:1](:[c:2]):[c:3]>>[c:2]:[c;H0;D3;+0:1]:[c:3]', Chem.MolFromSmiles('FC1=CC=C(C=C1)C1=CC=C(C=O)C=C1')))

# MEDIUM
# print(get_top_n('CCN1C(=O)COc2ccc(CN3CCN(CCOc4cccc5nc(C)ccc45)CC3)cc21'))
