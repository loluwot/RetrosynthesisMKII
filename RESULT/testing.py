import keras
from operator import add
from rdkit import Chem
from rdkit.Chem import Descriptors
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
from rdkit import RDLogger
from tqdm import tqdm
import mmap
import networkx as nx
from ordered_set import OrderedSet
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
RDLogger.DisableLog('rdApp.*')
reactions = []

with open('../TRAINING/RESULTS2') as f:
    for l in f:
        reactions.append(l)
def topology_from_rdkit(rdkit_molecule):

    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology
    
def retrosynthesis_step(smi, topn=10):
    model = keras.models.load_model('model')
    mol = Chem.MolFromSmiles(smi)
    mol.UpdatePropertyCache()
    #print(Chem.MolToSmiles(mol))
    #mol2 = Chem.MolFromSmiles('')
    vecs = AllChem.GetMorganFingerprintAsBitVect(mol, 2)    
    vecs = normalize(np.array(vecs))
    #print(vecs)
    #print(vecs.shape)
    res = model.predict(vecs)
    top10 = res.argsort()[0][::-1][:topn]
    #top10=[55]
    #print(top10)
    mols = OrderedSet()
    for k, v in enumerate(top10):
        #print(k, v)
        #os.mkdir('top{}'.format(k))
        rxn = reactions[v]
        rxns = rxn.split('>>')
        patt = Chem.MolFromSmiles(rxns[0], sanitize=False)
        patt2 = Chem.MolFromSmiles(rxns[1], sanitize=False)
        # #patt2.UpdatePropertyCache()
        # patt2 = Chem.RemoveHs(patt, sanitize=False)
        for a in patt.GetAtoms():
            a.SetNumRadicalElectrons(0)
            a.SetNumExplicitHs(0)
        for a in patt2.GetAtoms():
            a.SetNumRadicalElectrons(0)
            a.SetNumExplicitHs(0)
        # patt2 = Chem.AddHs(patt2)
        react = Chem.MolToSmiles(patt)
        prod = Chem.MolToSmiles(patt2)
        #print(react)
        #print(prod)
        #input()
        rxn2 = '{}>>({})'.format(react, prod)
        rxnrev = '({})>>{}'.format(prod, react)
       # print(rxn, rxn2, rxnrev)
        rxn = AllChem.ReactionFromSmarts(rxn2)
        if len(rxn.GetReactants()) > 1:
            print('TOO MANY COOKS')
            continue
        #rxn2 = AllChem.ReactionFromSmarts(rxn2)
        rxnrev = AllChem.ReactionFromSmarts(rxnrev)
        #Chem.SanitizeMol(patt)
        #print(mol.HasSubstructMatch(patt2))
        rxn.Initialize()
        #rxn2.Initialize()
        rxnrev.Initialize()
        #print(AllChem.ReactionToSmiles(rxn))
        #print(rxn.IsMoleculeReactant(mol))
        ps = list(rxn.RunReactants((mol,)))
        #ps2 = rxn2.RunReactants((mol,))
        #print(ps2)
        #ps.extend(ps2)
        #print(ps)
        for i, p in enumerate(ps):
            #os.mkdir('top{}/entry{}'.format(k, i))
            #print(p)
            s = ''
            for pp in p:
                for a in pp.GetAtoms():
                    a.SetNumExplicitHs(0)
                    a.SetNumRadicalElectrons(0)
                s += Chem.MolToSmiles(pp) + '.'
            #print(s)
            s1 = Chem.MolFromSmiles(s[:-1], sanitize=False)
            
            try:
                s1.UpdatePropertyCache()
            except:
               # print('ERROR')
                continue
            #print(s1)
            try:
                check = rxnrev.RunReactants((s1,))
                #print('CHECKED')
            except Exception as e:
                #print(e)
                continue
            found = False
            #print(check)
            for pp in check:
                for d in pp:
                    #print('ENTRY:{}, N: {}, Smiles: {}'.format(k, i, Chem.MolToSmiles(d)))
                    t1 = topology_from_rdkit(d)
                    t2 = topology_from_rdkit(mol)
                    if nx.is_isomorphic(t1,t2):
                        found = True
                        break
                if found:
                    break
                    
            if not found:
                #print('SANITY CHECK FAILED')
                continue
            rxn1 = []
            for j, pp in enumerate(p):
                try:
                    for a in pp.GetAtoms():
                        a.SetNumExplicitHs(0)
                    #print(Chem.MolToSmiles(pp))
                    #pp = Chem.MolFromSmiles(Chem.MolToSmiles(pp), sanitize=False)
                    #pp = Chem.RemoveAllHs(pp)
                    #pp.UpdatePropertyCache()
                    rxn1.extend(Chem.MolToSmiles(pp).split('.'))
                    #AllChem.Compute2DCoords(pp)
                    #Draw.MolToFile(pp, 'top{}/entry{}/{}.png'.format(k, i, j), kekulize=False)
                except KeyboardInterrupt:
                    exit()
                except:
                    #print('ERROR')
                    continue
            rxn1 = sorted(rxn1)
            rxn1 = tuple(rxn1)
            mols.add(rxn1)
    #print(top10)
    return mols
    
def retrosynthesis (smi, used, depth=0, lim=5, branching=1, write_tree=None):
    steps = []
    used.add(smi)
    #print(Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(smi, sanitize=False)))
    if write_tree is not None:
        write_tree.write(smi + '\n')
    else:
        print(smi)
    if depth > lim or Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(smi, sanitize=False)) < 100:
        return [smi], used
    results = retrosynthesis_step(smi, topn=branching)
    for i, potential in enumerate(results):
        rxn = []
        for p in potential:
            if p not in used:
                #found = True
                if write_tree is not None:
                    write_tree.write('\t'*(depth+1))
                else:
                    print('\t'*(depth+1), end='')
                ans, used = retrosynthesis(p, used, depth=depth+1, lim=lim, branching=branching, write_tree=write_tree)
                rxn.append(ans)
                #used.add(p)
        if i < len(results) - 1:
            if write_tree is not None:
                write_tree.write('\t'*(depth+1) + '---------------------------------\n')
            else:
                print('\t'*(depth+1) + '---------------------------------')
            #print('\t'*(depth+1) + '---------------------------------')
        steps.append(rxn)
    steps.append(smi)
        
    
    return steps[::], used
#print(retrosynthesis_step('CCC(COC(=O)[C@@H](NP(=O)(Oc1ccccc1)OC[C@H]1O[C@@]([C@@H]([C@@H]1O)O)(C#N)c1ccc2n1ncnc2N)C)CC', topn=20))
used = set()
#write_tree = open('OUTPUT_TREE', 'a')
retrosynthesis('C1C2=C(C=CC=C2)C2=C1C=CC=C2', used)
#write_tree.close()