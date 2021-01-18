#from xml.dom import minidom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem.rdChemReactions import *
import traceback
import csv
import networkx as nx
from networkx.algorithms.centrality import *
import copy
from generate_retro_templates import *
from sklearn.cluster import AffinityPropagation
import distance
import numpy as np
import os
from rdkit import RDLogger
from collections import defaultdict
import hashlib
import time
import datetime
import mmap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from operator import add
import pickle
RDLogger.DisableLog('rdApp.*')

def topology_from_rdkit(rdkit_molecule):

    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology

def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    # atoms = sorted([atom_to_label(bond.GetBeginAtom().GetIdx()), \
    #               atom_to_label(bond.GetEndAtom().GetIdx())])
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])
def atommap (mol):
    atomm = {}
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() != 0:
            atomm[atom.GetAtomMapNum()] = i
    return atomm

def clearmapping(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

def canonical_remap(rxn):
    rxn.Initialize()
    product = rxn.GetReactantTemplate(0)
    product.UpdatePropertyCache()
    mmap = atommap(product)
    counter = 1
    for reactant in rxn.GetProducts():
        reactant.UpdatePropertyCache()
        ranks = list(Chem.CanonicalRankAtoms(reactant, breakTies=True))
        #print(ranks)
        for i, atom in enumerate(reactant.GetAtoms()):
            val = atom.GetAtomMapNum()
            if val != 0:
                try:
                    product_atom = product.GetAtomWithIdx(mmap[val])
                    atom.ClearProp('molAtomMapNumber')
                    product_atom.ClearProp('molAtomMapNumber')
                    atom.SetIntProp('molAtomMapNumber', ranks[i] + counter)
                    product_atom.SetIntProp('molAtomMapNumber', ranks[i] + counter)
                except KeyError:
                    #print(atom.GetSymbol())
                    #print('bad atom')
                    continue
                except KeyboardInterrupt:
                    print("INTERRUPTED BY USER")
                    exit()
                except Exception as e:
                    #print(e)
                    #print(traceback.format_exc())
                    continue
        counter += len(reactant.GetAtoms())


#classes = open('RESULTS')
#training = open('TRAINING/TRAIN_INPUT', 'a')
#traininglabels = open('TRAINING/TRAIN_LABELS', 'a')
#testing = open('TRAINING/TESTING', 'a')
#testinglabels = open('TRAINING/TESTING_LABELS', 'a')

#   table = csv.reader(f, delimiter='\t', quotechar='|')
#nlim = 1000000
#traininglim = 750000
core = []
parsed_reactions = []
reaction_rules = []
good_cores = []
#cores_index = {}
unique_cores = defaultdict(int)
affprop = None
n3 = 0
bign = 15
TOTAL_CLASSES = 2000

def process_file(file):
    traininglim = 180894*3//4
    cores_index = {}
    ff = open('TRAINING/RESULTS')
    for index, l in enumerate(ff):
        cores_index[l[:-1]] = index
    f = open('DATA/1976_Sep2016_USPTOgrants_smiles/1976_Sep2016_USPTOgrants_smiles{}.rsmi'.format(file), 'r+b')
    mm = mmap.mmap(f.fileno(), 0)
    stime = []
    for i, reaction in enumerate(tqdm(iter(mm.readline, b''), total=180894, desc=str(file))):
        #print(i)
        start_time = time.time()
        try:
            #print(i)
            reaction = reaction.decode('utf-8').split('\t')
            rsmiles = reaction[0].split(' |')[0]
            parsed_reaction = AllChem.ReactionFromSmarts(rsmiles, useSmiles=True)
            parsed_reaction.Initialize()
            parsed_reaction.Validate()
            found = False
            rlist = []
            plist = []
            for reactant in parsed_reaction.GetReactants():
                rlist.append(rdmolops.RemoveHs(reactant))
            
            for product in parsed_reaction.GetProducts():
                plist.append(rdmolops.RemoveHs(product))
                
            parsed_reaction = ChemicalReaction()
            
            for reactant in rlist:
                parsed_reaction.AddReactantTemplate(reactant)
                
            for product in plist:
                parsed_reaction.AddProductTemplate(product)
            
            for reactant in parsed_reaction.GetReactants():
                if (found):
                    break
                for product in parsed_reaction.GetProducts():
                    #print(reactant, product)
                    if Chem.CanonSmiles(Chem.MolToSmiles(reactant)) == Chem.CanonSmiles(Chem.MolToSmiles(product)):
                        found = True
                        break
                        
            if found:
                continue
            parsed_reaction.RemoveUnmappedProductTemplates()
            parsed_reaction.RemoveUnmappedReactantTemplates()
            parsed_reaction.RemoveAgentTemplates()
            if len(parsed_reaction.GetProducts()) > 1:
                #print('Too many products.')
                continue
            
            rsmiles = AllChem.ReactionToSmiles(parsed_reaction)

            #print('preprocessing')
            processed = process_an_example(rsmiles)
            #print(processed)
            #print('processed')
            core_reaction = AllChem.ReactionFromSmarts(processed)
            canonical_remap(core_reaction)
            core_reaction.Initialize()
            processed = AllChem.ReactionToSmiles(core_reaction)
            if (processed in cores_index):
                fingies = [0 for _ in range(2048)]
                #compress = []
                for product in parsed_reaction.GetProducts():
                    product.UpdatePropertyCache()
                    Chem.GetSymmSSSR(product)
                    fing = AllChem.GetMorganFingerprintAsBitVect(product, 2)
                    fingies = list(map(add, fingies, fing))
                
                #for ii, v in enumerate(fingies):
                    #if not v == 0:
                        #compress.append(str(ii))
                        #compress.append(str(v))
                
                
                with open('TRAINING/TRAIN_PRE', 'ab') as training:
                    training.write(pickle.dumps(fingies) + b'|' + bytes(str(cores_index[processed]), encoding='utf8') + b'\n')
                #print('FOUND')
                
            #unique_cores[hashed_val.hexdigest()[:6]] += 1
            #print('Number of Unique: {}'.format(n3))
        except KeyboardInterrupt:
            print("INTERRUPTED BY USER")
            exit()
        except Exception as e:
            #print(e)
            #print(traceback.format_exc())
            continue
        end_time = time.time()
        stime.append(end_time - start_time)
        if (len(stime) > 20):
            del stime[0]
        avg = sum(stime)/len(stime)
        timeleft = (180894-i+1)*avg
        #print("Approximate time left: {}".format(datetime.timedelta(seconds=timeleft)))
        
files = [i for i in range(10)]
with Pool(10) as p:
    result = p.map(process_file, files)
   
#indexing   
with open('TRAINING/TRAIN_PRE', 'rb') as f:
    ff = open('TRAINING/TRAIN_INDEX', 'a')
    count = 0
    ff.write('{}\n'.format(count))
    for l in tqdm(f):
        #print(l, end='')
        count += len(l)
        ff.write('{}\n'.format(count))
    
    ff.close()
  
#scrambling  
import random
arr = []
with open('TRAINING/TRAIN_INDEX') as f:
    for l in f:
        arr.append(int(l))

random.shuffle(arr)
ff = open('TRAINING/TRAIN_PRE', 'rb')
f2 = open('TRAINING/TRAIN_SCRAMBLE', 'ab')
for a in tqdm(arr):
    ff.seek(a)
    ls = ff.readline()
    f2.write(ls)
    
f2.close()


