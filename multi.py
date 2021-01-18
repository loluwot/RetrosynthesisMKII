#from xml.dom import minidom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem import rdmolops
import traceback
import csv
import networkx as nx
from pathlib import Path
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
from tqdm import tqdm
import datetime
import mmap
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

def isChanged (mol1, mol2, atom1, atom2, deg1, deg2):
    #props = ['Z', 'A', 'C', 'O', 'R', 'S', 'U']
    mol1.UpdatePropertyCache()
    mol2.UpdatePropertyCache()
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    Chem.GetSymmSSSR(mol1)
    Chem.GetSymmSSSR(mol2)
    changed = atom1.GetAtomicNum() == atom2.GetAtomicNum()
    changed &= atom1.GetIsAromatic() == atom2.GetIsAromatic()
    changed &= atom1.GetDegree() == atom2.GetDegree()
    changed &= atom1.GetFormalCharge() == atom2.GetFormalCharge()
    changed &= mol1.GetRingInfo().NumAtomRings(atom1.GetIdx()) == mol2.GetRingInfo().NumAtomRings(atom2.GetIdx())
    changed &= mol1.GetRingInfo().MinAtomRingSize(atom1.GetIdx()) == mol2.GetRingInfo().MinAtomRingSize(atom2.GetIdx())
    changed &= atom1.GetExplicitValence() == atom2.GetExplicitValence()
    changed &= atom1.GetImplicitValence() == atom2.GetImplicitValence()
    changed &= atom1.GetSmarts() == atom2.GetSmarts()
    changed &= bonds1 == bonds2
    #print(atom1.GetSmarts())
    #print(atom2.GetSmarts())
    #print(deg2[atom2.GetIdx()] - deg1[atom1.GetIdx()])
    return not changed

def clearmapping(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
def changedatoms(react):
    product = react.GetProductTemplate(0)
    producttop = topology_from_rdkit(product)
    degprod = closeness_centrality(producttop)
    mmap = atommap(product)
    datoms = [set() for _ in range(len(react.GetReactants()))] + [set()]
    for i, reactant in enumerate(react.GetReactants()):
        reacttop = topology_from_rdkit(reactant)
        degreact = closeness_centrality(reacttop)
        for atom in reactant.GetAtoms():

            if atom.GetAtomMapNum() != 0:
                val = atom.GetAtomMapNum()
                try:
                    inproduct = product.GetAtomWithIdx(mmap[val])
                except:
                    datoms[i].add(atom.GetIdx())
                    continue

                if isChanged(reactant, product, atom, inproduct, degreact, degprod):
                    datoms[i].add(atom.GetIdx())
                    datoms[-1].add(mmap[val])
            else:
                datoms[i].add(atom.GetIdx())

    return datoms

def coresfromdatoms (datoms, parsed_reaction):
    removal = [[] for _ in range(len(datoms))]
    for j in range(len(datoms[:-1])):
        rnt = parsed_reaction.GetReactantTemplate(j)
        for jj in range(rnt.GetNumAtoms()):
            if jj not in datoms[j]:
                removal[j].append(jj)
    pro = parsed_reaction.GetProductTemplate(0)
    for jj in range(pro.GetNumAtoms()):
        if jj not in datoms[-1]:
            removal[-1].append(jj)
    for j in range(len(removal)):
        removal[j] = removal[j][::-1]

    cores = []

    for j in range(len(datoms) - 1):
        rnt = parsed_reaction.GetReactantTemplate(j)
        core = Chem.EditableMol(rnt)
        for remove in removal[j]:
            core.RemoveAtom(remove)

        cores.append(core.GetMol())

    pro = parsed_reaction.GetProductTemplate(0)
    core = Chem.EditableMol(pro)
    for remove in removal[-1]:
        core.RemoveAtom(remove)
    cores.append(core.GetMol())

    return cores

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
                    #print("INTERRUPTED BY USER")
                    exit()
                except Exception as e:
                    #print(e)
                    #print(traceback.format_exc())
                    continue
        counter += len(reactant.GetAtoms())

def updatecore(datoms, reaction):
    product = reaction.GetProductTemplate(0)
    mmap = atommap(product)
    new_datoms = copy.deepcopy(datoms)
    for i, tup in enumerate(datoms[:-1]):
        rnt = reaction.GetReactantTemplate(i)
        for a in tup:
            at = rnt.GetAtomWithIdx(a)
            for neigh in at.GetNeighbors():
                if neigh.GetAtomMapNum() != 0:
                    id = neigh.GetAtomMapNum()
                    new_datoms[i].add(neigh.GetIdx())
                    try:
                        new_datoms[-1].add(mmap[id])
                    except KeyboardInterrupt:
                        #print("INTERRUPTED BY USER")
                        exit()
                    except Exception as e:
                        #print(e)
                        #print(traceback.format_exc())
                        continue
    return new_datoms

#classes = open('RESULTS')
#training = open('TRAINING/TRAIN_INPUT', 'a')
#traininglabels = open('TRAINING/TRAIN_LABELS', 'a')
#testing = open('TRAINING/TESTING', 'a')
#testinglabels = open('TRAINING/TESTING_LABELS', 'a')
f = open('DATA/1976_Sep2016_USPTOgrants_smiles/1976_Sep2016_USPTOgrants_smiles.rsmi', 'r+b')
#   table = csv.reader(f, delimiter='\t', quotechar='|')
#nlim = 1000000
#traininglim = 750000
core = []
parsed_reactions = []
reaction_rules = []
good_cores = []
cores_index = {}
unique_cores = defaultdict(int)
affprop = None
n3 = 0
bign = 15
TOTAL_CLASSES = 2000
ff = open('TRAINING/RESULTS', 'a')



mm = mmap.mmap(f.fileno(), 0)
fsize = Path('DATA/1976_Sep2016_USPTOgrants_smiles/1976_Sep2016_USPTOgrants_smiles.rsmi').stat().st_size

stime = []
#tot = 0
with tqdm(total=fsize) as pbar:
    for i, reaction in enumerate(iter(mm.readline, b'')):
        #start_time = time.time()
        try:
            #print(i)
            tot = len(reaction)
            
            reaction = reaction.decode('utf-8').split('\t')
            rsmiles = reaction[0].split(' |')[0]
            parsed_reaction = AllChem.ReactionFromSmarts(rsmiles, useSmiles=True)
            parsed_reaction.Initialize()
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
            
            for reactant in rlist:
                if (found):
                    break
                for product in plist:
                    #print(reactant, product)
                    if Chem.CanonSmiles(Chem.MolToSmiles(reactant)) == Chem.CanonSmiles(Chem.MolToSmiles(product)):
                        found = True
                        break
            #print(found)
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
            
            patt = Chem.MolFromSmiles(processed.split('>>')[0], sanitize=False)
            products = processed.split('>>')[1]
            #patt2.UpdatePropertyCache()
            patt2 = Chem.RemoveHs(patt, sanitize=False)
            for a in patt2.GetAtoms():
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(0)
                
            patt2 = Chem.AddHs(patt2)
            patt2 = Chem.MolToSmiles(patt2)
            
            processed = patt2 + '>>' + products
            
            print(processed)
            hashed_val = hashlib.md5(processed.encode())
            if (unique_cores[hashed_val.hexdigest()[:6]] == bign-1):
                n3 += 1
                #good_cores.append(processed)
                cores_index[processed] = n3 - 1
                ff.write(processed + '\n')
            unique_cores[hashed_val.hexdigest()[:6]] += 1
            pbar.update(tot)
            #print('Number of Unique: {}'.format(n3))
        except KeyboardInterrupt:
            #print("INTERRUPTED BY USER")
            exit()
        except Exception as e:
            #print(e)
            #print(traceback.format_exc())
            continue
        #end_time = time.time()
        #stime.append(end_time - start_time)
        #if (len(stime) > 20):
        #    del stime[0]
        #avg = sum(stime)/len(stime)
        #timeleft = (1000001-i)*avg
        #print("Approximate time left: {}".format(datetime.timedelta(seconds=timeleft)))
