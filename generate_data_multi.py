from numpy.core.fromnumeric import argsort, trace
from ord_schema.proto import dataset_pb2
from ord_schema.proto import reaction_pb2 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import rdmolops
from tqdm import tqdm
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from generate_retro_templates import process_an_example
import pickle, gzip, os, random
import traceback
from mol_utils import *
import argparse
from collections import defaultdict
from multiprocessing import Manager, Lock, Pool
import multiprocessing
import functools
from joblib import Parallel, delayed
import time
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--keep', required=False,
                    help='Keep current dataset stored in NET_SET',
                    action='store_true', default=False)   
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    args = vars(ap.parse_args())
    return args

lg = rkl.logger()
lg.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')    

TRAINING_PATH = './TRAINING_DATA/'
# TEST_RATIO = 0.25

ROLE_TYPES = reaction_pb2.ReactionRole.ReactionRoleType
IDENTIFIER_TYPES = reaction_pb2.CompoundIdentifier.IdentifierType

def get_smiles(compound):
    for identifier in compound.identifiers:
        if identifier.type == IDENTIFIER_TYPES.SMILES:
            return identifier.value
    return -1

def get_reaction_smiles(reaction):
    for identifier in reaction.identifiers:
        if identifier.type == reaction_pb2.ReactionIdentifier.IdentifierType.REACTION_CXSMILES:
            return identifier.value
    return -1
    
def file_to_rxns(file):
    ds = dataset_pb2.Dataset()
    ds.ParseFromString(gzip.open(file, 'rb').read())
    return ds.reactions

args = get_arguments()

DATASETS = args['datasets']
if args['datasets'][0] == 'all':
    DATASETS = os.listdir('ord-data/data/')
print('Loading data...')
total_files = [f'ord-data/data/{DATASET}/{DFILE}' for DATASET in DATASETS for DFILE in os.listdir(f'ord-data/data/{DATASET}/')]
print('Loaded file names.')
print(total_files)
total_reactions = list(itertools.chain.from_iterable(map(file_to_rxns, total_files)))
print('Loaded all data into memory')

def process_rxn(reaction):
    DATASET_DICT = defaultdict(list)
    rxnstr = get_reaction_smiles(reaction)
    rxnstr = rxnstr.split(' |')[0]
    og_rxn = Reactions.ReactionFromSmarts(rxnstr, useSmiles=True)
    # print('ORIGINAL', rxnstr, '--------------------')
    for preprocessed in preprocessing(rxnstr):
        try:
            # print('PREPROCESSED', preprocessed)
            # start = time.time()
            rxn_corestr = corify(preprocessed)
            # print(time.time() - start)
            # print('PROCESSED', rxn_corestr)
            if rxn_corestr is None:
                continue
            rxn_hashed = HashedReaction(rxn_corestr)
            fingerprint = product_fingerprint(og_rxn)
            DATASET_DICT[rxn_hashed].append(pickle.dumps(fingerprint))
        except KeyboardInterrupt:
            import sys
            sys.exit(0)
        except:
            continue
    return DATASET_DICT

print('Loading reactions...')
counter = 0
rxn_to_id = dict()
for l in open('TRAINING_DATA/REACTIONS'):
    rxn_hashed = HashedReaction(l.strip())
    rxn_to_id[rxn_hashed] = counter
    counter += 1

num_cores = multiprocessing.cpu_count()
all_datapoints = Parallel(n_jobs=num_cores, verbose=3)(delayed(process_rxn)(i) for i in total_reactions)

print(len(all_datapoints))
start_time = time.time()
NET_SET = open(TRAINING_PATH + 'NET_SET', 'ab')
COLLECTIVE_DICT = defaultdict(list)
for datapoint in all_datapoints:
    for idx, v in datapoint.items():
        if idx in rxn_to_id:
            for case in v:
                NET_SET.write(case + b'|' + bytes(str(rxn_to_id[idx]), encoding='utf-8') + b'\n')
                
# DATASET_DICT = functools.reduce(lambda x, y: defaultdict(list, [(idx, x[idx] + y[idx]) for idx in set(y.keys()).union(set(x.keys())) if idx in rxn_to_id]), all_datapoints)
# NEW_DATASET = [case + b'|' + bytes(str(rxn_to_id[idx]), encoding='utf-8') + b'\n' for idx, data in DATASET_DICT.items() for case in data]
# print(NEW_DATASET)


print('TOOK:', time.time() - start_time)
