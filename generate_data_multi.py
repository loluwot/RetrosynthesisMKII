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
from utils import *
import math
import pandas
import csv
import base64
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--keep', required=False,
                    help='Keep current dataset stored in NET_SET',
                    action='store_true', default=False)   
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    ap.add_argument('-b', '--bitsize',
                    help='Size of Morgan Fingerprint', required=False,type=int, default=2048)
    ap.add_argument('-n', '--batchsize',
                    help='Size of batched datasets', required=False,type=int, default=10)
    ap.add_argument('-a', '--nbatches',
                    help='Number of batches for dataset', required=False,type=int, default=10)
    ap.add_argument('-s', '--simple',
                    help='Use simple templates', action='store_true',required=False, default=False)
    
    args = vars(ap.parse_args())
    return args

lg = rkl.logger()
lg.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')   

args = get_arguments()

FINGERPRINT_SIZE = args['bitsize']
N_BATCHES = args['nbatches']
DATASETS = args['datasets']
if args['datasets'][0] == 'all':
    DATASETS = os.listdir('ord-data/data/')
print('Loading data...')
total_files = [f'ord-data/data/{DATASET}/{DFILE}' for DATASET in DATASETS for DFILE in os.listdir(f'ord-data/data/{DATASET}/')]
print('Loaded file names.')
# print(total_files)
for i in range(N_BATCHES):
    open(TRAINING_PATH + f'NET_SET{i}', 'wb').write(b'')
print('Cleared file.')
print('Loading reactions...')
counter = 0
rxn_to_id = dict()
for l in open(f'TRAINING_DATA/REACTIONS{"_SIMPLE" if args["simple"] else ""}'):
    rxn_hashed = l.strip()
    rxn_to_id[rxn_hashed] = counter
    counter += 1
print('Loaded reactions.')

def process_rxn(reaction):
    DATASET_DICT = defaultdict(list)
    rxnstr = get_reaction_smiles(reaction)
    if rxnstr == -1:
        return DATASET_DICT
    rxnstr = rxnstr.split(' |')[0]
    # print('ORIGINAL', rxnstr, '--------------------')
    for preprocessed in preprocessing(rxnstr):
        try:
            og_rxn = Reactions.ReactionFromSmarts(preprocessed, useSmiles=True)
            # print()
            # start = time.time()
            rxn_corestr, rxn_smarts = corify(preprocessed, smarts=True, simple=args['simple'])
            # print('PREPROCESSED', preprocessed,'CORE_STR', rxn_corestr)
            # print(time.time() - start)
            # print('PROCESSED', rxn_corestr)
            if rxn_corestr is None:
                continue
            # rxn_hashed = HashedReaction(rxn_corestr)
            fingerprint = product_fingerprint(og_rxn, nbits=FINGERPRINT_SIZE)
            DATASET_DICT[rxn_smarts].append(pickle.dumps(fingerprint))
        except KeyboardInterrupt:
            import sys
            sys.exit(0)
        except:
            traceback.print_exc()
            continue
    return DATASET_DICT

BATCH_SIZE = args['batchsize']

num_cores = multiprocessing.cpu_count()
NET_SETS = [csv.writer(open(TRAINING_PATH + f'NET_SET{i}', 'a'), delimiter='\t') for i in range(N_BATCHES)]
for batchid, filenames in enumerate([total_files[i*BATCH_SIZE:min((i+1)*BATCH_SIZE, len(total_files))] for i in range(math.ceil(len(total_files)/BATCH_SIZE))]):
    total_reactions = list(itertools.chain.from_iterable(map(file_to_rxns, filenames)))
    print(f'Loaded batch {batchid} into memory')
    all_datapoints = Parallel(n_jobs=num_cores, verbose=3)(delayed(process_rxn)(i) for i in total_reactions)
    # print(len(all_datapoints))
    start_time = time.time()
    COLLECTIVE_DICT = defaultdict(list)
    counter = 0
    random.shuffle(all_datapoints)
    for datapoint in all_datapoints:
        for idx, v in datapoint.items():
            if idx in rxn_to_id:
                for case in v:
                    b64case = base64.b64encode(case).decode('utf-8')
                    NET_SETS[counter % N_BATCHES].writerow([b64case, str(rxn_to_id[idx])])
                    counter += 1
# DATASET_DICT = functools.reduce(lambda x, y: defaultdict(list, [(idx, x[idx] + y[idx]) for idx in set(y.keys()).union(set(x.keys())) if idx in rxn_to_id]), all_datapoints)
# NEW_DATASET = [case + b'|' + bytes(str(rxn_to_id[idx]), encoding='utf-8') + b'\n' for idx, data in DATASET_DICT.items() for case in data]
# print(NEW_DATASET)
print('TOOK:', time.time() - start_time)
