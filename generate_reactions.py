from numpy.core.fromnumeric import trace
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
import time
from joblib import Parallel, delayed
import multiprocessing
import itertools
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    ap.add_argument('-n', '--number',
                    help='Number of occurences before reaction is defined as valid', required=False, type=int, default=10)
    args = vars(ap.parse_args())
    return args

TRAINING_PATH = './TRAINING_DATA/'

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

open(TRAINING_PATH + 'REACTIONS', 'w').write('')

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
    rxnstr = get_reaction_smiles(reaction)
    if isinstance(rxnstr, int):
        return []
    rxnstr = rxnstr.split(' |')[0]
    rxns = []
    for preprocessed in preprocessing(rxnstr):
        # print('PREPROCESSED', preprocessed, '-------------------')
        try:
            # start_time = time.time()
            processed = process_an_example(preprocessed)
            # print('PROCESSED', processed)
            # print(time.time() - start_time)
            rxn_core = Reactions.ReactionFromSmarts(processed)
            canonical_remap(rxn_core)
            rxn_core.Initialize()
            rxn_corestr = Reactions.ReactionToSmiles(rxn_core)
            rxn_corestr = postprocessing(rxn_corestr)
            rxn_hashed = HashedReaction(rxn_corestr)
            # print('PROCESSED', rxn_corestr)
            rxns.append(rxn_hashed)
        except KeyboardInterrupt:
            import sys
            sys.exit(0)
        except Exception as e:
            continue
    return rxns

DATASET_DICT = defaultdict(int)
num_cores = multiprocessing.cpu_count()
all_canon = Parallel(n_jobs=num_cores, verbose=3)(delayed(process_rxn)(i) for i in total_reactions)
for all_canon_l in all_canon:
    for rxn in all_canon_l:
        DATASET_DICT[rxn] += 1

FILTERED_RXNS = list(filter(lambda x: DATASET_DICT[x] >= args['number'], DATASET_DICT.keys()))
REACTIONS = map(lambda x: x.real_smarts, FILTERED_RXNS)
REACTIONS = [rxn + '\n' for rxn in REACTIONS]

REACTIONS_FILE = open(TRAINING_PATH + 'REACTIONS', 'w')
REACTIONS_FILE.writelines(REACTIONS)
