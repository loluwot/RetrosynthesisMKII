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
from utils import *
RDLogger.DisableLog('rdApp.*')

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    ap.add_argument('-n', '--number',
                    help='Number of occurences before reaction is defined as valid', required=False, type=int, default=10)
    ap.add_argument('-k', '--keep',
                    help='Keep current reactions', action='store_true',required=False, default=False)
    args = vars(ap.parse_args())
    return args

# TRAINING_PATH = './TRAINING_DATA/'

# ROLE_TYPES = reaction_pb2.ReactionRole.ReactionRoleType
# IDENTIFIER_TYPES = reaction_pb2.CompoundIdentifier.IdentifierType

# def get_smiles(compound):
#     for identifier in compound.identifiers:
#         if identifier.type == IDENTIFIER_TYPES.SMILES:
#             return identifier.value
#     return -1

# def get_reaction_smiles(reaction):
#     for identifier in reaction.identifiers:
#         if identifier.type == reaction_pb2.ReactionIdentifier.IdentifierType.REACTION_CXSMILES:
#             return identifier.value
#     return -1




args = get_arguments()


DATASETS = args['datasets']
if args['datasets'][0] == 'all':
    DATASETS = os.listdir('ord-data/data/')
print('Loading data...')
total_files = [f'ord-data/data/{DATASET}/{DFILE}' for DATASET in DATASETS for DFILE in os.listdir(f'ord-data/data/{DATASET}/')]
print('Loaded file names.')
print(total_files)
def total_reactions(file_names):
    for file in file_names:
        frxn = file_to_rxns(file, additional_info=True)
        for rxn in frxn:
            yield rxn
        del frxn






# total_reactions = list(itertools.chain.from_iterable(map(lambda x: file_to_rxns(x, additional_info=True), total_files)))
# print('Loaded all data into memory')

def process_rxn(reaction):
    reaction, *additional_info = reaction
    rxnstr = get_reaction_smiles(reaction)
    if isinstance(rxnstr, int):
        return []
    rxnstr = rxnstr.split(' |')[0]
    rxns = []
    for preprocessed in preprocessing(rxnstr):
        # print('PREPROCESSED', preprocessed, '-------------------')
        try:
            rxn_corestr, smarts_str = corify(preprocessed, smarts=True)
            if len(list(filter(lambda x: len(x.strip()) != 0, rxn_corestr.split('>>')))) < 2:
                #print('INCOMPLETE', preprocessed, rxn_corestr)
                continue
            # rxn_hashed = HashedReaction(rxn_corestr)
            # rxn_hashed.real_smarts = smarts_str
            # print('PROCESSED', rxn_corestr)
            rxns.append((smarts_str,) + tuple(additional_info))
        except KeyboardInterrupt:
            import sys
            sys.exit(0)
        except Exception as e:
            # traceback.print_exc()
            continue
    return rxns


DATASET_DICT = defaultdict(int)
num_cores = multiprocessing.cpu_count()
all_canon = Parallel(n_jobs=num_cores, verbose=3, pre_dispatch=100000)(delayed(process_rxn)(i) for i in total_reactions(total_files))
EXAMPLE_REACTION = dict()
disallowed = set()
KEEP = args['keep']

if KEEP:
    for l in open(TRAINING_PATH + 'REACTIONS', 'r'):
        disallowed.add(l.strip())
else:
    open(TRAINING_PATH + 'REACTIONS', 'w').write('')
    open(TRAINING_PATH + 'REACTIONS_ADDITIONAL', 'w').write('')
    

for all_canon_l in all_canon:
    for rxn, *additional_info in all_canon_l:
        if not KEEP or rxn not in disallowed:
            DATASET_DICT[rxn] += 1
            if rxn not in EXAMPLE_REACTION:
                EXAMPLE_REACTION[rxn] = list(map(str, additional_info))

FILTERED_RXNS = list(filter(lambda x: DATASET_DICT[x] >= args['number'], DATASET_DICT.keys()))
# REACTIONS = map(lambda x: x.real_smarts+'\n', FILTERED_RXNS)
REACTIONS = map(lambda x: x +'\n', FILTERED_RXNS)
ADDITIONAL = map(lambda x: ','.join(EXAMPLE_REACTION[x]) + '\n', FILTERED_RXNS)

# REACTIONS = [rxn + '\n' for rxn in REACTIONS]

REACTIONS_FILE = open(TRAINING_PATH + 'REACTIONS', 'a')
REACTIONS_FILE.writelines(REACTIONS)
ADDITIONAL_FILE = open(TRAINING_PATH + 'REACTIONS_ADDITIONAL', 'a')
ADDITIONAL_FILE.writelines(ADDITIONAL)