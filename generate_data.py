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
from utils import *
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--keep', required=False,
                    help='Keep current dataset stored in NET_SET',
                    action='store_true', default=False)  
    ap.add_argument('-r', '--keepreactions', required=False,
                    help='Only use currently stored reaction list',
                    action='store_true', default=False)  
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    ap.add_argument('-n', '--number',
                    help='Number of occurences before reaction is defined as valid', required=False, type=int, default=10)
    args = vars(ap.parse_args())
    return args

args = get_arguments()


lg = rkl.logger()
lg.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')    

# TRAINING_PATH = './TRAINING_DATA/'
# # TEST_RATIO = 0.25

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

counter = 0
# all_reactions = set()
rxn_to_id = dict()
if not args['keep']:
    open(TRAINING_PATH + 'NET_SET', 'w').write('')
if args['keepreactions']:
    for l in open('TRAINING_DATA/REACTIONS'):
        rxn_hashed = HashedReaction(l.strip())
        # print(rxn_hashed.real_smarts)
        rxn_to_id[rxn_hashed] = counter
        counter += 1
else:
    open(TRAINING_PATH + 'REACTIONS', 'w').write('')


DATASET_DICT = defaultdict(list)
DATASETS = args['datasets']
if args['datasets'][0] == 'all':
    DATASETS = os.listdir('ord-data/data/')
    
for DATASET in DATASETS:
    ds = dataset_pb2.Dataset()
    DPATH = f'ord-data/data/{DATASET}/'
    for DFILE in os.listdir(DPATH):
        ds.ParseFromString(gzip.open(DPATH + DFILE, 'rb').read())
        for reaction in tqdm(ds.reactions):
            rxnstr = get_reaction_smiles(reaction)
            rxnstr = rxnstr.split(' |')[0]
            # print(rxnstr, '------------------')
            try:
                for preprocessed in preprocessing(rxnstr):
                    # print('POS', preprocessed)
                    try:
                        mini_rxn = Reactions.ReactionFromSmarts(preprocessed, useSmiles=True)
                        rxn_corestr = corify(preprocessed)
                        rxn_hashed = HashedReaction(rxn_corestr)
                        if rxn_hashed not in rxn_to_id:
                            if not args['keepreactions']:
                                rxn_to_id[rxn_hashed] = counter
                                counter += 1
                            else:
                                continue
                        fingerprint = product_fingerprint(mini_rxn)
                        DATASET_DICT[rxn_to_id[rxn_hashed]].append(pickle.dumps(fingerprint))
                    except KeyboardInterrupt:
                        import sys
                        sys.exit(0)
                    except Exception as e:
                        continue
            except KeyboardInterrupt:
                import sys
                sys.exit(0)
            except Exception as e:
                print('ISSUE')
                traceback.print_exc()
                continue
REACTIONS_FILE = open(TRAINING_PATH + 'REACTIONS', 'a')
NET_SET = open(TRAINING_PATH + 'NET_SET', 'ab')
REACTIONS = []
if not args['keepreactions']:
    FILTERED_DATASET = list(filter(lambda x: len(x[1]) >= args['number'], DATASET_DICT.items()))
    INDICES, DATA = zip(*FILTERED_DATASET) 
    index_remap = dict(zip(INDICES, range(len(INDICES))))
    new_reactions = [(index_remap[index], reaction.real_smarts) for reaction, index in rxn_to_id.items() if index in index_remap]
    new_reactions.sort()
    _, REACTIONS = zip(*new_reactions)
    REACTIONS = [rxn + '\n' for rxn in REACTIONS]
    NEW_DATASET = [case + b'|' + bytes(str(idx), encoding='utf-8') + b'\n' for idx, data in zip(range(len(INDICES)), DATA) for case in data]
else:
    NEW_DATASET = [case + b'|' + bytes(str(idx), encoding='utf-8') + b'\n' for idx, data in DATASET_DICT.items() for case in data]

REACTIONS_FILE.writelines(REACTIONS)
NET_SET.writelines(NEW_DATASET)

REACTIONS_FILE.close()
NET_SET.close()