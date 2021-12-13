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
import functools

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
    ap.add_argument('-p', '--processes',help='Number of processes to use for multiprocessing', required=False, type=int, default=4)
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
    
def f(DATASET, counter, rxn_to_id, lock, args):
    # print(list(map(lambda x: x.real_smarts, list(rxn_to_id.keys())))[:10])
    DATASET_DICT = defaultdict(list)
    ds = dataset_pb2.Dataset()
    DPATH = f'ord-data/data/{DATASET}/'
    for DFILE in os.listdir(DPATH):
        ds.ParseFromString(gzip.open(DPATH + DFILE, 'rb').read())
        for reaction in tqdm(ds.reactions):
            try:
                rxnstr = get_reaction_smiles(reaction)
                rxnstr = rxnstr.split(' |')[0]
                og_rxn = Reactions.ReactionFromSmarts(rxnstr, useSmiles=True)
                preprocessed = preprocessing(rxnstr)
                processed = process_an_example(preprocessed)
                # print('PREPROCESSED', preprocessed)
                rxn_core = Reactions.ReactionFromSmarts(processed)
                canonical_remap(rxn_core)
                rxn_core.Initialize()
                rxn_corestr = Reactions.ReactionToSmiles(rxn_core)
                rxn_corestr = postprocessing(rxn_corestr)
                rxn_hashed = HashedReaction(rxn_corestr)
                if rxn_hashed not in rxn_to_id:
                    if not args['keepreactions']:
                        rxn_to_id[rxn_hashed] = counter.value
                        with lock:
                            counter.value += 1
                    else:
                        continue
                fingerprint = product_fingerprint(og_rxn)
                DATASET_DICT[rxn_to_id[rxn_hashed]].append(pickle.dumps(fingerprint))
            except KeyboardInterrupt:
                break
            except Exception as e:
                # traceback.print_exc()
                # print('ISSUE')
                continue
    return DATASET_DICT
if __name__ == '__main__':
    args = get_arguments()
    pool = Pool(args['processes'])
    mgr = Manager()
    counter = mgr.Value('x', 0)
    rxn_to_id = mgr.dict()
    lock = mgr.Lock()
    if not args['keep']:
        open(TRAINING_PATH + 'NET_SET', 'w').write('')
    if args['keepreactions']:
        rxn_to_id = dict()
        counter = 0
        for l in open('TRAINING_DATA/REACTIONS'):
            rxn_hashed = HashedReaction(l.strip())
            # print(rxn_hashed.real_smarts)
            rxn_to_id[rxn_hashed] = counter
            counter += 1
    else:
        open(TRAINING_PATH + 'REACTIONS', 'w').write('')
    #print(rxn_to_id)
    # input()
    # DATASET_DICT = mgr.dict()
    DATASETS = args['datasets']
    if args['datasets'][0] == 'all':
        DATASETS = os.listdir('ord-data/data/')

    L_DATASET_DICT = pool.map(functools.partial(f, counter=counter, rxn_to_id=rxn_to_id, lock=lock, args=args), DATASETS)
    DATASET_DICT = functools.reduce(lambda x, y: defaultdict(list, [(idx, x[idx] + y[idx]) for idx in set(y.keys()).union(set(x.keys()))]), L_DATASET_DICT)
    # print(DATASET_DICT.keys())

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