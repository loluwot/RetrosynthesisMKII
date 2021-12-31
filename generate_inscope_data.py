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
from utils import *
from joblib import Parallel, delayed
import multiprocessing
import math
import pandas as pd
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datasets',
                    help='Name of datasets to use', required=True,nargs="+", type=str)
    ap.add_argument('-n', '--batchsize',
                    help='Size of batched datasets', required=False,type=int, default=10)
    ap.add_argument('-b', '--bitsize',
                    help='Size of Morgan Fingerprint', required=False,type=int, default=2048)
    ap.add_argument('-a', '--nbatches',
                    help='Number of batches for dataset', required=False,type=int, default=10)
    args = vars(ap.parse_args())
    return args

args = get_arguments()

DATASETS = args['datasets']
if args['datasets'][0] == 'all':
    DATASETS = os.listdir('ord-data/data/')
print('Loading data...')
total_files = [f'ord-data/data/{DATASET}/{DFILE}' for DATASET in DATASETS for DFILE in os.listdir(f'ord-data/data/{DATASET}/')]
print('Loaded file names.')
# print(total_files)


def process_rxn(reaction):
    rxnstr = get_reaction_smiles(reaction) 
    data_points = []
    if isinstance(rxnstr, int):
        return data_points
    # print(rxnstr)
    rxnstr = rxnstr.split(' |')[0]
    for mini_rxn in preprocessing(rxnstr):
        og_rxn = Reactions.ReactionFromSmarts(mini_rxn, useSmiles=True)
        cored = corify(mini_rxn)
        if cored is None or len(list(filter(lambda x: len(x.strip()) != 0, cored.split('>>')))) < 2:
            continue
        try:
            r_finger = reactant_fingerprint(og_rxn)
            p_finger = product_fingerprint(og_rxn)
            potential_products = forward_run(cored, [reactant for reactant in og_rxn.GetReactants()])
            # print(len(potential_products))
            data_points += [b','.join([pickle.dumps(r_finger), pickle.dumps(fingerprint(p)), b'0']) + b'\n' for p in potential_products if fingerprint(p) != p_finger] + [b','.join([pickle.dumps(r_finger), pickle.dumps(p_finger), b'1']) + b'\n']
            
        except KeyboardInterrupt:
            import sys
            sys.exit(0)
        except:
            continue
    # print(type(data_points))
    return data_points

BATCH_SIZE = args['batchsize']
N_BATCHES = args['nbatches']

for i in range(N_BATCHES):
    open(TRAINING_PATH + f'INSCOPE_DATA{i}', 'w').write('')

for batchid, filenames in enumerate([total_files[i*BATCH_SIZE:min((i+1)*BATCH_SIZE, len(total_files))] for i in range(math.ceil(len(total_files)/BATCH_SIZE))]):
    total_reactions = list(itertools.chain.from_iterable(map(file_to_rxns, filenames)))
    # print(total_reactions)
    print(f'Loaded batch {batchid} into memory')
    #reactant fingerprint then product fingerprint => 1 or 0
    num_cores = multiprocessing.cpu_count()
    all_datapoints = Parallel(n_jobs=num_cores, verbose=3)(delayed(process_rxn)(i) for i in total_reactions)
    # print([type(datapoint) for datapoint in all_datapoints])
    datapoints = list(itertools.chain.from_iterable(all_datapoints))
    random.shuffle(datapoints)
    B_SIZE = (len(datapoints) - 1)//N_BATCHES + 1
    for i in range(N_BATCHES):
        INSCOPE_FILE = open(TRAINING_PATH + f'INSCOPE_DATA{i}', 'ab')
        INSCOPE_FILE.writelines(datapoints[B_SIZE*i : min(len(datapoints), B_SIZE*(i+1))])
        # # for datapoints in all_datapoints:
        # df = pd.DataFrame(datapoints)
        # df.to_csv(INSCOPE_FILE, header=False)
        # del datapoints
        INSCOPE_FILE.close()
    del datapoints