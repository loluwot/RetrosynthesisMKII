#from xml.dom import minidom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
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
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
f = open('DATA/1976_Sep2016_USPTOgrants_smiles/1976_Sep2016_USPTOgrants_smiles.rsmi')
table = csv.reader(f, delimiter='\t', quotechar='|')
nlim = 1000000
traininglim = 750000
for i, reaction in tqdm(enumerate(table)):
    try:
        #print(i)
        rsmiles = reaction[0].split(' |')[0]
        with open('DATA/1976_Sep2016_USPTOgrants_smiles/1976_Sep2016_USPTOgrants_smiles{}.rsmi'.format(i % 10), 'a') as ff:
            ff.write(rsmiles + '\n')
        
    except KeyboardInterrupt:
        print("INTERRUPTED BY USER")
        exit()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        continue