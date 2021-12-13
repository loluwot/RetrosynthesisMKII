from rdkit.Chem import rdChemReactions as Reactions
from mol_utils import *
AA = '[O:1]=[C:2][c:3]>>[O:1][C:2][c:3]'
test_rxn = 'O=[C:1]([C:2])[O:3].[N:4][N:5][C:6]=[O:7]>>([C:1]([C:2])(=[O:3])[N:4][N:5][C:6]=[O:7])'
rxn_to_id = dict()
counter = 0
for l in open('TRAINING_DATA/REACTIONS'):
    rxn_hashed = HashedReaction(l.strip())
    # print(rxn_hashed.real_smarts)
    rxn_to_id[rxn_hashed] = counter
    counter += 1

print(HashedReaction(AA) in rxn_to_id)