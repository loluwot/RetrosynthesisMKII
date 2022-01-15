from numpy.core.fromnumeric import prod
from numpy.lib.arraysetops import isin
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import rdmolops
from rdkit.Chem import SaltRemover
import networkx as nx
from generate_retro_templates import *
from networkx.algorithms.isomorphism import is_isomorphic
import numpy as np
import itertools
from rdkit import RDLogger
import traceback


RDLogger.DisableLog('rdApp.*')
remover = SaltRemover.SaltRemover()

class HashedReaction:
    def __init__(self, rxn_smarts, idx=0) -> None:
        reactants, products = rxn_smarts.split('>>')
        rxn_mols = Reactions.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        
        self.real_smarts = rxn_smarts
        self.idx = idx
        self.rxn_mols = canon_rxn(rxn_mols)
        self.rxn_smarts = Reactions.ReactionToSmarts(self.rxn_mols)
        reactants, products = rxn_smarts.split('>>')
        reactants_hash = str(hash(HashedMolSet(reactants)))
        products_hash = str(hash(HashedMolSet(products)))
        self.full_hash = reactants_hash + products_hash
    
    def __eq__(self, __o: object) -> bool:
        return __o.full_hash == self.full_hash
    
    def __ne__(self, __o: object) -> bool:
        return __o.full_hash != self.full_hash

    def __hash__(self) -> int:
        return hash(self.full_hash)

    def __str__(self) -> str:
        return self.rxn_smarts

class HashedMol:
    def __init__(self, molstr) -> None:
        self.molstr = molstr
        mol = Chem.MolFromSmiles(molstr, sanitize=False)
        self.cmol = canon_mol(mol)
        self.hash = hashed_molecule(self.cmol)
        pass

    def __eq__(self, __o: object) -> bool:
        return is_equivalent(self.cmol, __o.cmol)

    def __ne__(self, __o: object) -> bool:
        return not is_equivalent(self.cmol, __o.cmol)

    def __hash__(self) -> int:
        return hash(self.hash)

    def __str__(self) -> str:
        return self.molstr

class HashedMolSet:
    def __init__(self, molstr) -> None:
        self.mols = molstr.split('.')
        # print('MOLS', self.mols)
        self.canon_mols = list(set([HashedMol(x) for x in self.mols]))
        hash_sep = [x.hash for x in self.canon_mols]
        self.hash = hash(''.join(sorted(hash_sep)))

    def __eq__(self, __o: object) -> bool:
        return self.hash == __o.hash

    def __ne__(self, __o: object) -> bool:
        return self.hash != __o.__hash__
    
    def __hash__(self) -> int:
        return self.hash
    
    def __str__(self) -> str:
        return '.'.join(list(map(lambda x: x.molstr, self.canon_mols)))

    def update(self): #changed canon_mols
        self.mols = list(map(lambda x: x.molstr, self.canon_mols))
        hash_sep = [x.hash for x in self.canon_mols]
        self.hash = hash(''.join(sorted(hash_sep)))

def process_mol(mol):
    try:
        mol.UpdatePropertyCache()
        for a in mol.GetAtoms():
            a.SetNumRadicalElectrons(0)
            a.SetNumExplicitHs(0)
    except:
        return mol
    return mol

def canon_mol(mol):
    smi = Chem.MolToSmiles(mol)
    molcopy = Chem.MolFromSmiles(smi, sanitize=False)
    # for atom in molcopy.GetAtoms():
    #     atom.ClearProp('molAtomMapNumber')
    molcopy = rdmolops.RemoveHs(molcopy, sanitize=False)
    return molcopy
    
def canon_rxn(rxn):
    rlist = list(map(canon_mol, rxn.GetReactants()))
    plist = list(map(canon_mol, rxn.GetProducts()))
    parsed = Reactions.ChemicalReaction()
    for reactant in rlist:
        parsed.AddReactantTemplate(reactant)
    for product in plist:
        parsed.AddProductTemplate(product)
    parsed.Initialize()
    return parsed

def topology_from_rdkit(mol):
    topology = nx.Graph()
    for atom in mol.GetAtoms():
        anstr = f"{atom.GetAtomicNum()}{atom.GetProp('molAtomMapNumber')}" if atom.HasProp('molAtomMapNumber') else f"{atom.GetAtomicNum()}"
        topology.add_node(atom.GetIdx(), an=anstr)
        for bond in atom.GetBonds():
            topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=str(bond.GetBondTypeAsDouble()))
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

def clearmapping(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

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
                    continue
                except KeyboardInterrupt:
                    print("INTERRUPTED BY USER")
                    exit()
                except Exception as e:
                    continue
        counter += len(reactant.GetAtoms())

def hashed_molecule(mol1):
    return nx.weisfeiler_lehman_graph_hash(topology_from_rdkit(mol1), edge_attr='order', node_attr='an')

def is_equivalent(mol1, mol2):
    t1, t2 = topology_from_rdkit(mol1), topology_from_rdkit(mol2)
    return is_isomorphic(t1, t2, node_match=lambda x, y: x['an'] == y['an'], edge_match=lambda x, y: x['order'] == y['order'])


#REMEMBER PREPROCESSING CHANGED TO OUTPUT MULTIPLE REACTIONS IF MULTIPLE PRODUCTS

def preprocessing(rxnstr):
    try:
        rxn = Reactions.ReactionFromSmarts(rxnstr, useSmiles=True)
        rxn.RemoveUnmappedProductTemplates()
        rxn.RemoveUnmappedReactantTemplates()
        rxn.RemoveAgentTemplates()
        rxnsmi = Reactions.ReactionToSmiles(rxn)
        # print(rxnsmi, ':::::')
        reactants_str, products_str = rxnsmi.split('>>')
        reactants, products = [reactants for reactants in rxn.GetReactants()], [products for products in rxn.GetProducts()]
        
        #arbitrary atom count filter used to remove random remaining small molecules
        product_atom_count = sum(map(lambda x: x.GetNumAtoms(), products))
        if product_atom_count < 3:
            return []
        #removing trivial common elements (this eliminates reactions like alkene metathesis but that will be dealt with later)
        reactants_h, products_h = HashedMolSet(reactants_str), HashedMolSet(products_str)
        reactants_set, products_set = set(reactants_h.canon_mols), set(products_h.canon_mols)
        reactants_set, products_set = reactants_set.difference(products_set), products_set.difference(reactants_set)
        reactants_h.canon_mols, products_h.canon_mols = list(reactants_set), list(products_set)
        reactants_h.update()
        products_h.update()
        reactants_str = str(reactants_h)
        products_str = str(products_h)
        if len(reactants_str) == 0 or len(products_str) == 0:
            return []
        if '.' in products_str:
            product_list = products_str.split('.')
            return itertools.chain.from_iterable(map(lambda x: preprocessing(f'{reactants_str}>>{x}'), product_list)) #multiple product case, ignore
        return [f'{reactants_str}>>{products_str}'] 

    except KeyboardInterrupt:
        import sys
        sys.exit(0)
    except Exception as e:
        # traceback.print_exc()
        print('ISSUE')
        return []
    # parsed_reaction = Reactions.ChemicalReaction()
    # for reactant in reactants:
    #     parsed_reaction.AddReactantTemplate(reactant)
    # for product in products:
    #     parsed_reaction.AddProductTemplate(product)
    
    
    # parsed_reaction.RemoveAgentTemplates()

    # return Reactions.ReactionToSmarts(parsed_reaction)

def postprocessing(rxnstr):
    reactants, products = map(lambda x: Chem.MolToSmiles(process_mol(Chem.MolFromSmiles(x, sanitize=False))), rxnstr.split('>>'))
    reactants_h = HashedMolSet(reactants)
    products_h = HashedMolSet(products)
    reactants_set = set(reactants_h.canon_mols)
    products_set = set(products_h.canon_mols)
    reactants_set, products_set = reactants_set.difference(products_set), products_set.difference(reactants_set)
    reactants_h.canon_mols = list(reactants_set)
    products_h.canon_mols = list(products_set)
    reactants_h.update()
    products_h.update()
    reactants = str(reactants_h)
    products = str(products_h)
    if '.' in products:
        return None #multiple product case returns somehow, ignore
    return f'{reactants}>>{products}'

def smarts_to_smiles(smarts):
    rxn_core = Reactions.ReactionFromSmarts(smarts)
    canonical_remap(rxn_core)
    rxn_core.Initialize()
    rxn_corestr = Reactions.ReactionToSmiles(rxn_core)
    #print('PREPOSTPROCESSING2', rxn_corestr)
    #print('--------------------------------------------')
    rxn_corestr = postprocessing(rxn_corestr)
    return rxn_corestr

def corify(rxnstr, smarts=False, simple=False):
    try:
        processed = process_an_example(rxnstr, super_general=simple)
        #print('PREPOSTPROCESSING1', processed)
        rxn_corestr = smarts_to_smiles(processed)
        if smarts:
            return rxn_corestr, processed
        else:
            return rxn_corestr
    except KeyboardInterrupt:
        import sys
        sys.exit(0)
    except:
        # traceback.print_exc()
        if smarts:
            return None, None
        return None

# def sanity_check(rxnstr, mol):
#     rxn = Reactions.ReactionFromSmarts(rxnstr)
#     hashed_mol = HashedMol(Chem.MolToSmiles(mol))
#     reactants, products = map(lambda x: Chem.MolToSmiles(process_mol(Chem.MolFromSmiles(x, sanitize=False))), rxnstr.split('>>'))
#     if '.' in reactants:
#         return []
#     temp = [f'{reactants}>>({products})', f'({products})>>{reactants}']
#     prxn, prxnrev = map(Reactions.ReactionFromSmarts, temp)
#     try:
#         split_results = list(prxn.RunReactants((mol,)))
#     except:
#         return []
#     net_res = set()
#     for res in split_results:
#         new_results = prxnrev.RunReactants((res[0],))
#         if len(new_results) == 0:
#             continue
#         hashed_res = list(map(lambda x: HashedMol(Chem.MolToSmiles(x[0])), new_results))
#         if hashed_mol in hashed_res:
#             if Chem.MolFromSmiles(Chem.MolToSmiles(res[0])) != None:
#                 net_res.add(HashedMolSet(Chem.MolToSmiles(res[0])))
#     net_res = [x.mols for x in net_res]
#     return net_res

def fingerprint(mol_set, nbits=2048):
    net_fingerprint = [0]*nbits
    for mol in mol_set:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        finger = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        net_fingerprint = list(map(lambda x, y: x + y, net_fingerprint, finger))
    return net_fingerprint

def product_fingerprint(rxn, nbits=2048):
    if isinstance(rxn, str):
        return fingerprint([Chem.MolFromSmiles(x) for x in rxn.split('>>')[1].split('.')])
    return fingerprint([product for product in rxn.GetProducts()], nbits=nbits)

def reactant_fingerprint(rxn, nbits=2048):
    if isinstance(rxn, str):
        return fingerprint([Chem.MolFromSmiles(x) for x in rxn.split('>>')[0].split('.')])
    return fingerprint([reactant for reactant in rxn.GetReactants()], nbits=nbits)

def forward_run(rxnstr, reactants, use_smiles = True, one_product=False):
    reactants_s, products = rxnstr.split('>>')
    # # if products[0] == '(' and products[-1] == ')':
    # #     products = products[1:-1]
    # reactants_s, products = map(lambda x: Chem.MolToSmiles(process_mol(Chem.MolFromSmiles(x, sanitize=False))), [reactants_s, products])
    rxnstr_bracketed = f'({reactants_s})>>({products})' if one_product else f'({reactants_s})>>{products}' 
    rxn = Reactions.ReactionFromSmarts(rxnstr_bracketed, useSmiles=use_smiles)
    merged_reactants = Chem.MolFromSmiles('.'.join([Chem.MolToSmiles(x) for x in reactants]))
    try:
        split_results = list(rxn.RunReactant(merged_reactants, 0))
    except:
        traceback.print_exc()
        return []   
    return split_results    

def get_agents(rxnstr):
    try:
        rxn = Reactions.ReactionFromSmarts(rxnstr, useSmiles=True)
        rxn.RemoveUnmappedProductTemplates()
        rxn.RemoveUnmappedReactantTemplates()
        return [agent for agent in rxn.GetAgents()]
    except:
        return []

def reaction_fingerprint(reaction, agents, nbits=2048, weight=-1):
    p_f = product_fingerprint(reaction, nbits=nbits)
    r_f = reactant_fingerprint(reaction, nbits=nbits)
    a_f = fingerprint(agents, nbits=nbits)
    return [p_f[i] - r_f[i] + weight*a_f[i] for i in range(nbits)]
    # pass

def reaction_fingerprint_scratch(p_f, r_f, agents, nbits=2048, weight=-1):
    a_f = fingerprint(agents, nbits=nbits)
    return [p_f[i] - r_f[i] + weight*a_f[i] for i in range(nbits)]

# def get_fingerprint_from_str(s):
#     val, base = map(int, s.split())
#     def base_str(n,base):
#         convert = "0123456789"
#         if n < base:
#             return convert[n]
#         else:
#             return base_str(n//base,base) + convert[n%base]
#     return map(int, list(base_str(val, base).zfill(2048)))
