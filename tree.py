#assume state is [(react, id), ....]
#id is prefix + 1,2,3,4....
#if 2 ids have same prefix, they are reactants to the same reaction
#include map of id to reactions based on current state actions?

#state[0] is list of reactants
#state[1] is map of id to reactions
#state[2] is used?

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from hashlib import new
from operator import pos
from standalone_model_numpy import SCScorer
import math
# from testing import retrosynthesis_step
import matplotlib.pyplot as plt
from model_utils import *
import random
from copy import deepcopy
from rdkit.Chem import AllChem
from mol_utils import *
from collections import defaultdict
import time
import networkx as nx
THRESHOLD = 2
DEPTH_LIMIT = 5
scorer = SCScorer()
scorer.restore('./full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz')
EXPLORATION_CONSTANT = 3
SOLVE_REWARD = 10
EXPANSION_WIDTH = 15
ROLLOUT_WIDTH = 15
MAX_DEPTH = 25
DAMPENING = 0.99

def weighted_avg(vals, weights):
    # print('WORST', vals[1])
    return sum(map(lambda x, y: x*y, vals, weights))/sum(weights)

class MolNode:
    def __init__(self, molstr, parent=(None, None)) -> None: #parent is (parent_mol, rxn_id)
        self.hashed = HashedMol(molstr)
        self.molstr = molstr
        self.parent_mol, self.parent_rxn = parent
        self.real_mol = Chem.MolFromSmiles(molstr)
    
    def is_solved(self):
        score = scorer.get_score_from_smi(self.molstr)[1]
        return score < THRESHOLD

    def __hash__(self):
        return hash(self.hashed)

class State:
    def __init__(self, molecules, action_scores=None, action_counts = None, parent=(None, None), depth=0) -> None: #parent is (parent_state, parent_action)
        self.molecules = set(molecules) if isinstance(molecules, list) else molecules #molecules is set of MolNodes
        self.action_scores = defaultdict(int) if action_scores is None else action_scores
        self.action_counts = defaultdict(int) if action_counts is None else action_counts
        self.parent_state, self.parent_action = parent
        self.children = dict()
        self.actions = None
        self.G = None
        self.depth = depth

    def __str__(self) -> str:
        return ' '.join([mol.molstr for mol in self.molecules])

    #"dead end" terminal occurs when ANY molecule is a dead end
    def is_terminal(self): #1 if "solved" terminal, -1 if dead end terminal, fraction of solved molecules if not terminal
        actions = self.get_actions()
        counts = defaultdict(int)
        for action in actions:
            counts[action[1]] += 1
        count = 0
        for mol in self.molecules: 
            if mol.is_solved():
                count += 1
            elif counts[mol] == 0:
                return -1
            # if not mol.is_solved() and counts[mol] == 0:
            #     return -1
            # count += 1
        # print('SCORED', count/len(self.molecules)/(5 - THRESHOLD))
        # return weighted_avg([count/len(self.molecules), min([min(1, (5-scorer.get_score_from_smi(mol.molstr)[1])/(5 - THRESHOLD)) for mol in self.molecules])], [1, 1])
        return count/len(self.molecules)
        # return count/len(self.molecules) + min([min(1, (5-scorer.get_score_from_smi(mol.molstr)[1])/(5 - THRESHOLD)) for mol in self.molecules])

    def is_leaf(self):
        return len(list(self.children.keys())) == 0

    def do_action(self, action, virtual=False): #action is (((rxn_id, res_id), prob), mol), mol is MolNode
        if action in self.children:
            return self.children[action]
        ((rxn_id, res_id), prob), mol = action
        new_molecules = self.molecules.copy()
        new_molecules.remove(mol)
        # new_res, _ = sanity_check(rxn_id, mol.real_mol)[res_id]
        # print(self.res_map[action])
        new_res = self.res_map[action]
        # print(new_res)
        for res in new_res:
            new_mol = MolNode(res, parent=(mol, rxn_id))
            new_molecules.add(new_mol)
        new_state = State(new_molecules, parent=(self, action), depth=self.depth+1)
        if not virtual:
            self.children[action] = new_state
        return new_state

    def score_action(self, action):
        if self.action_counts[action] == 0:
            return math.inf
        ((rxn_id, res_id), prob), mol = action
        last_action = 1 if self.parent_state is None else self.parent_state.action_counts[self.parent_action]
        return self.action_scores[action]/self.action_counts[action] + EXPLORATION_CONSTANT*prob*math.sqrt(last_action)/(1 + self.action_counts[action])

    def get_actions(self):
        if self.actions is not None:
            return self.actions
        net_actions = []
        net_res_map = []
        for mol in self.molecules:
            if mol.is_solved():
                continue
            actions = get_top_n(mol.molstr, topn=EXPANSION_WIDTH)
            if len(actions) == 0:
                continue
            actions, results = zip(*actions)
            # print(results)
            actions = list(zip(actions, [mol for _ in actions]))
            net_res_map.extend(zip(actions, results))
            net_actions.extend(actions)
        self.res_map = dict(net_res_map)
        self.actions = net_actions
        return net_actions
        # pass

    #MCTS
    def select(self):
        if self.is_leaf() or abs(self.is_terminal()) == 1 or self.depth > MAX_DEPTH:
            return self
        return self.children[max(list(self.children.keys()), key=lambda x: self.score_action(x))].select()

    def expand(self):
        if abs(self.is_terminal()) == 1 or self.depth > MAX_DEPTH:
            return self
        print('EXPANDING IS LEAF: ', self.is_leaf())
        print('EXPANDING TERMINALITY: ', self.is_terminal())
        actions = self.get_actions()
        for action in actions:
            self.do_action(action)
        return self.children[max(actions, key=lambda x: x[0][1])]

    def rollout(self, depth=0):
        terminality = self.is_terminal()
        if abs(terminality) == 1 or self.depth > MAX_DEPTH:
            return terminality*SOLVE_REWARD if terminality > 0 else terminality, self
        if depth >= DEPTH_LIMIT:
            return terminality, self
        new_state = self.do_action(random.choice(self.get_actions()[:ROLLOUT_WIDTH]), virtual=True)
        return new_state.rollout(depth=depth+1)

    def get_branch_props(self, length=0, confidence=0):
        if self.parent_state is None:
            return length, confidence
        return self.parent_state.get_branch_props(length=length+1, confidence=DAMPENING*confidence + self.parent_action[0][1])
        # return parent_l + 1, parent_val*DAMPENING + self.parent_action[0][1]

    def backprop(self, reward):
        if self.parent_state is None:
            return
        cur_count = self.parent_state.action_counts[self.parent_action]
        cur_value = self.parent_state.action_scores[self.parent_action]
        self.parent_state.action_scores[self.parent_action] = (cur_value*cur_count + reward)/(cur_count + 1)
        self.parent_state.action_counts[self.parent_action] += 1
        self.parent_state.backprop(reward)
        
    def run(self):
        best = self.select()
        print(f'SELECTED: {str(best)}')
        leaf = best.expand()
        reward, term = leaf.rollout()
        print(f'ROLLOUT: ', str(term))
        length, confidence = term.get_branch_props()
        temp = length - confidence
        weight = max(0, (MAX_DEPTH - temp)/MAX_DEPTH)
        print(f'FINAL REWARD: ', reward*weight)
        leaf.backprop(reward*weight)

    def get_synthesis(self):
        best = self.select()
        print('IS LEAF?', best.is_leaf())
        if best.is_terminal() == 1:
            print('SOLVED')
            return best
        return best
        
    def create_tree(self, graph=None):
        if graph is None:
            self.G = nx.DiGraph()
            graph = self.G    
        graph.add_node(str(self))
        if self.parent_state is not None:
            graph.add_edge(str(self.parent_state), str(self))
        for children in self.children.values():
            children.create_tree(graph=graph)
    
    def draw_tree(self, filename='tree.png'):
        if self.G is None:
            self.create_tree()
        pos = graphviz_layout(self.G, prog='dot')
        nx.draw(self.G, pos)
        fig = plt.gcf()
        fig.set_size_inches(100, 30)
        fig.savefig(filename, dpi=200)






