#assume state is [(react, id), ....]
#id is prefix + 1,2,3,4....
#if 2 ids have same prefix, they are reactants to the same reaction
#include map of id to reactions based on current state actions?

#state[0] is list of reactions
#state[1] is map of id to reactions
#state[2] is used?


from operator import pos

from numpy.lib.type_check import real
from standalone_model_numpy import SCScorer
import math
# from testing import retrosynthesis_step
from model_utils import *
import random
from copy import deepcopy
from rdkit.Chem import AllChem

THRESHOLD = 2
DEPTH_LIMIT = 10
scorer = SCScorer()
scorer.restore('./full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz')

def is_terminal(node):
    # print(node.state)
    reactants = list(zip(*node.state[0]))[0]
    # print(node.state[0])
    return all([scorer.get_score_from_smi(r)[1] < THRESHOLD or len(get_top_n(r)) == 0 for r in reactants])
    
class Node:
    def __init__(self, state, parent, step_size=0.08, depth=0) -> None:
        self.state = state
        print('LEGALITY CHECK', self.state)
        self.parent = parent
        self.children = []
        self.explored = set()
        self.n = 0
        self.depth = depth
        self.v = 0
        self.step_size = step_size
    
    def score(self,c=1):
        tup = list(zip(*self.state[0]))[0]
        return math.inf if self.n == 0 else self.v/self.n + c*math.sqrt(math.log(self.parent.n)/self.n)  

    def back_prop(self, counter):
        self.n += 1
        self.v -= counter
        if self.parent is not None:
            self.parent.back_prop(counter)

    def select(self):
        if is_terminal(self) or len(self.children) == 0:
            return self
        else:
            return self.ucb_select().select()
        
    def ucb_select(self):
        return max(self.children, key=lambda x: x.score())

    def simulate(self):
        if is_terminal(self):
            self.back_prop(self.depth)
            return self.depth
        return self.random_action().simulate()
    
    def random_action(self):
        while True:
            try:
                random_reactant = random.randint(0, len(self.state[0]) - 1)
                rndrct = self.state[0][random_reactant][0]
                if scorer.get_score_from_smi(rndrct)[1] < THRESHOLD:
                    print('SCORE', scorer.get_score_from_smi(rndrct))
                    continue
                temp_state = deepcopy(self.state[0])
                temp_state1 = deepcopy(self.state[1])
                reactsmi, prefix = temp_state.pop(random_reactant) #remove reactant
                print('TEMP STATE', temp_state)
                possibilities = get_top_n(self.state[0][random_reactant][0])
                if len(possibilities) == 0:
                    for i, r in enumerate(temp_state[::-1]):
                        real_i = len(temp_state) - 1 - i
                        if prefix[:-1][:len(r) - 1] == r[:-1]:
                            x, y = temp_state.pop(real_i)
                            print(x, y, 'REMOVED FOR PROXIMITY TO TERMINAL')
                    new_state = [temp_state, temp_state1]
                    self.state = new_state
                    continue
                    # return Node(new_state, self, depth=self.depth+1)  
                random_possibility = random.randint(0, len(possibilities) - 1)
                # temp_state = deepcopy(self.state[0])
                # temp_state1 = deepcopy(self.state[1])
                # reactsmi, prefix = temp_state.pop(random_reactant) #remove reactant
                # print('TEMP STATE', temp_state)
                new_reactants, reaction_id = possibilities[random_possibility]
                print('RANDOM POSSIBILITY', possibilities[random_possibility])
                random_set = random.choice(new_reactants)
                print('RANDOM SET', random_set)
                for k, new_reactant in enumerate(random_set):
                    temp_state.append((new_reactant, prefix + (k, )))
                print('NEW STATE', temp_state)
                print('--------------')
                temp_state1[prefix] = reaction_id
                new_state = [temp_state, temp_state1]
                return Node(new_state, self, depth=self.depth+1)  
            except KeyboardInterrupt:
                return
            except:
                print('ERROR')
                import traceback
                traceback.print_exc()
                continue

    def expand(self):
        if is_terminal(self) or self.depth > DEPTH_LIMIT:
            return 
        for i in range(len(self.state[0])):
            reactant = self.state[0][i][0]
            possibilities = get_top_n(reactant)
            for j in range(len(possibilities)):
                temp_state = deepcopy(self.state[0])
                temp_state1 = deepcopy(self.state[1])
                reactsmi, prefix = temp_state.pop(i) #remove reactant
                new_reactants, reaction_id = possibilities[j]
                for l in range(len(new_reactants)):
                    if (i, j, l) not in self.explored:
                        for k, new_reactant in enumerate(new_reactants[l]):
                            temp_state.append((new_reactant, prefix + (k,)))
                        temp_state1[prefix] = reaction_id
                        new_state = [temp_state, temp_state1]
                        new_node = Node(new_state, self, depth=self.depth + 1)   
                        self.children.append(new_node)
                        self.explored.add((i, j, l))
                        break
                else:
                    continue
                break
            else:
                continue
            break

    def run(self):
        potential = self.select()
        potential.expand()
        potential.simulate()
        
