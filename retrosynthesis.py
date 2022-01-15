from tree import *
import pickle
import heapq
root = State([MolNode('CC(C)(C)OC(=O)N1CCN(CC1)C1=CC(NS(=O)(=O)C2=CC=CC=C2F)=CC2=C1OC1(CCC1)CC2')])
# root = pickle.load(open('tree2000.pkl', 'rb'))
# root = State([MolNode(x) for x in 'CC(C)(C)OC(=O)N1CCNCC1 Nc1cc(Br)c2c(c1)CCC1(CCC1)O2 O=S(=O)(Cl)c1ccccc1F'.split(' ')])
# root = State([MolNode('[O-][N+](=O)C1=CC2=C(OC3(CCC3)CC2=O)C=C1')])
pathways = []
TOPN = 10
for i in range(10000):
    try:
        print(f'i:{i}')
        if i % 1000 == 0 and i != 0:
            root.save(filename=f'checkpoints/chkpoint{i}')
        res = root.run()
        if res:
            if len(pathways) < TOPN:
                heapq.heappush(pathways, res)
            else:
                heapq.heappushpop(pathways, res)
    except KeyboardInterrupt:
        break
# root.draw_tree()
# term = root.get_synthesis()
# cur_list, reactions = term.get_synthesis_list()
# print(list(map(str, cur_list)))
pathways.sort(key=lambda x: x[0], reverse=True)
cur_list, reactions = pathways[0].get_synthesis_list()
print(list(map(str, cur_list)))

root.save(filename='tree10000')
f = open('pathways.pkl', 'wb')
pickle.dump(pathways, f)