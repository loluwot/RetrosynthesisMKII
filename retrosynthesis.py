from tree import *
root = State([MolNode('CC(C)(C)OC(=O)N1CCN(CC1)C1=CC(NS(=O)(=O)C2=CC=CC=C2F)=CC2=C1OC1(CCC1)CC2')])
# root = State([MolNode(x) for x in 'CC(C)(C)OC(=O)N1CCNCC1 Nc1cc(Br)c2c(c1)CCC1(CCC1)O2 O=S(=O)(Cl)c1ccccc1F'.split(' ')])
# root = State([MolNode('[O-][N+](=O)C1=CC2=C(OC3(CCC3)CC2=O)C=C1')])
for i in range(20): 
    print(f'i:{i}')
    root.run()

root.draw_tree()

print(str(root.get_synthesis()))