from tree import Node
root = Node([[('CCC(COC(=O)[C@@H](NP(=O)(Oc1ccccc1)OC[C@H]1O[C@@]([C@@H]([C@@H]1O)O)(C#N)c1ccc2n1ncnc2N)C)CC', tuple())], dict()], None)
for i in range(10):
    print(i)
    root.run()

print(root.select().state)