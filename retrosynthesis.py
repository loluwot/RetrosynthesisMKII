from tree import Node
<<<<<<< HEAD
root = Node([[('CCN1C(=O)COc2ccc(CN3CCN(CCOc4cccc5nc(C)ccc45)CC3)cc21', tuple())], dict()], None)
=======
root = Node([[('CCC(COC(=O)[C@@H](NP(=O)(Oc1ccccc1)OC[C@H]1O[C@@]([C@@H]([C@@H]1O)O)(C#N)c1ccc2n1ncnc2N)C)CC', tuple())], dict()], None)
>>>>>>> 55cf01f8d477ca7fea3c995ce0bf2369b82606fc
for i in range(10):
    print(i)
    root.run()

print(root.select().state)