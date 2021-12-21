from tree import Node

root = Node([[('CCN1C(=O)COc2ccc(CN3CCN(CCOc4cccc5nc(C)ccc45)CC3)cc21', tuple())], dict()], None)

for i in range(10):
    print(i)
    root.run()

print(root.select().state)
