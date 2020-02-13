# Example code

execfile('decisionNode.py')

# Construct a tree with both left and right children
n=decisionNode() # root node
n.child0=decisionNode()
n.child1=decisionNode()

n.setup()

n.simulate()