# Example code

execfile('decisionNode.py')

# Construct a tree with both left and right children
n=decisionNode() # root node 						0
n.child0=decisionNode() 					#	1		3
n.child1=decisionNode() 				#	2		
n.child0.child0=decisionNode()

n.setup()

n.simulate()

n.mOut = n.parseMStates(n.zOut)

n.mOptions = n.parseLeafStates(n.mOut)