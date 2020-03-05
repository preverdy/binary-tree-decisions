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

# Now, construct a tree with four options and explore the effect of v
n = decisionNode() # root node 						0
n.child0 = decisionNode() 				#	1				2
n.child1 = decisionNode()		# options 1   2			  3   4

# Default values of v result in deadlock.
n.setup()
n.simulate()

n.mOut = n.parseMStates(n.zOut)
n.mOptions = n.parseLeafStates(n.mOut)

plot(n.tt, n.mOptions)

# Increase v to destabilize deadlock
n.child0.v *= 100
n.child1.v *= 100

n.setup()
n.simulate()

n.mOut = n.parseMStates(n.zOut)
n.mOptions = n.parseLeafStates(n.mOut)

plot(n.tt, n.mOptions)

# Even with large v values symmetric initial conditions result in deadlock
# Perturb initial conditions to avoid deadlock.
n.m0[0] = 0.1
n.child0.m0[1] = 0.1

n.setup()
n.simulate()

n.mOut = n.parseMStates(n.zOut)
n.mOptions = n.parseLeafStates(n.mOut)

plot(n.tt, n.mOptions)

# Perturb v to make a clear winner.
n.child1.v[0] += 10
n.child0.m0[0] = 0.1

n.setup()
n.simulate()

n.mOut = n.parseMStates(n.zOut)
n.mOptions = n.parseLeafStates(n.mOut)

plot(n.tt, n.mOptions)

# And now, perturb v again so there are two winners
n.child1.v[1] += 10
n.child0.m0[1] = 0.095 	# Make the initial condition symmetric

n.setup()
n.simulate()

n.mOut = n.parseMStates(n.zOut)
n.mOptions = n.parseLeafStates(n.mOut)

figure()
plot(n.tt, n.mOptions)