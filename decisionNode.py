#####
# Python code to implement a decision-making dynamical system
# The main object is a node, which decides between two options.
# Methods allow nodes to be parsed into a binary tree.
# Further methods allow simulation.
#####

# Import relevant code
import numpy as np
from scipy.integrate import odeint

class decisionNode:
	def __init__(self):
		'''Initialize the decision node.'''

		self.N 	= 2 	# number of options at this node

		# Initial conditions for the state variables and parameters
		#self.v0	= 0.1*np.ones(self.N)	# Initial values > 0
		self.m0 = np.zeros(self.N)		# Initial motivation state
										# Initially, 100% uncommitted

		# Simulation parameters
		self.t0		= 0. 				# Initial time
		self.tf 	= 10. 				# Final time
		self.tpts	= 100				# Number of solution points

		# Decision-making model parameters
		self.sigma	= 4. 				# Stop signal parameter
		self.v 	= 0.1*np.ones(self.N)	# Values for the two options
		self.gain 	= 1. 				# Value gain (default = 1)

		# Set up pointers for the node's children
		self.child0 = None				# By default, no children
		self.child1 = None

		# Set up locations for option names
		self.option0 = ''				# Option 0
		self.option1 = ''				# Option 1

	def setup(self):
		'''Method to traverse the decision tree in depth-first order using
		recursion to setup the initial conditions for the model dynamics.
		The convention used to construct the state is 
		z = [z_self, z_child0, z_child1].
		If any child is null, the corresponding state is omitted.'''

		# Initialize number of descendants of this node
		self.nDescendants = 0	# By default, no descendants.

		# Set up the node's own initial state
		self.z0 = np.r_[self.m0]

		if self.child0 is not None:	# If there is a left child, set it up
			self.child0.setup()

			# Update the number of descendants
			self.nDescendants += 1 	# Add this child
			self.nDescendants += self.child0.nDescendants 	# Add the child's
															# descendants

			# Concatenate your state plus the states of the children
			self.z0 = np.r_[self.z0, self.child0.z0]

			# Update the associated value to be the mean of the children's
			self.v[0] = self.child0.v.mean()

		if self.child1 is not None:	# If there is a right child, set it up
			self.child1.setup()

			# Update the number of descendants
			self.nDescendants += 1 	# Add this child
			self.nDescendants += self.child1.nDescendants 	# Add the child's
															# descendants

			# Concatenate your state plus the states of the children
			self.z0 = np.r_[self.z0, self.child1.z0]

			# Update the associated value to be the mean of the children's
			self.v[1] = self.child1.v.mean()

	def nodeFlow(self, m, v):
		'''Implements the Seeley et al. model embedding an unfolded pitchfork.
		This drives decision-making at a given node on the basis of the option
		values v.'''

		# Uses coordinates m = (m1, m2). Then mU = 1-m1-m2
		# Initialize output
		mDot = np.zeros(self.N)

		# Scale the values by the gain factor (default = 1)
		h = self.gain * v

		# Compute dynamics for each option
		mU = 1-np.sum(m)
		mDot += -m/h + h*mU*(1+m) - self.sigma*m.cumprod()[-1]

		# Adjust for the "unmotivated" motivation (so sum = 0)
		#mDot[-1] = -np.sum(mDot[:-1])

		return mDot

	def nodeFlowz(self, zi, ziParent, ziParentDot, vi):
		'''Implements the flow for the z variables associated with the children
		of a node i. zi = ziParent*mi.'''

		# Extract mi
		mi = zi/ziParent

		# Compute miDot
		miDot = self.nodeFlow(mi, vi)

		# Compute ziDot
		ziDot = ziParentDot*mi + ziParent*miDot

		return ziDot

	def flow(self, m, t):
		'''Flow function for the dynamics associated to the node and its 
		descendants, if they exist.'''

		# Extract the node's own state
		mSelf = m[ : self.N]

		mDot = self.nodeFlow(mSelf, self.v)

		# Now, parse the four cases: no children, child0, child1, and both
		if self.child0 is None:
			if self.child1 is None:
				# No children: no recursion needed (base case)

				# Concatenate the vector fields
				zDot = np.r_[mDot]

			else:
				# Only child1
				# Parse the state variable: m = r_[mSelf, m1]
				# Own state
				mSelf = m[ : self.N] # r_[mSelf, m1]

				# Child1 state variables
				m1 = m[self.N : ]

				# Compute the child flow
				m1Dot = self.child1.flow(m1, t)

				# Concatenate the flow vector fields
				mDot = np.r_[mDot, m1Dot]
		else:	# child0 exists
			if self.child1 is None:
				# Only child0
				# Parse the state variable: m = r_[mSelf, m0]
				# Own state
				mSelf = m[ : self.N]

				# Child0 state variables
				m0 = m[self.N : ]

				# Compute the child flow
				m0Dot = self.child0.flow(m0, t)

				# Concatenate the flow vector fields
				mDot = np.r_[mDot, m0Dot]
			else:
				# Both children exist; need to be careful about parsing states
				# Parse the state variables: m = r_[mSelf, m0, m1]
				# Own state
				mSelf = m[ : self.N ]

				# Child0 state variables (1 + nDescendants copies of m)
				m0 = m[self.N : (2+self.child0.nDescendants)*self.N]

				# Child1 state variables (1 + nDescendants copies of m)
				m1 = m[(2+self.child0.nDescendants)*self.N : ]

				# Compute the child flows
				m0Dot = self.child0.flow(m0, t)
				m1Dot = self.child1.flow(m1, t)

				mDot = np.r_[mDot, m0Dot, m1Dot]

		return mDot

	def flowz(self, z, t):
		'''Flow function for the node states and their dynamics using the 
		recursive z coordinates.'''

		# Extract the node's own state
		zSelf = z[ : self.N]

		# Start at the root node (special case)
		if self.isRoot:
			ziParent 	= 1
			ziParentDot = 0
			zSelfDot = self.nodeFlowz(zSelf, ziParent, ziParentDot, self.v)
		else: # Not at the root; must have a parent
			# Parse the four cases: no children, child0, child1, and both
			if self.child0 is None:
				if self.child1 is None:
					# No children: just output zSelfDot
					zDot = zSelfDot
				else: # Only child1
					# Parse the state variable: z = r_[zSelf, z1]
					z1 = z[self.N : ]
					z1Dot = self.child1.flow()

		return zDot


	def parseMStates(self, mOut):
		'''Assumes that mOut is the output of the simulation. mOut is an array
		of shape (self.tpts, 2*total states).'''

		# Extract nodes's own state
		zSelfOut = mOut[:, :self.N]

		if self.child0 is None:
			if self.child1 is None:
				# No children: just output zSelfOut
				zOut = zSelfOut
			else:	# Only child1
				mOut1 = [:, self.N :]
				zOut1 = zSelfOut[:, 1]*mOut1

				zOut = c_[zSelfOut, zOut1]
		else:	# child0 exists
			if self.child1 is None:
				# Only child0
				mOut0 = [:, self.N :]
				zOut0 = zSelfOut[:, 0]*mOut0

				zOut = c_[zSelfOut, zOut1]
			else:
				# Both children exist; need to be careful about parsing states
				# Parse the state variables: zOut = c_[zSelf, z0, z1]

				# Child0 state variables (nChildren copies of z)
				mOut0 = [:, self.N : (2+self.child0.nDescendants)*self.N]

				# Child1 state variables (nChildren copies of z)
				mOut1 = [:, (2+self.child0.nDescendants)*self.N :]

				zOut0 = zSelfOut[:, 0]*mOut0
				zOut1 = zSelfOut[:, 1]*mOut1

				zOut = c_[zSelfOut, zOut0, zOut1]

		return zOut


	def simulate(self):
		'''Method to integrate the vector field flow and parse the resulting
		state traces.'''

		self.tt = np.linspace(self.t0, self.tf, self.tpts)

		z = odeint(self.flow, self.z0, self.tt)

		self.zOut = z