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
		self.v0	= 0.1*np.ones(self.N)	# Initial values > 0
		self.m0 = np.zeros(self.N + 1)	# Initial motivation state
		self.m0[-1]	= 1.				# Initially, 100% uncommitted

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

		if self.child1 is not None:	# If there is a right child, set it up
			self.child1.setup()

			# Update the number of descendants
			self.nDescendants += 1 	# Add this child
			self.nDescendants += self.child1.nDescendants 	# Add the child's
															# descendants

			# Concatenate your state plus the states of the children
			self.z0 = np.r_[self.z0, self.child1.z0]

	def nodeFlow(self, m, v):
		'''Implements the Seeley et al. model embedding an unfolded pitchfork.
		This drives decision-making at a given node on the basis of the option
		values v.'''
		# Initialize output
		mDot = np.zeros(self.N + 1)

		# Scale the values by the gain factor (default = 1)
		h = self.gain * v

		# Compute dynamics for each option
		mDot[:-1] += -m[:-1]/h + h*m[-1]*(1+m[:-1]) - self.sigma*m[:-1].cumprod()[-1]

		# Adjust for the "unmotivated" motivation (so sum = 0)
		mDot[-1] = -np.sum(mDot[:-1])

		return mDot

