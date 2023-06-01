#####
# Author: William Magrogan
# Email: william.magrogan@colorado.edu
#
# Purpose: This module contains the PDE formatted for use with DynaNet

# Imports
import numpy as np

def AdvectionDiffusionStep(time, state, laplacian, gradient, ks, vs):
	'''
	time: current time
	state: current state vector
	laplacian: current graph laplacian
	gradient: current graph gradient
	ks: vector of diffusion coefficients
	vs: vector of velocities at different nodes

	returns: change in state with respect to the advection diffusion equation
	'''

	return np.dot(laplacian, ks*state) - vs*gradient

def FisherKPPStep(time, state, laplacian, gradient, ks, vs):
	'''
	time: current time
	state: current state vector
	laplacian: current graph laplacian
	gradient: current graph gradient
	ks: vector of diffusion coefficients
	vs: vector of velocities at different nodes

	returns: change in state with respect to the Fisher-KPP equation
	'''

	return np.dot(laplacian, ks*state) + state*(1-state)

def FisherKPPExtendedStep(time, state, laplacian, gradient, ks, vs, rate, capacity):
	'''
	time: current time
	state: current state vector
	laplacian: current graph laplacian
	gradient: current graph gradient
	ks: vector of diffusion coefficients
	vs: vector of velocities at different nodes

	returns: change in state with respect to the Fisher-KPP equation
	'''

	return np.dot(laplacian, ks*state) + rate*state*(capacity-state)

def MultiplexFisherKPPStep(time, state, multi_laplacian, ks, vs, coupling):
	'''
	time: current time
	state: current state vector
	multilaplacian: current graph laplacian
	ks: vector of diffusion coefficients
	vs: vector of velocities at different nodes
	coupling: a fuction of state specifying how the equations are coupled

	returns: change in state with respect to the Fisher-KPP equation
	'''

	return np.dot(multi_laplacian, ks*state) + state*(1-state) + coupling(state)

def MultiplexRxnDffn(time, state, multi_laplacian, ks, vs, coupling):
	'''
	time: current time
	state: current state vector
	multilaplacian: current graph laplacian
	ks: vector of diffusion coefficients
	vs: vector of velocities at different nodes
	coupling: a fuction of state specifying how the equations are coupled

	returns: change in state with respect to the Fisher-KPP equation
	'''

	return np.dot(multi_laplacian, ks*state) + coupling(state)
