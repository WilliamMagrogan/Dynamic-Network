#####
# Author: William Magrogan
# Email: william.magrogan@colorado.edu
#
# Purpose: This module contains the essential code for the Dynamic Network
# object

# Imports
import numpy as np
import networkx as nx
import scipy as sp
from scipy.integrate import solve_ivp as scipy_solve
from scipy.linalg import block_diag


class DynaNet():
    def __init__(self, structure = '', adj_mtx = [], adj_lst = []):
        '''
        structure: if predifined structure, it will generate such structure.(future iterations)
        adj_mtx: Adjacency matrix if available
        adj_lst: Adjacency List as a dictionary
        '''
        if adj_mtx == []:
            self.adj_lst =  adj_lst
            self.adj_mtx = np.zeros()
            for i in adj_lst.keys():
                for j in adj_lst[i]:
                    self.adj_mtx[i, j] = 1
        else:
            self.adj_mtx = adj_mtx
        
        self.vert = [i for i in range(0, adj_mtx.shape[0])]
        self.edge = []
        for i in self.vert:
            for j in range(i, self.adj_mtx.shape[1]):
                if self.adj_mtx[i,j] == 1:
                    self.edge += [[i,j]]
    
    def get_edges(self, m):
    	edges = []
    	for i in self.vert:
            for j in range(i, self.adj_mtx.shape[1]):
                if self.adj_mtx[i,j] == 1:
                    edges += [[i,j]]

    def generate_incidence(self):
        '''
        populates internal incidence matrix state
        '''
        self.incidence_mtx = np.zeros([len(self.vert), len(self.edge)])
        for e in range(0, len(self.edge)):
            self.incidence_mtx[ min(self.edge[e]),  e] = 1
            self.incidence_mtx[ max(self.edge[e]),  e] = -1

    def graph_gradient(self):
        '''
        populates adjacency like matrix that is the gradient of each vetice
        '''
        self.grad_mtx = np.zeros(self.adj_mtx.shape)
        
        for e in self.edge:
            self.grad_mtx[e[0], e[0]] += 1
            self.grad_mtx[e[0], e[1]] = -1
            self.grad_mtx[e[1], e[1]] += 1
            self.grad_mtx[e[1], e[0]] = -1

    def generate_laplacian(self):
    	self.generate_incidence()
    	K = self.incidence_mtx
    	self.laplacian_mtx = np.dot(K, np.transpose(K))

    def solve_ivp(self, df, f0, t0, tf, nt, *args, **kwargs):
        '''
        df: function of f' takes in a incidence matrix, gradient matrix a state i, and a time t returns derivative
        f0: Initial value
        t0: initial time
        tf: final time
        nt: Number of steps
        '''
        
        # Populate our incidence, gradient, and laplacian graphs 
        self.generate_incidence()
        self.graph_gradient()
        K = self.incidence_mtx
        Glap = np.dot(K, np.transpose(K))
        Ggrad = self.grad_mtx

        # PDE starting conditions
        t = np.linspace(t0, tf, nt)
        dt = (tf-t0)/nt
        f = np.zeros([self.adj_mtx.shape[0], len(t)])

        # check if alternative solver is being used
        if "solver" in kwargs.keys():
            solver = kwargs["solver"]
        else:
            solver = scipy_solve

        # Run graph PDE and store results in internal state
        results = solver(df, [t[0], t[-1]], f0, t_eval=t, args=tuple([Glap, Ggrad] + list(args)))
        self.solve_ivp_results = results
            

class MultiplexDynaNet():
    def __init__(self, adj_matrices=[]):
        '''
        This class is similar to DynaNet, but incorporates the ability to simulate dynamics on a MultiGraph or a Multiplexed graph.

        adj_matrices: array-like of adjacency matrices on the same set of nodes 
        '''

        self.adj_matrices = adj_matrices
        self.vert = [i for i in range(0, adj_matrices[0].shape[0])]
        self.edge_sets = []
    
    def get_edges(self, m):
    	'''
    	return a set of edges for a matrix with respect to the vertices of the multiplexed network
    	'''

    	edges = []
    	for i in self.vert:
            for j in range(i, m.shape[1]):
                if m[i,j] == 1:
                    edges += [[i,j]]

    	return edges

    def generate_incidence(self):
        '''
        populates internal incidence matrix state
        '''

        # We need one incidence matrix for each layer in the multiplex
        self.incidence_matrices = []
        for m in self.adj_matrices:
        	edges = self.get_edges(m)
	        incidence_mtx = np.zeros([len(self.vert), len(edges)])
	        for e in range(0, len(edges)):
	            incidence_mtx[ min(edges[e]),  e] = 1
	            incidence_mtx[ max(edges[e]),  e] = -1
	        self.incidence_matrices.append(incidence_mtx)

    def generate_laplacian(self):
    	'''
    	generate laplacian matrices for each network in the multiplex
    	'''

    	# We need one laplacian for each layer
    	self.laplacian_matrices = []
    	self.generate_incidence()
    	for ii in range(len(self.incidence_matrices)):
    		K = self.incidence_matrices[ii]
    		L = np.dot(K, np.transpose(K))
    		self.laplacian_matrices.append(np.dot(K, np.transpose(K)))

    def generate_multi_laplacian(self):
    	'''
    	creates an internal variable that combines the laplacians of the individual layers into a multiplexed one for use in PDE dynamics
    	'''

    	self.multi_laplacian = block_diag(*[k for k in self.laplacian_matrices])


    def solve_ivp(self, df, f0, t0, tf, nt, *args, **kwargs):
        '''
        df: function of f' takes in a incidence matrix, gradient matrix a state i, and a time t returns derivative
        f0: Initial value
        t0: initial time
        tf: final time
        nt: Number of steps
        '''
        
        # Populate our incidence, gradient, and laplacian graphs 
        self.generate_laplacian()
        self.generate_multi_laplacian()
        Glap = self.multi_laplacian

        # PDE starting conditions
        t = np.linspace(t0, tf, nt)
        dt = (tf-t0)/nt

        # check for alternative ivp solver
        if "solver" in kwargs.keys():
            solver = kwargs["solver"]
        else:
            solver = scipy_solve

        # Run graph PDE and store results in internal state
        results = solver(df, [t[0], t[-1]], f0, t_eval=t, args=tuple([Glap] + list(args)))
        self.solve_ivp_results = results