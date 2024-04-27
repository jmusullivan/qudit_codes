"""Functions and classes for qutrit TC decoder"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Dict, Any, Set, FrozenSet, Iterable, Tuple
import math
import random
import networkx as nx

import pandas as pd
import os
import galois

d = 3
GF = galois.GF(d )


class TCd():
    def __init__(self, Lx, Ly):
        self.Lx, self.Ly = [Lx,Ly]

    def check_data_lists(self):
        qudits_from_plaq: List[Tuple[complex]] = [(1+0*1j), (0+1j)]
        plaq_centers: List[Tuple[complex,complex]] = [((2*i) % (2*self.Lx) + ((2*k) % (2*self.Ly))*1j) for i in range(self.Lx) for k in range(self.Ly)]

        qudits_list: List[Tuple[complex]] = []
        for plaq in plaq_centers:
            qudits_list.append(tuple_sum(plaq, qudits_from_plaq[0], self.Lx, self.Ly))
            qudits_list.append(tuple_sum(plaq, qudits_from_plaq[1], self.Lx, self.Ly))

        plaq_qudit_dictionary = {}
        for count, plaq in enumerate(plaq_centers):
            plaq_qudit_dictionary[count] = [tuple_sum(plaq, qudits_from_plaq[0],self.Lx, self.Ly), tuple_sum(plaq, qudits_from_plaq[1],self.Lx, self.Ly),tuple_sum(plaq, -qudits_from_plaq[0],self.Lx, self.Ly), tuple_sum(plaq, -qudits_from_plaq[1],self.Lx, self.Ly) ]


        self.plaq_pos = plaq_centers
        self.qutrit_pos = qudits_list
        self.plaq_vs_qutrits = plaq_qudit_dictionary

    def parity_check(self):
        H = np.zeros((len(self.plaq_pos),len(self.qutrit_pos)), dtype=int)

        for count_p, p in enumerate(self.plaq_pos):
            for count_q, q in enumerate(self.qutrit_pos):
                if q == self.plaq_vs_qutrits[count_p][0]:
                    H[count_p,count_q] = 1
                elif q == self.plaq_vs_qutrits[count_p][1]:
                    H[count_p,count_q] = 1
                elif q == self.plaq_vs_qutrits[count_p][2]:
                    H[count_p,count_q] = d-1
                elif q == self.plaq_vs_qutrits[count_p][3]:
                    H[count_p,count_q] = d-1
        
        self.H = GF(H)

    def tanner_graph(self):
        "Create tanner graph from a parity check matrix H."
        m, n = self.H.shape
        T = nx.Graph()

        T.H = self.H
        # nodes
        T.VD = [i for i in range(n)]
        T.Dpos = [self.qutrit_pos[i] for i in range(n)]
        T.VC = [-j-1 for j in range(m)]
        T.Cpos = [self.plaq_pos[j] for j in range(m)]

        # add them, with positions
        #for i, node in enumerate(T.VD):
            #T.add_node(node, pos=((i-n/2), 0), label='$d_{'+str(i)+'}$')
        #for j, node in enumerate(T.VC):
            #T.add_node(node, pos=((j-m/2), 1), label='$c_{'+str(j)+'}$')

             #add them, with positions
        for i, node in enumerate(T.VD):
            T.add_node(node, pos=(T.Dpos[i].real, T.Dpos[i].imag), label='$d_{'+str(i)+'}$')
        for j, node in enumerate(T.VC):
            T.add_node(node, pos=(T.Cpos[j].real, T.Cpos[j].imag), label='$c_{'+str(j)+'}$')

        # add checks to graph
        for j, check in enumerate(T.H):
            for i, v in enumerate(check):
                if v==1:
                    T.add_edge(-j-1, i, label = 'Z')
                elif v==2:
                    T.add_edge(-j-1, i, label = 'Zt')

        self.T = T

    def logicals(self):
        qudits = self.T.VD
        Z_vertical, Z_horizontal = [np.zeros(len(qudits), dtype=int), np.zeros(len(qudits), dtype = int)]
        Z_vlist, Z_hlist = [set(), set()]

        for y in range(self.Ly):
            Z_vertical[2*y] = 1
            Z_vlist.add(2*y)

        for x in range(self.Lx):
            Z_horizontal[1+2*self.Ly*x] = 1
            Z_hlist.add(1+2*self.Ly*x)


        self.Z1, self.Z1list = [Z_vertical, Z_vlist]
        self.Z2, self.Z2list = [Z_horizontal, Z_hlist]

    def generate_code(self):
        self.check_data_lists()
        self.parity_check()
        self.tanner_graph()
        self.logicals()




#####################################################################
#####################################################################

def draw_tanner_graph(T, highlight_vertices=None):
    "Draw the graph. highlight_vertices is a list of vertices to be colored."
    plt.figure(figsize=(20,15))
    
    pl=nx.get_node_attributes(T,'pos')
    lbls_nodes = nx.get_node_attributes(T, 'label')
    lbls_edges = nx.get_edge_attributes(T, 'label')

    nx.draw_networkx_nodes(T, pos=pl, nodelist=T.VD, node_shape='o')
    nx.draw_networkx_nodes(T, pos=pl, nodelist=T.VC, node_shape='s')
    nx.draw_networkx_labels(T, pos=pl, labels=lbls_nodes)

    #for i, edge in enumerate(T.edges):
        #nx.draw_networkx_edges(T, pos=pl, edgelist=T.edges[edge])
    
    nx.draw_networkx_edges(T, pos=pl, label= lbls_edges)

    if highlight_vertices:
        nx.draw_networkx_nodes(T,
                               pos=pl,
                               nodelist=[int(v[1:]) for v in highlight_vertices if v[0] == 'd'],
                               node_color='red',
                               node_shape='o')
        nx.draw_networkx_nodes(T,
                       pos=pl,
                       nodelist=[-int(v[1:])-1 for v in highlight_vertices if v[0] == 'c'],
                       node_color='red',
                       node_shape='s')

    
    plt.axis('off');


# these four functions allow us to convert between 
# (s)tring names of vertices and (i)nteger names of vertices
def s2i(node):
    return int(node[1:]) if node[0] == 'd' else -int(node[1])-1

def i2s(node):
    return f'd{node}' if node>=0 else f'c{-node-1}'

def ms2i(W: set):
    return set(map(s2i, W))

def mi2s(W: set):
    return set(map(i2s, W))

def i2pos(T,node):
    return T.Dpos[node] if node>=0 else T.Cpos[-node-1]


#####################################################################
#####################################################################

def interior(T, W): # interior is subset S of cluster W for which all neighbors of S are in W
    "Determine interior of vertex subset W of Tanner graph T."
    IntW = set()
    for v in W:
        if set(nx.neighbors(T,v)).issubset(W):
            IntW.add(v)
    return IntW


def solvable_system(A,b):
    "Determines if there is a solution to Ax=b."
    A_rank = np.linalg.matrix_rank(A)

    # create augmented matrix
    Ab = np.hstack((A, np.atleast_2d(b).T))

    # Must be true for solutions to be consistent
    return A_rank == np.linalg.matrix_rank(Ab)

def solve_underdetermined_system(A, b):
    "Returns a random solution to Ax=b."
    n_vars = A.shape[1]
    A_rank = np.linalg.matrix_rank(A)

    # create augmented matrix
    Ab = np.hstack((A, np.atleast_2d(b).T))

    # Must be true for solutions to be consistent
    if A_rank != np.linalg.matrix_rank(Ab):
        raise Exception("No solution exists.")

    # reduce the system
    Abrr = Ab.row_reduce()

    # additionally need form in which the identity
    # is moved all the way to the left. Do some
    # column swaps to achieve this.
    swaps = []
    for i in range(min(Abrr.shape)):
        if Abrr[i,i] == 0:
            for j in range(i+1,n_vars):
                if Abrr[i,j] == 1:
                    Abrr[:, [i,j]] = Abrr[:, [j,i]]
                    swaps.append((i,j))
                    break

    # randomly generate some independent variables
    n_ind = n_vars - A_rank
    ind_vars = GF(np.zeros(n_ind,dtype = int))

    # compute dependent variables using reduced system and dep vars
    dep_vars = -Abrr[:A_rank,A_rank:n_vars]@ind_vars + Abrr[:A_rank,-1]

    # x is just concatenation of the two
    x = np.hstack((dep_vars, ind_vars))

    # swap the entries of x according to the column swaps earlier
    # to get the solution to the original system.
    for s in reversed(swaps):
        x[s[0]], x[s[1]] = x[s[1]], x[s[0]]

    return x

def is_valid_cluster(T, syndrome, cluster):
    "Given a syndrome and cluster, determines if is is valid."

    #cluster_interior = interior(T, cluster)

    data_qubits_in_interior = sorted([i for i in cluster if i>=0])
    check_qubits_in_cluster = sorted([-i-1 for i in cluster if i<0])

    #GF = galois.GF(3)
    A = GF(T.H[check_qubits_in_cluster][:,data_qubits_in_interior])
    b = GF([syndrome[i] for i in check_qubits_in_cluster])

    solved = solvable_system(A,b)

    return solved


def find_valid_correction(T, syndrome, cluster): # only to be used when is_valid_cluster = True

    #cluster_interior = interior(T, cluster)

    data_qubits_in_interior = sorted([i for i in cluster if i>=0])
    check_qubits_in_cluster = sorted([-i-1 for i in cluster if i<0])

    #GF = galois.GF(3)
    A = GF(T.H[check_qubits_in_cluster][:,data_qubits_in_interior])
    b = GF([syndrome[i] for i in check_qubits_in_cluster]) 

    sol = solve_underdetermined_system(A,b)

    return sol, data_qubits_in_interior


def cluster_neighbors(T, cluster):
    nbrs = set()
    for v in cluster:
        nbrs.update(set(nx.neighbors(T,v)))
    return nbrs



def grow_cluster_nhood(T, cluster, n): #grows cluster by another layer of check qubits
    
    for rounds in range(n):
        cluster.update(cluster_neighbors(T,cluster))
        cluster.update(cluster_neighbors(T,cluster))

    return cluster



#################################################
#################################################

def my_decoder(T, error):
    n = len(T.VD)
    syndrome_update = T.H @ error
    e_estimate = GF(np.array([0]*n, dtype=int))


    cluster_size = 1
    n_loop_counter=0
    while True:
        e_estimate_temp = GF(np.array([0]*n, dtype=int))


        n_loop_counter+=1
        #print(f'starting round {n_loop_counter}')
        
        K = [grow_cluster_nhood(T, set([-i-1]), cluster_size) for i,s in enumerate(syndrome_update) if s!=0]
        #print(f'there are {len(K)} clusters')
        
        # now see if merging helps

        merged_to_delete = set()
        for i,Ki in enumerate(K):
            for j,Kj in enumerate(K):
                if j < i and (not Ki.isdisjoint(Kj)):
                    Kj.update(Ki)
                    merged_to_delete.update([i])
                        
                elif j > i and (not Ki.isdisjoint(Kj)):
                    Ki.update(Kj)
                    merged_to_delete.update([j])
                        

        for i in reversed(sorted(merged_to_delete)):
            K.pop(i)

        
        #print('==============================')
        #print(f'there are {len(K)} clusters after merging')
        #repeat about for the above for merged clusters
        for i, Ki in enumerate(K):
            if is_valid_cluster(T, syndrome_update, Ki):
                correction, correction_data_qubits = find_valid_correction(T, syndrome_update, Ki)
                e_estimate_temp[correction_data_qubits] += correction


        syndrome_update -= T.H @ e_estimate_temp
        e_estimate += e_estimate_temp
        
        if np.array(syndrome_update).sum() ==0:
            break
        
        #infinite loop failsafe
        if n_loop_counter> 20:
            print('infinite loop, something wrong')
            break
        
        # grow size of clustering
        cluster_size+=1


        
    return e_estimate, syndrome_update
    









#####################################################################
#####################################################################

#other useful functions
def tuple_sum(p:Tuple[complex], e:Tuple[complex], Lx, Ly):
    x = (p.real + e.real) % (2*Lx)
    y = (p.imag + e.imag) % (2*Ly)

    return (x+1j*y)

def Error(p, H):
    data_qutrit = np.zeros(H.shape[1], dtype=int)
    for q in range(H.shape[1]):
        p0 = np.random.random(1)
        if p0<p:
            data_qutrit[q] = 1
        elif p<p0<2*p:
            data_qutrit[q] = d-1

    return data_qutrit