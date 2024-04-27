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

d = 2
GF = galois.GF(d)


class TC3():
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
            Z_vertical[1+2*y] = 1
            Z_vlist.add(1+2*y)

        for x in range(self.Lx):
            Z_horizontal[1+2*self.Ly*x] = 1
            Z_hlist.add(1+2*self.Ly*x)


        self.Z1, self.Z1list = [Z_vertical, Z_vlist]
        self.Z2, self.Z2list = [Z_horizontal, Z_hlist]



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


# these four functions allow us to convert between T
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

# Define Errors

def Error(p, H):
    data_qutrit = np.zeros(H.shape[1], dtype=int)
    for q in range(H.shape[1]):
        p0 = np.random.random(1)
        if p0<p:
            data_qutrit[q] = 1
        elif p<p0<2*p:
            data_qutrit[q] = 2

    return GF(data_qutrit)

def syndrome(E, code):
    return code.T.H @ E
    

def Bad_syndromes(E, code):
    S = syndrome(E, code)
    S_list = [] # want this in terms of node convention above where check nodes labelled by n<0
    for i in range(len(S)):
        if S[i]!=0:
            S_list.append(-i-1)
    
    return S_list


#####################################################################
#####################################################################

#Define clustering


# want to write everything in terms of node labels

def Manhattan(nodeC, nodeD, code): #S and T have format [(x+iy, t), syndrome]
    Lx, Ly = [code.Lx, code.Ly]
    pC, pD = [i2pos(code.T, nodeC), i2pos(code.T,nodeD)]
    
    lattice_correction = 0
    if nodeC>-1 and nodeD>-1 and nodeC!=nodeD:
        if nodeC%2==0 and nodeD%2 ==0 and np.real(pC) == np.real(pD):
            lattice_correction +=2
        if nodeC%2==1 and nodeD%2 ==1 and np.imag(pC) == np.imag(pD):
            lattice_correction +=2

    x = min(np.real(pC - pD)%(2*Lx), np.real(-pC + pD)%(2*Lx))
    y = min(np.imag(pC - pD)%(2*Ly), np.imag(-pC + pD)%(2*Ly))
    
    
    
    return x+ y + lattice_correction


def plaquette_nhoods(Error, delta, code, string = False):
    Sbad = Bad_syndromes(Error, code)

    data_nodes = code.T.VD
    check_nodes = code.T.VC
    nodes = list(set(data_nodes).union(set(check_nodes)))
    cluster_list = []

    if string == False:
        for p_bad in Sbad:
            cluster_p = [set([p_bad]), set([])]
            for i,node in enumerate(nodes):
                if Manhattan(p_bad, node, code) < delta:
                    cluster_p[1].add(node)
            cluster_list.append(cluster_p)

    elif string == True:
        for p_bad in Sbad:
            cluster_p = [set([i2s(p_bad)]), set()]
            for i,node in enumerate(nodes):
                if Manhattan(p_bad, node, code) < delta:
                    cluster_p[1].add(i2s(node))
            cluster_list.append(cluster_p)

    return cluster_list
    
# returns a list of clusters [{S1,S2,S3,...}, {nodesD,nodesC}]
# add function to trim interior
def merge_clusters(Error, delta, code):
    balls_list = plaquette_nhoods(Error, delta, code, string = False)
    
    n=0
    cluster_list = []
    test_ticker = 0
    if len(balls_list) == 0:   
        n+=1
    
    else:
        while(len(balls_list))>0:
            to_delete = [0]
            temp_cluster = balls_list[0] #ball_list[0] = [{plaq}, {qutrits}] 
            for b, ball in enumerate(balls_list[1:]):
                
                if ball[1].isdisjoint(temp_cluster[1]) == False:
                    temp_cluster[0] = temp_cluster[0].union(ball[0])
                    temp_cluster[1] = temp_cluster[1].union(ball[1])

                    to_delete.append(b+1)
                    
            for i in reversed(to_delete):
                balls_list.pop(i)
                
            cluster_list.append(temp_cluster)

    return cluster_list

#####################################################################
#####################################################################

#Decoding clusters, we will add option to move all charge to a single stabilizer in a cluster

def solvable(H, S):

    H_rank = np.linalg.matrix_rank(H)
    HS = np.hstack((H, np.atleast_2d(S).T))
    
    return H_rank == np.linalg.matrix_rank(HS) 

def solve_cluster(H,S):
    #find random solution to He = S
    n_data = H.shape[1]
    H_rank = np.linalg.matrix_rank(H)

    HS = np.hstack((H, np.atleast_2d(S).T))
    if solvable(H,S) == False:
        raise Exception('something wrong, no solution')
    
    HSrr = HS.row_reduce()

    #swaps 
    swaps = []
    for i in range(min(HSrr.shape)):
        if HSrr[i,i] == 0:
            for j in range(i+1,n_data):
                if HSrr[i,j] != 0:
                    HSrr[:, [i,j]] = HSrr[:, [j,i]]
                    swaps.append((i,j))
                    break

    n_ind = n_data-H_rank
    ind_vars = GF(np.zeros(n_ind,dtype = int))

    dep_vars = -HSrr[:H_rank, H_rank:n_data] @ ind_vars + HSrr[:H_rank, -1]

    e = np.hstack((dep_vars, ind_vars))

    for s in reversed(swaps):
        e[s[0]], e[s[1]] = e[s[1]], e[s[0]]


    return GF(e)

#####################################################################
#####################################################################

#Decoding clusters, we will add option to move all charge to a single stabilizer in a cluster

def decode_cluster(Syndrome, cluster, code):
    cluster_nodes = cluster[1]
    
    check_nodes = sorted([-node-1 for node in cluster_nodes if node<0])
    data_nodes = sorted([node for node in cluster_nodes if node>=0]) 

    H_sub = code.H[check_nodes][:,data_nodes]
    S_sub = Syndrome[check_nodes]

    full_error = GF(np.zeros(2*code.Lx*code.Ly, dtype=int)) #need to embedd the solution in an array for all the data qubits
    
    if solvable(H_sub, S_sub):
        solution = solve_cluster(H_sub, S_sub) #this will return an correction vector lying in the cluster
        
        for i, node in enumerate(data_nodes):
            full_error[node] = solution[i]


    return GF(full_error)

def decode_cluster_layer(Syndrome, clusters,code):
    n_clusters = len(clusters)

    correction = GF(np.zeros(2*code.Lx*code.Ly,dtype=int))
    
    for i, cluster in enumerate(clusters):
        correction += decode_cluster(Syndrome, cluster, code)

    return correction



def cluster_decoder(Error, code):
    E = Error
    S = syndrome(E,code)
    S_index = Bad_syndromes(E,code)

    #print(S)
    #E_correction = GF(np.zeros(2*code.Lx*code.Ly,dtype=int))
    delta = 2.1

    while not Decoded(code.H, E): 
     # change this
        cluster_list = merge_clusters(E, delta, code)

        E_correction = decode_cluster_layer(S, cluster_list, code)
        E-=E_correction

        S = syndrome(E,code)
        S_index = Bad_syndromes(E,code)
        #print(f"did layer {delta}")

        delta +=2
        if delta>max(2*code.Lx+2, 2*code.Ly+2):
            break

    return E       #returns E_inital-E_correction





#####################################################################
#####################################################################

#other useful functions
def tuple_sum(p:Tuple[complex], e:Tuple[complex], Lx, Ly):
    x = (p.real + e.real) % (2*Lx)
    y = (p.imag + e.imag) % (2*Ly)

    return (x+1j*y)

def Decoded(H, E):
    clean_syndrome = GF(np.zeros(H.shape[0], dtype=int))

    return (H @ E == clean_syndrome).all()