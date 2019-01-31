# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 12:25:18 2018

@author: RCM
"""
from __future__ import division
import math
import numpy as np
from numpy import linalg as LA
def quaternion_multiplication (x1, ux, y1, uy):
    m1 = x1 * y1 - np.dot(ux, uy)
    m2 = x1 * uy + y1 * ux + ux * uy
    return [m1,m2]
def conjugate_quaternion (x1, u):
    x1 = x1
    u = (-1)*u
    return [x1, u]
def file_len(f):
    for i, l in enumerate(f):
        pass
    return i + 1
f = open('disco1.csv', 'r')
i = 0;
j = 0;
for i in range(file_len(f)):
    for line in f.readlines():
        #print ('line')
        #L[j]= line.split(',')
        if j == 1:
            #print (line)
            L1 = line.split(',')
        if j == 2:
            #print (line)
            L2 = line.split(',')
            break
        #Convertimos line en una lista
        #Usando la librería math calculamos (18) de hojas y vemos si coincide con los 
        #quaterniones de los datos        
        t0 = float(L1[0])
        array_aux = np.array([L1[7],L1[8],L1[9]])
        angular_rate = array_aux.astype(np.float)
        norm_angular_rate = LA.norm(angular_rate)
        Ts = float(L2[0]) - t0
        aux_q1 = (1/2) * (norm_angular_rate * Ts)
        q1 = math.cos(aux_q1)
        aux_q2 = angular_rate / norm_angular_rate
        q2 = math.sin(aux_q1) * aux_q2
        qw = [q1,q2] #Ojo qw es del tipo list
        aux_quaternion1 = np.array([L1[10],L1[11],L1[12],L1[13]])
        aux_quaternion2 = aux_quaternion1.astype(np.float)
        quaternion = [aux_quaternion2[0], np.array([aux_quaternion2[1],aux_quaternion2[2],aux_quaternion2[3]])]
        #quaternion_0 = [0,np.zeros(3)]
        q_n_b = (quaternion_multiplication(quaternion[0], quaternion[1], q1, q2))
        position_vector_0 = [0,0,1]
        quaternion_vector_0 = [ 0,np.array(position_vector_0)]
        conjugated_quaternion = conjugate_quaternion(q_n_b[0],q_n_b[1])
        aux_position_vector_1 = quaternion_multiplication(q_n_b[0], q_n_b[1], quaternion_vector_0[0], quaternion_vector_0[1])
        position_vector_1 = quaternion_multiplication(aux_position_vector_1[0], aux_position_vector_1[1], conjugated_quaternion[0],conjugated_quaternion[1])
        j = j+1
f.close()