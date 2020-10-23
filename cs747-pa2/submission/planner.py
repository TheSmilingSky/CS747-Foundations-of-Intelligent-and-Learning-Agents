import argparse
import random
import numpy as np
import math
from pulp import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("-mdp", "--mdp", required=True)
parser.add_argument("-al", "--algorithm", required=True)
args = parser.parse_args()

# 

def storeMDP(mdpPath):
	with open(mdpPath) as f:
		lines = f.readlines()
	S = int(lines[0].split()[-1])
	A = int(lines[1].split()[-1])
	start = int(lines[2].split()[-1])
	end = [int(x) for x in lines[3].split()[1:]]
	T = np.zeros((S,A,S))
	R = np.zeros((S,A,S))
	T_dict = {}
	R_dict = {}
	for line in lines[4:]:
		if line.startswith('transition'):
			line = line.split()
			s1,ac,s2,r,p = int(line[1]),int(line[2]),int(line[3]),float(line[4]),float(line[5])
			T_dict[(S*A)*s1 + S*ac + s2] = p
			R_dict[(S*A)*s1 + S*ac + s2] = r
			T[s1,ac,s2] = p
			R[s1,ac,s2] = r
	mdptype = lines[-2].split()[-1]
	discount = float(lines[-1].split()[-1])
	return S,A,start,end,T,R,mdptype,discount,T_dict,R_dict

def ValueIteration(S,A,T,R,discount):
	V = np.random.uniform(0,1,S)
	Vprev = np.random.uniform(0,1,S)*0.5
	while np.linalg.norm(V-Vprev)>1.e-9:
		Vprev = V.copy()
		V = np.max(np.sum(np.multiply(T,R + discount*V),axis=2),axis=1)
	pi = np.argmax(np.sum(np.multiply(T,R + discount*V),axis=2),axis=1)
	return V, pi

def PolicyIteration(S,A,T,R,discount,T_dict,R_dict):
	V = np.random.uniform(0,1,S)
	Q = np.sum(np.multiply(T,R + discount*V),axis=2)
	pi = np.argmax(Q,axis=1)
	improving = True
	while improving:
		cnt = 0
		pi = np.argmax(np.sum(np.multiply(T,R + discount*V),axis=2),axis=1)
		Vprev = V.copy()
		V = PolicyEvaluation(S,A,T,R,discount,pi,V,T_dict,R_dict)
		if np.linalg.norm(V-Vprev)<1.e-9:
			improving = False
	return V, pi

def PolicyEvaluation(S,A,T,R,discount,pi,V,T_dict,R_dict):
	problem = LpProblem('pitoV',LpMinimize)
	V = LpVariable.dicts('V',np.arange(S))
	problem += 0
	for s in range(S):
		a = int(pi[s])
		exp = 0
		for s_ in range(S):
			if (S*A)*s + S*a + s_ in T_dict.keys():
				exp += T_dict[(S*A)*s + S*a + s_]*(R_dict[(S*A)*s + S*a + s_] + discount*V[s_])
		problem += lpSum([exp])==V[s]
	PULP_CBC_CMD(msg=0).solve(problem)
	Val = np.zeros(S)
	for s in range(S):
		Val[s] = value(V[s])
	return Val

def LinearProgramming(S,A,T,R,discount,T_dict,R_dict):
	problem = LpProblem('MDP',LpMinimize)
	V = LpVariable.dicts('V',list(range(S)))
	problem += lpSum([V[s] for s in range(S)])
	for s in range(S):
		for a in range(A):
			exp = 0
			for s_ in range(S):
				if (S*A)*s + S*a + s_ in T_dict.keys():
					exp += T_dict[(S*A)*s + S*a + s_]*(R_dict[(S*A)*s + S*a + s_] + discount*V[s_])
			problem += lpSum([exp])<=V[s]
	problem.solve(PULP_CBC_CMD(msg=0))
	for s in range(S):
		V[s] = pulp.value(V[s])
	V = np.array(list(V.values()), dtype=float)
	pi = np.argmax(np.sum(np.multiply(T,R + discount*V),axis=2),axis=1)
	return V, pi


if __name__=='__main__':
	mdpPath = args.mdp
	algorithm = args.algorithm
	S,A,start,end,T,R,mdptype,discount,T_dict,R_dict = storeMDP(mdpPath)
	if algorithm == 'vi':
		V, pi = ValueIteration(S,A,T,R,discount)
	elif algorithm == 'hpi':
		V, pi = PolicyIteration(S,A,T,R,discount,T_dict,R_dict)
	elif algorithm == 'lp':
		V, pi = LinearProgramming(S,A,T,R,discount,T_dict,R_dict)
	for s in range(S):
		print("{0:6f}".format(V[s]),pi[s])