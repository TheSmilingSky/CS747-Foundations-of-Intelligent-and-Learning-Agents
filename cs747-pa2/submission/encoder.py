import numpy as np
import sys
import os
import argparse

dirname = os.path.dirname('encoder.py')
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--grid", required=True)
args = parser.parse_args()
grid = np.loadtxt(os.path.join(dirname,args.grid), dtype=int)
def createMDP(grid):
	M,N = grid.shape
	S = 0
	STATES = []
	# start = -1
	end = []
	s = 0
	for i in range(M):
		for j in range(N):
			if grid[i,j]==2:
				start = (s,i*N+j)
			if grid[i,j]==3:
				end.append((s,i*N+j))
			if grid[i,j]!=1:
				STATES.append((s,i*N+j))
				s+=1
	print('numStates',s)
	# ACTIONS = ['N','S','W','E']
	ACTIONS = ['0','1','2','3']
	A = len(ACTIONS)
	print('numActions',A)
	print('start',start[0])
	print('end',*[x for (x,y) in end])
	for i in range(M):
		for j in range(N):
			if i*N+j in [y for (x,y) in end]:
				continue
			if grid[i,j]==1:
				continue
			S_ = [(i-1)*N+j,(i+1)*N+j,i*N+j-1,i*N+j+1]
			lstate = [x for (x, y) in STATES if y == i*N+j][0]
			for k in range(len(S_)):
				if S_[k] in [y for (x,y) in end]:
					print('transition',lstate, ACTIONS[k], [x for (x, y) in STATES if y == S_[k]][0], -1, 1)
				elif S_[k] in [y for (x,y) in STATES]:
					print('transition',lstate, ACTIONS[k], [x for (x, y) in STATES if y == S_[k]][0], -1, 1)
				else:
					print('transition',lstate,ACTIONS[k],lstate,-1,1)
	if not end:
		print('mdptype continuing')
	else:
		print('mdptype episodic')
	print('discount 0.9')

if __name__=='__main__':
	createMDP(grid)

