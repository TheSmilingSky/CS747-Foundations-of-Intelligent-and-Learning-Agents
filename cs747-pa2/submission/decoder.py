import numpy as np
import argparse
import sys
import os

dirname = os.path.dirname('decoder.py')

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--grid", required=True)
parser.add_argument("-vp", "--value_policy", required=True)
args = parser.parse_args()

grid = np.loadtxt(os.path.join(dirname,args.grid), dtype=int)
vp = np.loadtxt(os.path.join(dirname,args.value_policy), dtype=float)
# print(vp[2,1])
def nextstate(ystate,action,N,STATES):
	# print(ystate+N,len(STATES))
	if action==0:
		return [x for (x, y) in STATES if y == ystate-N][0]
	if action==1:
		return [x for (x, y) in STATES if y == ystate+N][0]
	if action==2:
		return [x for (x, y) in STATES if y == ystate-1][0]
	if action==3:
		return [x for (x, y) in STATES if y == ystate+1][0]

def path(grid,vp):
	M,N = grid.shape
	S = 0
	STATES = []
	# start = -1
	end = []
	s = 0
	actions = ['N','S','W','E']
	for i in range(M):
		for j in range(N):
			if grid[i,j]==2:
				start = (s,i*N+j)
			if grid[i,j]==3:
				end.append((s,i*N+j))
			if grid[i,j]!=1:
				STATES.append((s,i*N+j))
				s+=1
	state = start[0]
	# print(end)
	while state not in [x for (x,y) in end]:
		# print(vp[state])
		# print(state)
		# print(int(vp[state,1]))
		print(actions[int(vp[state][1])],end=" ")
		currentaction = actions[int(vp[state,1])]
		ystate = [y for (x, y) in STATES if x == state][0]
		state = nextstate(ystate,int(vp[state,1]),N,STATES)
		# print(state)
if __name__=='__main__':
	path(grid,vp)