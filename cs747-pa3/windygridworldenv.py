import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.axes import Subplot as sp
import matplotlib.cm as cm
import matplotlib.animation as animation
import pylab as pl
from pylab import imread
import matplotlib.gridspec as gridspec

def gridnormal():
	shape = (7,10)
	start = (3,0)
	goal = (3,7)
	S = shape[0]*shape[1]
	A = 4 #]N,W,S,E]
	moves = [(-1,0),(0,-1),(1,0),(0,1)]
	winds = np.zeros(shape)
	winds[:,[3,4,5,8]] = 1
	winds[:,[6,7]] = 2

	T = np.zeros((S,A))
	R = np.zeros((S,A))
	end = False
	for s in range(S):
		pos = np.unravel_index(s,shape)
		for a in range(A):
			newpos = np.array(pos) + np.array(moves[a]) + np.array([-1,0])*winds[pos]
			newpos[0] = max(0,min(newpos[0],shape[0]-1))
			newpos[1] = max(0,min(newpos[1],shape[1]-1))
			s_ = np.ravel_multi_index(tuple([int(i) for i in newpos]),shape)
			T[s,a] = s_
			R[s,a] = -1
			
			if s_== np.ravel_multi_index(tuple(goal),shape):
				end = True
				R[s,a] = 0
	return T,R,goal,start		

def sarsa(T,R,goal,init):
	shape = (7,10)
	S,A = T.shape[0],T.shape[1]
	Q = np.random.uniform(0,1,(S,A))
	num_episodes = 1001
	Time = 0
	eps = 0.1
	alpha = 0.5
	gamma = 1
	Reward = 0
	img = []
	xx, xy = [],[]
	for episode in range(num_episodes):
		t = 0
		s = np.ravel_multi_index(tuple(init),shape)
		eps = min(1/(episode+1),0.1)
		a = np.argmax(Q[s]) if np.random.random()>eps else np.random.choice(range(T.shape[1]))
		while True:
			if episode==1000:
				img.append(render(s))
			# # print(T)
			s_,r = int(T[s,a]), R[s,a]
			a_ = np.argmax(Q[s_]) if np.random.random()>eps else np.random.choice(range(T.shape[1]))
			Q[s,a] += alpha*(r + gamma*Q[s_,a_] - Q[s,a])
			Reward += r
			t += 1
			# print(s_,T[s_],Q[s_])
			s = s_
			a = a_
			if s_ == np.ravel_multi_index(tuple(goal),shape):
				if episode==1000:
					img.append(render(s))
				break
		Time+=t
	# 	xx.append(Time)
	# 	xy.append(episode)
	# 	print(t,episode)
	# plt.plot(xx,xy)
	# plt.show()
	return img


def render(s=np.ravel_multi_index((3,0),(7,10))):
	shape = (7,10)
	maze = np.zeros(shape)
	maze[:,[3,4,5,8]] = -1
	maze[:,[6,7]] = -2
	maze[np.unravel_index(s,shape)] = 2.0
	maze[(3,7)] = 1.0
	img = np.array(maze, copy=True)
	return img

if __name__ == '__main__':
	# gridnormal()
	# img = render()
	T,R,goal,init = gridnormal()
	img = sarsa(T,R,goal,init)
	# img = [] # some array of images
	st = None
	for im in img:
		bc = im
	
		if st is None:
			fig = pl.figure(figsize = (5,5))
			ax = pl.subplot(gridspec.GridSpec(1,1)[:,0])
			ax.get_xaxis().set_visible(True)
			ax.get_yaxis().set_visible(True)
			ax.set_xticks(np.arange(-.5,10,1),minor=True)
			ax.set_yticks(np.arange(-.5,7,1),minor=True)
			ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
			st = pl.imshow(bc)
		else:
			st.set_data(bc)
		pl.pause(.1)
		pl.draw()
	# plt.show()



