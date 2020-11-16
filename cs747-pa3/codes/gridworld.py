import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import argparse
import math
import pylab as pl

parser = argparse.ArgumentParser()
parser.add_argument("-env", "--environment", required=False, default='gridnormal')
parser.add_argument("-sto", "--stochasticity", required=False, default=False)
parser.add_argument("-algo", "--algorithm", required=False, default='Sarsa')
parser.add_argument("-eps", "--epsilon", required=False, default=0.1)
parser.add_argument("-alpha","--alpha", required=False, default=0.5)
parser.add_argument("-num_epi", "--num_episodes", required=False, default=200)
parser.add_argument("-gamma", "--gamma", required=False, default=1)
parser.add_argument("-vis", "--visualise", required=False, default=0)
parser.add_argument("-rndm", "--randomseed", required=False, default=10)
args = parser.parse_args()

def gridnormal():
	shape = (7,10)
	start = (3,0)
	goal = (3,7)
	S = shape[0]*shape[1]
	A = 4 #[N,W,S,E]
	moves = [(-1,0),(0,-1),(1,0),(0,1)]
	winds = np.zeros(shape)
	winds[:,[3,4,5,8]] = 1
	winds[:,[6,7]] = 2
	return shape,S,A,moves,winds,start,goal

def gridking():
	shape = (7,10)
	start = (3,0)
	goal = (3,7)
	S = shape[0]*shape[1]
	A = 8 #[N,NW,W,SW,S,SE,E,NE]
	moves = [(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
	winds = np.zeros(shape)
	winds[:,[3,4,5,8]] = 1
	winds[:,[6,7]] = 2	
	return shape,S,A,moves,winds,start,goal

def update(MDP,s,a,stochastic):
	shape,S,A,moves,winds,start,goal = MDP
	pos = np.unravel_index(s,shape)
	if stochastic and winds[pos]:
		wind = winds[pos] + np.random.randint(-1,2)
	else:
		wind = winds[pos]
	newpos = np.array(pos) + np.array(moves[a]) + np.array([-1,0])*wind
	newpos[0] = max(0,min(newpos[0],shape[0]-1))
	newpos[1] = max(0,min(newpos[1],shape[1]-1))
	s_ = np.ravel_multi_index(tuple([int(i) for i in newpos]),shape)
	return (int(s_),0) if s_== np.ravel_multi_index(tuple(goal),shape) else (int(s_),-1)

def sarsa(MDP,params,stochastic,vis):
	shape,S,A,moves,winds,start,goal = MDP
	Q = np.zeros((S,A))
	eps, alpha, gamma, num_episodes = params
	x = (num_episodes-1)**(1/vis) if vis else 0
	Visuals = []
	if x:
		for i in range(1,vis+1):
			Visuals.append(min(round(x**i),num_episodes))
	Time = 0
	pTime, pEpi = [0],[0]

	for episode in range(num_episodes):
		t = 0
		s = np.ravel_multi_index(tuple(start),shape)
		if episode in Visuals:
			imgs = []
			imgs.append(render(s))
		# eps = min(10/(episode+1),0.1)
		a = np.argmax(Q[s]) if np.random.random()>eps else np.random.choice(range(A))
		while s != np.ravel_multi_index(tuple(goal),shape):
			s_,r = update(MDP,s,a,stochastic)
			a_ = np.argmax(Q[s_]) if np.random.random()>eps else np.random.choice(range(A))
			Q[s,a] += alpha*(r + gamma*Q[s_,a_] - Q[s,a])
			t += 1
			s,a = s_,a_
			if episode in Visuals:
				imgs.append(render(s))
		Time+=t
		if episode in Visuals:
			# print(episode)
			visualise(imgs)
		pTime.append(Time)
		pEpi.append(episode)
		# print(t,episode)
	return pTime, pEpi


def qlearning(MDP,params,stochastic,vis):
	shape,S,A,moves,winds,start,goal = MDP
	Q = np.zeros((S,A))
	for a in range(A):
		Q[np.ravel_multi_index(tuple(goal),shape),a] = 0
	eps, alpha, gamma, num_episodes = params
	x = (num_episodes-1)**(1/vis) if vis else 0
	Visuals = []
	if x:
		for i in range(1,vis+1):
			Visuals.append(min(round(x**i),num_episodes))
	Time = 0
	pTime, pEpi = [0],[0]

	for episode in range(num_episodes):
		t = 0
		s = np.ravel_multi_index(tuple(start),shape)
		# eps = min(10/(episode+1),0.1)
		if episode in Visuals:
			imgs = []
			imgs.append(render(s))
		while s != np.ravel_multi_index(tuple(goal),shape):
			a = np.argmax(Q[s]) if np.random.random()>eps else np.random.choice(range(A))
			s_,r = update(MDP,s,a,stochastic)
			Q[s,a] += alpha*(r + gamma*np.max(Q[s_]) - Q[s,a])
			t += 1
			s = s_
			if episode in Visuals:
				imgs.append(render(s))
		Time+=t
		if episode in Visuals:
			visualise(imgs)
		pTime.append(Time)
		pEpi.append(episode)
		# print(t,episode)
	return pTime, pEpi

def expectedsarsa(MDP,params,stochastic,vis):
	shape,S,A,moves,winds,start,goal = MDP
	Q = np.zeros((S,A))
	for a in range(A):
		Q[np.ravel_multi_index(tuple(goal),shape),a] = 0
	eps, alpha, gamma, num_episodes = params
	x = (num_episodes-1)**(1/vis) if vis else 0
	Visuals = []
	if x:
		for i in range(1,vis+1):
			Visuals.append(min(round(x**i),num_episodes))
	Time = 0
	pTime, pEpi = [0],[0]

	for episode in range(num_episodes):
		t = 0
		s = np.ravel_multi_index(tuple(start),shape)
		# eps = min(10/(episode+1),0.1)
		if episode in Visuals:
			imgs = []
			imgs.append(render(s))
		while s != np.ravel_multi_index(tuple(goal),shape):
			a = np.argmax(Q[s]) if np.random.random()>eps else np.random.choice(range(A))
			s_,r = update(MDP,s,a,stochastic)
			exp = eps*np.mean(Q[s_]) + (1-eps)*np.max(Q[s_])
			Q[s,a] += alpha*(r + gamma*exp - Q[s,a])
			t += 1
			s = s_
			if episode in Visuals:
				imgs.append(render(s))

		Time+=t
		if episode in Visuals:
			visualise(imgs)
		pTime.append(Time)
		pEpi.append(episode)
		# print(t,episode)
	return pTime, pEpi

def visualise(img):
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

def render(s=np.ravel_multi_index((3,0),(7,10))):
	shape = (7,10)
	maze = np.zeros(shape)
	maze[:,[3,4,5,8]] = -1
	maze[:,[6,7]] = -2
	maze[np.unravel_index(s,shape)] = 2.0
	maze[(3,7)] = 1.0
	img = np.array(maze, copy=True)
	return img

def plot_and_save(Times, Epis, A, stochastic, algos):
	fig, ax = plt.subplots()
	# print(algos)
	for i in range(len(algos)):
		ax.plot(Times[i],Epis[i],label=str((1-stochastic)*"Non-") + "Stochastic " + str(algos[i]) + " for " + str(A) + " Moves")
	ax.grid()
	ax.legend()
	name = ''
	for i in range(len(algos)):
		name += str(algos[i])
		name += " "
	plt.savefig('../plots/' + str((1-stochastic)*"Non-") + "Stochastic " + name + " for " + str(A) + " Moves" + '.png')
	# plt.show()

if __name__ == '__main__':
	if args.environment == 'gridnormal':
		MDP = gridnormal()
	elif args.environment == 'gridking':
		MDP = gridking()

	stochastic = True if args.stochasticity=='True' else False
	params = [float(args.epsilon), float(args.alpha), float(args.gamma), int(args.num_episodes)]
	Times, Epis, algos = [], [], []
	args.algorithm = args.algorithm.split(',')
	if 'Sarsa' in args.algorithm:
		episodes = None
		for i in range(int(args.randomseed)):
			np.random.seed(i)
			tSarsa, epiSarsa = sarsa(MDP, params, stochastic, int(args.visualise))
			if episodes == None:
				episodes = epiSarsa.copy()
			else:
				episodes = [episodes[j] + epiSarsa[j] for j in range(params[-1]+1)]
		Times.append(tSarsa)
		Epis.append([i/int(args.randomseed) for i in episodes])
		algos.append('Sarsa')
		
	if 'Qlearning' in args.algorithm:
		episodes = None
		for i in range(int(args.randomseed)):
			np.random.seed(i)
			tQ, epiQ = qlearning(MDP, params, stochastic, int(args.visualise))
			if episodes == None:
				episodes = epiQ.copy()
			else:
				episodes = [episodes[j] + epiQ[j] for j in range(params[-1]+1)]
		Times.append(tQ)
		Epis.append([i/int(args.randomseed) for i in episodes])
		algos.append('Q learning')

	if 'ExpectedSarsa' in args.algorithm:
		episodes = None
		for i in range(int(args.randomseed)):
			np.random.seed(i)
			tExpSarsa, epiExpSarsa = expectedsarsa(MDP, params, stochastic, int(args.visualise))
			if episodes == None:
				episodes = epiExpSarsa
			else:
				episodes = [episodes[j] + epiExpSarsa[j] for j in range(params[-1]+1)]
		Times.append(tExpSarsa)
		Epis.append([i/int(args.randomseed) for i in episodes])
		algos.append('Expected Sarsa')
		
	plot_and_save(Times, Epis, MDP[2], stochastic,algos)




