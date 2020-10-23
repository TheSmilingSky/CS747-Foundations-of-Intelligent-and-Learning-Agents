import argparse
import random
import numpy as np
import math

######## used for data plotting #########
# import matplotlib.pyplot as plt
# import pandas as pd
######## used for data plotting #########

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--instance", required=True)
parser.add_argument("-al", "--algorithm", required=True)
parser.add_argument("-rs", "--randomSeed", required=True)
parser.add_argument("-ep", "--epsilon", required=True)
parser.add_argument("-hz", "--horizon", required=True)
args = parser.parse_args()



def read_instance(instance):
	p = []
	with open(instance) as f:
		for line in f:
			p.append(float(line))
	return p

def armReward(instance, chosenArm):
	p = read_instance(instance)
	if random.random() >= p[chosenArm]:
		return 0.
	else:
		return 1.

def eG(instance, eps, horizon, numArms):
	means = np.ones(numArms)
	counts = np.zeros(numArms)
	reward = 0
	
	for t in range(horizon):
		if random.random() > eps:
			#ties broken randomly
			maxArms = np.argwhere(means == np.amax(means))
			chosenArm = random.choice(maxArms)[0]
		else:
			chosenArm = random.randrange(numArms)
	
		currentReward = armReward(instance, chosenArm)
		reward += currentReward

		means[chosenArm] = (counts[chosenArm]*means[chosenArm] 
			+ currentReward)/(counts[chosenArm] + 1)
		counts[chosenArm] += 1

	return reward

def ucb(instance, horizon, numArms):
	means = np.zeros(numArms)
	counts = np.zeros(numArms)
	ucb = np.zeros(numArms)
	reward = 0

	# round robin first
	for t in range(numArms):
		currentReward = armReward(instance, t)
		reward += currentReward

		means[t] = currentReward
		counts[t] += 1
		ucb[t] = means[t] + math.sqrt(2*math.log(t+1)/counts[t])

	# now UCB algorithm
	for t in range(numArms,horizon):
		maxArms = np.argwhere(ucb == np.amax(ucb))
		chosenArm = random.choice(maxArms)[0]

		currentReward = armReward(instance, chosenArm)
		reward += currentReward

		means[chosenArm] = (counts[chosenArm]*means[chosenArm] 
			+ currentReward)/(counts[chosenArm] + 1)
		counts[chosenArm] += 1

		for i in range(numArms):
			ucb[i] = means[i] + math.sqrt(2*math.log(t+1)/counts[i])

	return reward

def kl(x,y):
	if x == 0:
		return -math.log(1-y)
	elif x == 1:
		return -math.log(y)
	elif y == 0 or y == 1:
		return float('inf')
	else:
		return x*math.log(x/y) + (1-x)*math.log((1-x)/(1-y))

def findQ(p, rhs):

	if p==1:
		return 1.
	tol = 0.0001
	minq = p
	maxq = 1-0.00001
	if abs(kl(p,minq)-rhs)<=tol:
		return minq
	elif abs(kl(p,maxq)-rhs)<=tol:
		return maxq
	q = 0.5*(minq+maxq)
	if abs(kl(p,q)-rhs)<=tol:
		return q
	while abs(kl(p,q)-rhs)>tol:
		if kl(p,q)>rhs:
			maxq = q
		elif kl(p,q)<rhs:
			minq = q
		q = 0.5*(minq + maxq)
		if abs(kl(p,q)-rhs<tol):
			return q


def klucb(instance, horizon, numArms):
	means = np.zeros(numArms)
	counts = np.zeros(numArms)
	ucb = np.ones(numArms)
	reward = 0

	# 2 round robins first so that ln(ln(t)) is well defined
	for t in range(2*numArms):
		currentReward = armReward(instance,t%numArms)
		reward += currentReward
		means[t%numArms] = (counts[t%numArms]*means[t%numArms] 
			+ currentReward)/(counts[t%numArms] + 1)
		counts[t%numArms] += 1
	rhs = (math.log(t+2) + 3.0*math.log(math.log(t+2)))/2
	for i in range(numArms):
		ucb[i] = findQ(means[i],rhs)

	for t in range(2*numArms,horizon):
		maxArms = np.argwhere(ucb == np.amax(ucb))
		chosenArm = random.choice(maxArms)[0]

		currentReward = armReward(instance, chosenArm)
		reward += currentReward

		means[chosenArm] = (counts[chosenArm]*means[chosenArm] 
			+ currentReward)/(counts[chosenArm] + 1)
		counts[chosenArm] += 1

		for i in range(numArms):
			rhs = (math.log(t+1) + 3.0*math.log(math.log(t+1)))/counts[i]
			ucb[i] = findQ(means[i],rhs)

	return reward

def thompson(instance, horizon, numArms):
	success = np.zeros(numArms)
	counts = np.zeros(numArms)
	means = np.zeros(numArms)
	reward = 0
	for t in range(numArms):
		currentReward = armReward(instance, t)
		reward += currentReward

		means[t] = currentReward
		counts[t] += 1

	for t in range(numArms, horizon):
		for i in range(numArms):
			means[i] = random.betavariate(1+success[i],1+counts[i]-success[i])
		maxArms = np.argwhere(means == np.amax(means))
		chosenArm = random.choice(maxArms)[0]

		currentReward = armReward(instance, chosenArm)
		if currentReward == 1:
			success[chosenArm] += 1
		reward += currentReward

		counts[chosenArm] += 1

	return reward

def thompson_hint(instance, horizon, numArms, q):
	beliefs = np.ones((numArms,numArms))*(1./numArms)
	reward = 0

	for t in range(horizon):
		maxArms = np.argwhere(beliefs[:,-1] == np.amax(beliefs[:,-1]))
		chosenArm = random.choice(maxArms)[0]
		currentReward = armReward(instance, chosenArm)
		reward += currentReward

		if currentReward == 1:
			beliefs[chosenArm] = np.multiply(beliefs[chosenArm],q)
		else:
			beliefs[chosenArm] = np.multiply(beliefs[chosenArm],[1-x for x in q])
		beliefs[chosenArm] /= sum(beliefs[chosenArm])

	return reward

def T3data(p, horizon = 102400):
	instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt"]
	for instance in instances:
		for eps in [0.0001]:
			sum = 0
			for s in range(50):
				random.seed(s)
				sum += eG(read_instance(instance), eps, horizon)
				print(eps, s)
			print(sum/50)


def chooseAlgoRegret(ins, algo, p , e, T):
	regret = max(p)*T
	if algo == 'epsilon-greedy':
		regret -= eG(ins, e, T, len(p))
	elif algo == 'ucb':
		regret -= ucb(ins, T, len(p))
	elif algo == 'kl-ucb':
		regret -= klucb(ins, T, len(p))
	elif algo == 'thompson-sampling':
		regret -= thompson(ins, T, len(p))
	elif algo == 'thompson-sampling-with-hint':
		regret -= thompson_hint(ins, T, len(p), sorted(p))
	else:
		print("Algo not found")
	return regret


def generateT1Data(eps = 0.02):

	instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt"]
	algorithms = ["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
	horizons = [100, 400, 1600, 6400, 25600, 102400]
	cnt = 0
	with open("outputDataT1x.txt", 'w+') as file:
		for instance in instances:
			p = read_instance(instance)
			print(p)
			for algo in algorithms:
				for horizon in horizons:
					for s in range(50):
						random.seed(s)
						string = str(instance) + ', ' +  str(algo) + ', ' + str(s) + ', ' + str(eps) + ', ' +  str(horizon) + ', ' + str(chooseAlgoRegret(instance, algo, p, eps, horizon))
						file.write(string + '\n')
						cnt += 1
						print("inst ", instance, " algo ", algo," horizon ", horizon, " seed ", s)

			
		print(cnt)

def generateT2Data(eps = 0.02):

	instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt"]
	algorithms = ["thompson-sampling", "thompson-sampling-with-hint"]
	horizons = [100, 400, 1600, 6400, 25600, 102400]
	cnt = 0
	with open("outputDataT2x.txt", 'w+') as file:
		for instance in instances:
			p = read_instance(instance)
			# print(p)
			for algo in algorithms:
				for horizon in horizons:
					for s in range(50):
						random.seed(s)
						string = str(instance) + ', ' +  str(algo) + ', ' + str(s) + ', ' + str(eps) + ', ' +  str(horizon) + ', ' + str(chooseAlgoRegret(instance, algo, p, eps, horizon))
						file.write(string + '\n')
						cnt += 1
						print("inst ", instance, " algo ", algo," horizon ", horizon, " seed ", s)

			
		print(cnt)

def load_data():
	instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt"]
	algorithms = ["epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
	horizons = [100, 400, 1600, 6400, 25600, 102400]
	df = pd.read_csv('outputDataT1.txt', sep = ', ', header = None)
	a = []
	for i in range(72):
		a.append(sum(df[5][50*i:50*(i+1)])/50)

	plt.plot(horizons,a[:6], label = 'epsilon-greedy')
	plt.plot(horizons,a[6:12], label = 'ucb')
	plt.plot(horizons,a[12:18], label = 'kl-ucb')
	plt.plot(horizons,a[18:24], label	= 'thompson-sampling')
	plt.xscale("log")
	plt.legend()
	plt.show()

	plt.plot(horizons,a[24:30], label = 'epsilon-greedy')
	plt.plot(horizons,a[30:36], label = 'ucb')
	plt.plot(horizons,a[36:42], label = 'kl-ucb')
	plt.plot(horizons,a[42:48], label	= 'thompson-sampling')
	plt.xscale("log")
	plt.legend()
	plt.show()

	plt.plot(horizons,a[48:54], label = 'epsilon-greedy')
	plt.plot(horizons,a[54:60], label = 'ucb')
	plt.plot(horizons,a[60:66], label = 'kl-ucb')
	plt.plot(horizons,a[66:], label	= 'thompson-sampling')
	plt.xscale("log")
	plt.legend()
	plt.show()

def load_data1():
	instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt"]
	algorithms = ["thompson-sampling", "thompson-sampling-with-hint"]
	horizons = [100, 400, 1600, 6400, 25600, 102400]
	df = pd.read_csv('outputDataT2.txt', sep = ', ', header = None)
	a = []
	for i in range(36):
		a.append(sum(df[5][50*i:50*(i+1)])/50)

	plt.plot(horizons,a[:6], label = 'thompson-sampling')
	plt.plot(horizons,a[6:12], label = 'thompson-sampling-with-hint')
	plt.xscale("log")
	plt.legend()
	plt.show()

	plt.plot(horizons,a[12:18], label = 'thompson-sampling')
	plt.plot(horizons,a[18:24], label	= 'thompson-sampling-with-hint')
	plt.xscale("log")
	plt.legend()
	plt.show()

	plt.plot(horizons,a[24:30], label = 'thompson-sampling')
	plt.plot(horizons,a[30:36], label	= 'thompson-sampling-with-hint')
	plt.xscale("log")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	seed_ = int(args.randomSeed)
	random.seed(seed_)
	np.random.seed(seed_)
	p = read_instance(args.instance)
	e = float(args.epsilon)
	T = int(args.horizon)
	regret = chooseAlgoRegret(args.instance, args.algorithm, p, e, T)
	print(regret)



