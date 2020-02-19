import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
	def __init__(self,env,gamma,lr):
		self.gamma = gamma
		self.lr=lr
		self.Q=np.zeros((env.observation_space.n,env.action_space.n))

	
	def chooseAction(self,state,actionSpace,epsilon):
		if random.uniform(0, 1) < epsilon:
			return actionSpace.sample()

		else:
			maxValue = max(self.Q[state,:])
			actionIndex=[]
			for i in range(len(self.Q[state,:])):
				if maxValue==self.Q[state][i]:
					actionIndex.append(i)
			return random.choice(actionIndex)

	def updateQtable(self,state,action,nextState,reward):
		# self.Q[state][action]+= self.lr*(reward + self.gamma*np.max(self.Q[nextState,:]) - self.Q[state][action])
		self.Q[state][action]=(1-self.lr)*self.Q[state][action]+ self.lr*(reward + self.gamma*np.max(self.Q[nextState,:]))

	def plot(self,winLose,iterations):
		x = np.arange(iterations)
		plt.plot(x,winLose[:,0])
		# plt.plot(x,winLose[:,1])
		unique, counts = np.unique(winLose[:,0], return_counts=True)
		d=dict(zip(unique, counts))
		if d.get(1) is None:
			wins=0
		else:
			wins=d.get(1)
		unique, counts = np.unique(winLose[:,1], return_counts=True)
		d=dict(zip(unique, counts))
		# loses=d[1]
		if d.get(1) is None:
			loses=0
		else:
			loses=d.get(1)
		# plt.legend([str(wins)+' Wins by agent',str(loses)+' Loses by agent'], loc='upper left')
		plt.legend([str(wins)+' Wins by agent'], loc='upper left')
		
		plt.suptitle('Agent playing Frozen Lake using Q Learning')
		plt.title('Training after '+str(iterations)+' Games')
		plt.savefig("TrainingHistory.png") # save as png
		plt.show()

	def plotLearningCurve(self,winLose,iterations):
		x = np.arange(iterations)
		plt.plot(x,winLose[:,2])
		# plt.plot(x,winLose[:,1])
		unique, counts = np.unique(winLose[:,0], return_counts=True)
		d=dict(zip(unique, counts))
		if d.get(1) is None:
			wins=0
		else:
			wins=d.get(1)
		# print(max(winLose[:,2]))
		plt.xlabel('iterations')  
		plt.ylabel('no of games won') 
		plt.suptitle('Learning Curve of Agent playing Frozen Lake using Q Learning')
		plt.title('Training after '+str(iterations)+' Games, won '+str(wins)+' matches')
		plt.savefig("LearningCurve.png") # save as png
		plt.show()


		