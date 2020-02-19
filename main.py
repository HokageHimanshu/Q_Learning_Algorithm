# Himanshu Singh 2017291
import gym
import numpy as np 
from agent import Agent


env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n,env.action_space.n])
print('q table dimensions are '+str(env.observation_space.n)+' X '+str(env.action_space.n))
lr = .6
gamma = .9
epsilon=0.4
noOfGames = 10000
rewardList = []
# for plotting the graph, didWin, didLose , trackWin
winOrLose = np.zeros((noOfGames,3))
secretAgent=Agent(env,gamma,lr)

print('Learning..')
for i in range(noOfGames):
    state = env.reset()
    # print('Game '+str(i))
    totalReward = 0
    gameOver = False
    findTrial = 0
    # if i%1000:
    #     print('.', end =" ")  
    while not gameOver:
        # env.render()
        # findTrial+=1
        action= secretAgent.chooseAction(state,env.action_space,epsilon) 
        nextState,reward,gameOver,info = env.step(action)

        # NEGATIVE REWARD ----------------------------
        if gameOver == True and reward==0:
            reward=-1
            winOrLose[i,2]=winOrLose[i-1,2]
        #---------------------------------------------

        secretAgent.updateQtable(state,action,nextState,reward)
        # print('state ' +str(nextState))
        # print('action is' +str(action))
        # if reward==-1:
        #     reward=0
        totalReward += reward
        state = nextState
        if gameOver == True:
            if reward!=0 and reward!=-1: 
                # print('reward'+str(reward))
                winOrLose[i,0]=1
                if i!=0:
                    winOrLose[i,2]=winOrLose[i-1,2]+1
            else:
                winOrLose[i,2]=winOrLose[i-1,2]
                winOrLose[i,1]=1
            break
        # if findTrial==110:
        #     winOrLose[i,1]=1
        #     break
    rewardList.append(totalReward)
    # env.render()
print('Results of Learning..')
print("Reward Sum on all episodes (customised reward system)" + str(sum(rewardList)/noOfGames))
print("Final Values Q-Table")
print(secretAgent.Q)
secretAgent.plot(winOrLose,noOfGames)
secretAgent.plotLearningCurve(winOrLose,noOfGames)
