import blackjack
from pylab import *
import numpy

Q1 = numpy.random.uniform(low = 0, high = 0.01, size=(181,2))
Q2 = numpy.random.uniform(low = 0, high = 0.01, size=(181,2))

def learn(alpha, eps, numTrainingEpisodes):
    returnSum = 0.0
    totalStay = 0
    totalHit = 0
    for episodeNum in range(numTrainingEpisodes):
        G = 0

        # Fill in Q1 and Q2
        currentState = blackjack.init()
        
        terminated = False
        
        while(not terminated):
            action = -1
            #blackjack.printState(currentState)
            
            #Choose A from S using policy derived from Q1 and Q2 (epsilon greedy in Q1+Q2)
            
            #Decide to explore vs. Exploit
            randomE = random()                
            if (randomE < eps):
                #explore
                action = randint(0,2)    
            else:
                #Exploit / Choose the best current action
                action= policy(currentState)

            #print("Action: " + str(action))
            if (action ==0):
                totalStay = totalStay+1
            else:
                totalHit = totalHit + 1

                
            """
            if (currentState > 0):
                if (action == 0):
                    print("Action: Stay")
                else:
                    print("Action: Hit")
            else:
                print("Action: Hit (until 10 or more)")
            """
            
            #Take action A, observe R, S'
            transitionTuple = blackjack.sample(currentState, action)
            newState = transitionTuple[1]
            reward = transitionTuple[0]
            newStateValue = 0
            
            #Do learning - updating Q values
            randomQ = random()
            if (randomQ > 0.5):
                #print("Using Q1")
                nextStateValue = 0
                if (newState):
                    #Non terminal
                    q1BestNextAction = argmax(Q1[newState])                    
                    newStateValue = Q2[newState, q1BestNextAction]
                """
                print("Before update:" + str(Q1[currentState, action]))
                print("Reward:" + str(reward))
                print("Next state value: " + str(newStateValue))                    
                """
                Q1[currentState, action] = Q1[currentState, action] + alpha*(reward + newStateValue - Q1[currentState, action])
                #print("After update:" + str(Q1[currentState, action]))                
            else:
                #print("Using Q2")                
                if (newState):
                    #Non terminal
                    q2BestNextAction = argmax(Q2[newState])                    
                    newStateValue = Q1[newState, q2BestNextAction]        
                """
                print("Before update:" + str(Q2[currentState, action]))                
                print("Reward:" + str(reward))
                print("Next state value: " + str(newStateValue))                    
                """                        
                Q2[currentState, action] = Q2[currentState, action] + alpha*(reward + newStateValue - Q2[currentState, action])                
                #print("After update:" + str(Q2[currentState, action]))                
                            
            currentState = newState
            if (currentState == False):
                G = reward
                terminated = True
                #print("Finished episode with reward: " + str(G))
            
        """
        print("Episode: ", episodeNum, "Return: ", G)
        print("=======================")    
        print("")
        """
        returnSum = returnSum + G
   
        if episodeNum % 10000 == 0 and episodeNum != 0:
            print("Average return so far: ", returnSum/episodeNum)

    print("Total hit: " + str(totalHit))
    print("Total stay: " + str(totalStay))

def simpleEvaluate(numEvaluationEpisodes):
    learn(0, 0, numEvaluations)
    
def evaluate(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0
        ...
        # Use deterministic policy from Q1 and Q2 to run a number of
        # episodes without updates. Return average return of episodes.
        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes
    
def policy(state):
    q1 = Q1[state]
#    print("q1: " + str(q1))
    q2 = Q2[state]
#    print("q2: " + str(q2))
    q = [q1[0] + q2[0], q1[1] + q2[1]]
#    print("q: " + str(q))
    action = numpy.argmax(q)
    return action
    
