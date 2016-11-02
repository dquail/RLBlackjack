import blackjack
from pylab import *

def run(numEvaluationEpisodes):
    returnSum = 0.0
    for episodeNum in range(numEvaluationEpisodes):
        G = 0

        currentState = blackjack.init()
        
        terminated = False
        
        while(not terminated):
            blackjack.printState(currentState)
            
            action = randint(0,2)
            if (currentState > 0):
                if (action == 0):
                    print("Action: Stay")
                else:
                    print("Action: Hit")
            else:
                print("Action: Hit (until 10 or more)")
                
            transitionTuple = blackjack.sample(currentState, action)
            newState = transitionTuple[1]

            reward = transitionTuple[0]
            currentState = newState
            if (currentState == False):
                G = reward
                terminated = True
                print("Finished episode with reward: " + str(G))
            
            
        print("Episode: ", episodeNum, "Return: ", G)
        print("=======================")    
        print("")
        returnSum = returnSum + G
    return returnSum / numEvaluationEpisodes
