# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        currvalues = util.Counter()
        for i in range(self.iterations):
            currvalues = self.values.copy()
            states = self.mdp.getStates()

            for state in states:
                val_list=[]
                actions = self.mdp.getPossibleActions(state)
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    for action in actions:
                        val = 0
                        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
                        for nextstate, prob in transitions:
                            val += prob*(self.mdp.getReward(state,action,nextstate) + self.discount*currvalues[nextstate])
                        val_list.append(val)
                    self.values[state] = max(val_list)
                
                    
                    


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        val = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
        for nextstate, prob in transitions:
            val += prob*(self.mdp.getReward(state,action,nextstate) + self.discount*self.values[nextstate])
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None
        bestval = float("-inf")
        bestaction = 0
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            value = self.computeQValueFromValues(state,action)
            if value > bestval:
                bestval = value
                bestaction = action
        return bestaction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # TODO Only update state i for iteration i, instead of all iteration 
        currvalues = util.Counter()
        for i in range(self.iterations):
            currvalues = self.values.copy()
            val_list=[]
            states = self.mdp.getStates()
            lenStates = len(states)
            state = states[i%lenStates]
            actions = self.mdp.getPossibleActions(state)
            if self.mdp.isTerminal(state):
                self.values[state] = 0
            else:
                for action in actions:
                    val = 0
                    transitions = self.mdp.getTransitionStatesAndProbs(state,action)
                    for nextstate, prob in transitions:
                        val += prob*(self.mdp.getReward(state,action,nextstate) + self.discount*currvalues[nextstate])
                    val_list.append(val)
                self.values[state] = max(val_list)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Compute predecessors of all states 
        predecessors = {}
        
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for SnP in self.mdp.getTransitionStatesAndProbs(state, action):
                    if SnP[0] in predecessors:
                        predecessors[SnP[0]].add(state)
                    else:
                        predecessors[SnP[0]] = {state}

        # Initialize an empty priority queue

        queue = util.PriorityQueue()

        # For each non-terminal state
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                # Compute diff
                maxQ = max(self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s))
                diff = abs(self.values[s] - maxQ)
                # Push s into queue with priority -diff
                queue.push(s, -diff)

        # For iteration [0, self.iteration - 1]
        for iteration in range(self.iterations):
            # If queue is empty terminate
            if queue.isEmpty():
                break
            # Pop state s off queue
            s = queue.pop()
            # Update s (if it is not a terminal state) in self.values
            if not self.mdp.isTerminal(s):
                maxQ = max(self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s))
                self.values[s] = maxQ
            # For each predecessor p of s 
            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    # Compute diff
                    maxQ = max(self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s))
                    diff = abs(self.values[s] - maxQ)
                    # If diff > theta, push p into queue with priority -diff
                    if diff > self.theta:
                        queue.push(p, -diff)