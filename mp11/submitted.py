'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import random
import numpy as np
import torch
import torch.nn as nn


NUM_ACTIONS = 3

class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        # The state will be a list of 5 integers,
        # such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        self.number_of_states = (int)(np.prod(state_cardinality))
        self.q = np.zeros(shape = (self.number_of_states, NUM_ACTIONS))
        self.n = np.zeros(shape = (self.number_of_states, NUM_ACTIONS))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nfirst = nfirst
        self.state_cardinality = state_cardinality

        # print("self.number_of_states = ", self.number_of_states)
        # print("self.q.shape = ", self.q.shape)


    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        # get_state_idx = (int)(np.prod(state))
        # print("get_state_idx = ", get_state_idx)
        # print("method = ", self.get_state_idx(state))
        return self.n[self.get_state_idx(state)]

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        state_idx = self.get_state_idx(state)
        # print("self.n[state_idx] = ", self.n[state_idx])
        action_list = np.array((self.n[state_idx] < self.nfirst).nonzero()).flatten()

        if len(action_list) == 0:
            return None

        # print("action_list = ", action_list.shape)
        # update self.n before returning the mapped action
        action_idx = np.random.choice(action_list)
        self.n[state_idx, action_idx] += 1
        mapped_action = action_idx - 1
        return mapped_action

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        state_idx = self.get_state_idx(state)
        return self.q[state_idx]

    # Q: should I check if the actions in newstate is legal?
    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q
        '''
        # iterate the action in the newsate
        newstate_idx = self.get_state_idx(newstate)
        # print("self.q[newstate_idx] = ", self.q[newstate_idx])
        max_of_q_next_state = max(self.q[newstate_idx])
        # print("max_of_q_next_state = ", max_of_q_next_state)
        Q_local = reward + self.gamma * max_of_q_next_state
        return Q_local

    # Q: Just pass newstate to q_local? Or, I need to do the given action on newstate?
    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''
        state_idx = self.get_state_idx(state)
        # new_state_idx = self.get_state_idx(newstate)
        mapped_action = action + 1
        original_q = self.q[state_idx, mapped_action]
        q_local_next_state = self.q_local(reward, newstate)
        self.q[state_idx, mapped_action] = original_q + self.alpha * (q_local_next_state - original_q)

        # print("q_local_next_state = ", q_local_next_state, ", original_q = ", original_q)
        # print("learned q with mapped_action ", mapped_action, " = ", self.q[state_idx, mapped_action])
        # print("self.q[state_idx] = ", self.q[state_idx])

        return self.q[state_idx, mapped_action]
    
    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        np.savez(filename, q_saved = self.q, n_saved = self.n)
        
    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        loaded_arrays = np.load(filename, allow_pickle = True)
        self.q = loaded_arrays['q_saved']
        self.n = loaded_arrays['n_saved']
        
    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''
        state_idx = self.get_state_idx(state)
        sate_q_values = self.q[state_idx]
        max_q_idx = np.argmax(sate_q_values)
        # print("max_q_idx = ", max_q_idx)
        max_q = self.q[state_idx, max_q_idx]
        out_action = max_q_idx - 1
        # print("max_q = ", max_q)

        return (out_action, max_q)
    
    def act(self, state):
        '''
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        explorable_action = self.choose_unexplored_action(state)
        if explorable_action is not None:
            return explorable_action

        # keep explore with prob = epsilon
        random_action_prob = np.random.random()
        if random_action_prob < self.epsilon:
            # choose action uniformly at random
            # print("choose action uniformly at random with random_action_prob = ", random_action_prob, "self.epsilon = ", self.epsilon)
            # print("choose action uniformly at random")
            # return self.get_random_action(state)
            action_idx = np.random.choice(NUM_ACTIONS)
            mapped_action = action_idx - 1
            return mapped_action
        else:
            # choose action with best Q
            # print("choose action with best Q")
            # print("choose action with best Q")
            max_q_action = self.exploit(state)[0]
            return max_q_action

    def get_state_idx(self, state):
        return (int)(np.prod(state))
        
class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        raise RuntimeError('You need to write this!')

    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.
        
        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        raise RuntimeError('You need to write this!')
        
    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.
        
        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state
        
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
        
    def load(self, filename):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        raise RuntimeError('You need to write this!')
