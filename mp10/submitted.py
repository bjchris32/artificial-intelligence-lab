'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def is_boundary(model, r, c):
    if r < 0 or r > (model.M - 1) or c < 0 or c > (model.N - 1):
        return True
    else:
        return False

def is_wall(model, r, c):
    if model.W[r, c] == True:
        return True
    else:
        return False

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''

    # will be along the intended direction with probability model.D[r, c, 0],
    # will be at the right angles to the intended direction with probability model.D[r, c, 1] (counter-clockwise)
    # model.D[r, c, 2] (clockwise).

    # print("model.D = ", model.D)
    # print("model.D.shape = ", model.D.shape)
    M = model.M
    N = model.N

    transition_matrix = np.zeros(shape=(M, N, 4, M, N))

    # action_idx => [counter_clockwise_rotation, clockwise_rotation]
    rotate_dic = {
        0: { 'counter_clokwise': [+1, 0], 'clock_wise': [-1, 0]},
        1: { 'counter_clokwise': [0, -1], 'clock_wise': [0, +1]},
        2: { 'counter_clokwise': [-1, 0], 'clock_wise': [+1, 0]},
        3: { 'counter_clokwise': [0, +1], 'clock_wise': [0, -1]},
    }

    for r in range(M):
        for c in range(N):
            # do not move at terminal
            if model.T[r, c] == True:
                continue
            # start moving with each action
            for action_idx, action in enumerate([[0, -1], [-1, 0], [0, +1], [+1, 0]]):
                go_intended_prob = model.D[r, c, 0]
                go_counterclock_prob = model.D[r, c, 1]
                go_clock_prob = model.D[r, c, 2]
                # if r == 1 and c == 0 and action_idx == 2:
                #     print("action_idx = ", action_idx)
                #     print("go_intended_prob = ", go_intended_prob)
                #     print("go_counterclock_prob = ", go_counterclock_prob)
                #     print("go_clock_prob = ", go_clock_prob)

                try:
                    # go intended direction
                    next_r, next_c = r + action[0], c + action[1]
                    if is_boundary(model, next_r, next_c) or is_wall(model, next_r, next_c):
                        transition_matrix[r, c, action_idx, r, c] = go_intended_prob
                    else:
                        transition_matrix[r, c, action_idx, next_r, next_c] = go_intended_prob
                except Exception as err:
                    print("when next action = ", action_idx, " at..... r = ", r, ", c = ",  c)
                    print("next_r = ", next_r, ", next_c = ", next_c)
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise

                try:
                    # go counter clock direction
                    counter_clock_r, counter_clock_c = r + rotate_dic[action_idx]['counter_clokwise'][0], c + rotate_dic[action_idx]['counter_clokwise'][1]
                    if is_boundary(model, counter_clock_r, counter_clock_c) or is_wall(model, counter_clock_r, counter_clock_c):
                        transition_matrix[r, c, action_idx, r, c] += go_counterclock_prob
                    else:
                        transition_matrix[r, c, action_idx, counter_clock_r, counter_clock_c] += go_counterclock_prob
                except Exception as err:
                    print("when counter clock action = ", action_idx, "at ..... r = ", r, ", c = ",  c)
                    print("counter_clock_r = ", counter_clock_r, ", counter_clock_c = ",  counter_clock_c)
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise

                try:
                    # go clock direction
                    clock_wise_r, clock_wise_c = r + rotate_dic[action_idx]['clock_wise'][0], c + rotate_dic[action_idx]['clock_wise'][1]
                    if is_boundary(model, clock_wise_r, clock_wise_c) or is_wall(model, clock_wise_r, clock_wise_c):
                        transition_matrix[r, c, action_idx, r, c] += go_clock_prob
                    else:
                        transition_matrix[r, c, action_idx, clock_wise_r, clock_wise_c] += go_clock_prob
                except Exception as err:
                    print("when clock action = ", action_idx, " at..... r = ", r, ", c = ",  c)
                    print("clock_wise_r = ", clock_wise_r, ", clock_wise_c = ",  clock_wise_c)
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise

    return transition_matrix

def get_utility_with_action(model, current_r, current_c, P, U_current, action_idx):
    M = model.M
    N = model.N

    # if current_r == 1 and current_c == 0 and action_idx == 2:
    #     print("at r = 1, c = 0, action = 2, transition P is =====")
    #     model.visualize(P[current_r, current_c, action_idx, :, :])
    #     print("U_current ======= ")
    #     model.visualize(U_current)

    sum = 0
    for next_r in range(M):
        for next_c in range(N):
            sum += P[current_r, current_c, action_idx, next_r, next_c] * U_current[next_r, next_c]
    return sum

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    gamma = model.gamma
    U_next = np.zeros(shape=U_current.shape)

    # iterate each loc
        # iterate each action to get the max of (sum each s' of P(s' | s,a)*U(s'))
        # Q: how do we get U(s')?
    M = model.M
    N = model.N
    R = model.R
    best_action = -1

    for r in range(M):
        for c in range(N):
            # calculate U_next[r, c]
            # terminal state do not take any action
            if model.T[r, c] == True:
                # print("terminal at r = ", r, ", c = ", c, " is set to be model.R[r, c] = ", model.R[r, c])
                U_next[r, c] = model.R[r, c]
                continue
            # non-terminal state
            max_utility = np.NINF
            for action_idx in range(4):
                temp_utility = get_utility_with_action(model, r, c, P, U_current, action_idx)
                if temp_utility > max_utility:
                    best_action = action_idx
                    max_utility = temp_utility
            U_next[r, c] = R[r, c] + gamma * max_utility
            # print("best action at r=", r, "c = ", c, ", action = ", best_action)

    return U_next
    # raise RuntimeError("You need to write this part!")

def is_converge(U_previous, U_current):
    for r in range(len(U_current)):
        for c in range(len(U_current[0])):
            # check converge
            if np.abs(U_previous[r, c] - U_current[r, c]) >= epsilon:
                return False
    return True


def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    M = model.M
    N = model.N
    R = model.R

    # initialize P and U_current
    U_current = np.zeros(shape=(M, N))
    # U_current = R
    P = compute_transition_matrix(model)

    # print("model.gamma == ", model.gamma)
    # print("R === ", )
    # model.visualize(R)
    # print("P ==== P[1, 0, 2, :, :]")
    # model.visualize(P[1, 0, 2, :, :])


    # while there is any s which does not converge in U_current
        # update all U_current to U_next
        # update_utility(model, P, U_current)
    num_of_iterations = 0
    while(True):
        # print("keep iteration ... ", num_of_iterations)
        if num_of_iterations == 100:
            break
        U_previous = U_current
        # print("previous utility =====")
        # model.visualize(U_previous)
        U_current = update_utility(model, P, U_previous)
        # print("after update utility ==== ")
        # model.visualize(U_current)
        if is_converge(U_previous, U_current):
            break
        num_of_iterations += 1
        # if num_of_iterations == 2:
        #     break

    return U_current

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
