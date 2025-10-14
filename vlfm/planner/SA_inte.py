# stimulated annealing algorithm

import random
import time
import numpy as np
import cupy as cp
import json

def generate_neighbor(path, operation='swap'):
    """
    generate a neighbor path by applying the operation on the path.
    swap: randomly swap two cities in the path. 2 influences
    shift: randomly shift a subsequence of cities in the path. 3 influences
    reverse: randomly reverse a subsequence of cities in the path. 2 influences
    """
    new_path = path.copy()
    # if pathlength <= 1, no operation is available
    if len(new_path) <= 1:
        return new_path

    # if pathlength <= 2, only swap is available
    if len(new_path) <= 2:
        operation = 'swap'
        new_path = new_path[::-1]
        return new_path
    
    if operation == 'swap':
        idx1, idx2 = random.sample(range(len(new_path)), 2)
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
    
    elif operation == 'shift':
        sub_len = random.randint(1, len(new_path) - 1)
        start = random.randint(0, len(new_path) - sub_len)
        end = start + sub_len
        sub_sequence = new_path[start:end]
        new_path = new_path[:start] + new_path[end:]
        insert_pos = random.randint(0, len(new_path))
        new_path = new_path[:insert_pos] + sub_sequence + new_path[insert_pos:]
    
    elif operation == 'reverse':
        sub_len = random.randint(1, len(new_path) - 1)
        start = random.randint(0, len(new_path) - sub_len)
        end = start + sub_len
        sub_sequence = new_path[start:end]
        sub_sequence.reverse()
        new_path = new_path[:start] + sub_sequence + new_path[end:]
    
    else:
        raise ValueError("invalid operation type.")
    
    return new_path

# def generate_path_T(path, T): # DEPRECATED
#     '''
#     generate a new path by applying the operation on the path.
#     Temperature is used to control the acceptance of the new path.
#     '''
#     sttime = time.time()
#     operations = ['swap', 'shift', 'reverse']
#     influences = [2, 3, 2]
#     total_influence = 0
#     while total_influence <= T:
#         op_index = random.choices(range(3), k=1)[0]
#         total_influence += influences[op_index]
#         new_path = generate_neighbor(path, operations[op_index])
#     edtime = time.time()
#     # print('adj generation time:',edtime-sttime)
#     return new_path

def generate_path_T_multip(paths, T):
    '''
    generate a new path by applying the operation on the path.
    Temperature is used to control the acceptance of the new path.
    '''
    sttime = time.time()
    operations = ['swap', 'shift', 'reverse']
    influences = [2, 3, 2]
    new_paths = []

    paths_cp = np.array(paths)
    path_heads = paths_cp[:, 0].tolist()
    paths = paths_cp[:, 1:].tolist()
    
    for path in paths:
        total_influence = 0
        new_path = []
        while total_influence <= T:
            op_index = random.choices(range(3), k=1)[0]
            total_influence += influences[op_index]
            new_path = generate_neighbor(path, operations[op_index])
        new_path = [path_heads[0]] + new_path
        new_paths.append(new_path)
    
    edtime = time.time()
    # print('adj generation time:',edtime-sttime)
    return new_paths

def generate_random_paths(npaths, ncities, start=0):
    """
    generate npaths random paths with ncities cities.
    """
    paths = []
    for _ in range(npaths):
        path = list(range(ncities))
        path.remove(start)
        random.shuffle(path)
        paths.append(path)
        path.insert(0, start)
    return paths

def abs_angle(x1, x2, unit='deg'):
    '''
    calculate the absolute angle between two angles.
    unit: degree or radian
    '''
    if unit == 'deg':
        return np.abs((x1 - x2 + 180) % 360 - 180)
    elif unit == 'rad':
        return np.abs((x1 - x2 + np.pi) % np.pi*2 - np.pi)
    else:
        raise ValueError("invalid unit.")

def build_distance_matrix(waypoints, probs, ignore_rot=False):
    '''
    build the distance matrix of the waypoints.
    Args:
        waypoints: waypoints [[direction, x, y], ...]
        probs: view probabilities
        ignore_rot: whether to ignore the rotation cost. in this case, waypoints should be [[x, y], ...]
    Returns:
        distance_matrix: distance matrix
    '''
    sttime = time.time()
    distance_matrix = cp.zeros((len(waypoints), len(waypoints)))
    probs_rel = np.array(probs) / np.max(probs)

    if ignore_rot: # this may cause performance loss
        # add 0 as the direction of the waypoints
        waypoints = [[0, waypoint[0], waypoint[1]] for waypoint in waypoints]

    for i in range(len(waypoints)):
        for j in range(len(waypoints)):
            move_cost = np.linalg.norm(np.array(waypoints[i][1:]) - np.array(waypoints[j][1:]))
            rot_cost = abs_angle(waypoints[i][0], waypoints[j][0], unit='rad') / np.pi * 10
            distance_matrix[i, j] = (move_cost + rot_cost) * (1 - probs_rel[j]/2)
    edtime = time.time()
    # print('distance matrix build time:',edtime-sttime)
    return distance_matrix

def build_index_matrix_paths(paths, gamma=0.95):
    '''build the quick index matrix of the paths for reward calculation.'''
    nwaypoints = len(paths[0])
    index_matrix = np.zeros((len(paths), nwaypoints, nwaypoints))
    for i in range(len(paths)):
        for j in range(nwaypoints-1):
            index_matrix[i, paths[i][j], paths[i][j+1]] = (nwaypoints - j - 1) * gamma ** j
    return index_matrix

class SA_ENV:
    '''only used for the inference of the path.'''
    def __init__(self, dist_matrix):
        self.dist_matrix = dist_matrix

    def infer_cupy(self, paths, gamma=0.95):
        '''
        use cupy to speed up the inference.
        '''
        sttime = time.time()
        index_matrix = cp.array(build_index_matrix_paths(paths, gamma))
        # index_matrix is (npaths, nwaypoints, nwaypoints), dist_matrix is (nwaypoints, nwaypoints)
        # the slices of index_matrix dot dist_matrix is the cost of each path
        cost_matrix = cp.tensordot(index_matrix, self.dist_matrix, axes=([1, 2], [0, 1]))
        # cost_matrix is (npaths), the cost of each path
        edtime = time.time()
        # print('infer time:',edtime-sttime)
        # cost_matrix = cp.asnumpy(cost_matrix)
        return cost_matrix


class SA_ALGORITHM:
    def __init__(self, env, iter=400, T0=10, Tf=5.5, alpha=0.97, gamma=0.95, verbose=False):
        self.env = env
        self.iter = iter        # number of iterations (or particles)
        self.alpha = alpha      # cooling rate
        self.gamma = gamma      # discount
        self.verbose = verbose  # whether to print the process
        self.T0 = T0            # initial temperature
        self.Tf = Tf            # final temperature
        self.T = T0             # current temperature
        self.paths = generate_random_paths(iter, self.env.dist_matrix.shape[0])
        self.history = {'f': [], 'T': []}

    def reset(self):
        self.T = self.T0
        self.paths = generate_random_paths(self.iter, self.env.dist_matrix.shape[0])
        self.history = {'f': [], 'T': []}

    # def Metrospolis(self, f, f_new):
    #     if f_new <= f:
    #         return 1
    #     else:
    #         p = math.exp((f - f_new) / self.T)
    #         if random.random() < p:
    #             return 1
    #         else:
    #             return 0
            
    def Metrospolis_multi_cupy(self, f, f_new):
        f = cp.array(f)
        f_new = cp.array(f_new)
        mask = f_new <= f
        p = cp.exp((f - f_new) / self.T)
        mask = mask + (cp.random.random(f.shape) < p)
        return cp.asnumpy(mask)

    # def best(self):
    #     f_list = []
    #     for i in range(self.iter):
    #         f = infer(self.env, self.paths[i])
    #         f_list.append(f)
    #     f_best = min(f_list)
        
    #     idx = f_list.index(f_best)
    #     return f_best, idx
    
    def best_multi(self):
        # f = self.env.infer_cupy(self.paths, gamma=self.gamma)
        f = self.env.infer_cupy(self.paths)
        f_best = cp.min(f)
        idx = cp.argmin(f)
        return cp.asnumpy(f_best), cp.asnumpy(idx)

    def run(self):
        sttime = time.time()
        count = 0
        flag = True
        while self.T > self.Tf:
           
            # for i in range(self.iter): 
            #     f = infer(self.env, self.paths[i])
            #     path_new = generate_path_T(self.paths[i], self.T)
            #     f_new = infer(self.env, path_new)
            #     if self.Metrospolis(f, f_new):
            #         self.paths[i] = path_new

            # cupy acceleration
            f = self.env.infer_cupy(self.paths, gamma=self.gamma)

            # if flag == True:
            #     print('first infer time:',time.time()-sttime)
            #     flag = False

            paths_new = generate_path_T_multip(self.paths, self.T)
            f_new = self.env.infer_cupy(paths_new, gamma=self.gamma)
            mask = self.Metrospolis_multi_cupy(f, f_new)
            self.paths = [paths_new[i] if mask[i] else self.paths[i] for i in range(self.iter)]

            # ft, _ = self.best()
            ft, _ = self.best_multi()

            self.history['f'].append(ft)
            self.history['T'].append(self.T)

            # cooldown
            self.T = self.T * self.alpha
            count += 1
            if self.verbose:
                print(f"iter={count}, T={self.T}, f={ft}, path={self.paths[_]}")
            
        # f_best, idx = self.best()
        f_best, idx = self.best_multi()

        edtime = time.time()
        if self.verbose:
            print('time elapsed:',edtime-sttime)
            print(f"F={f_best}, path={self.paths[idx]}")
        return f_best, self.paths[idx]
    
    
class SA:
    '''a class wrapping the simulated annealing algorithm and the environment.'''
    def __init__(self, nparticles=400, T_init=10, T_final=5.5, alpha=0.97, gamma=0.95, verbose=False):
        # init params
        self.nparticles = nparticles
        # self.T_init = T_init
        # self.T_final = T_final
        # self.alpha = alpha
        # self.gamma = gamma
        
        env = SA_ENV(cp.zeros((10, 10)))
        self.sa = SA_ALGORITHM(env, iter=nparticles, T0=T_init, Tf=T_final, alpha=alpha, gamma=gamma, verbose=verbose)

        self.wram_up()

    def wram_up(self):
        # warm up all functinos which use cupy
        # use 10 waypoints by default
        build_distance_matrix(np.random.random((10, 3)).tolist(), np.random.random(10).tolist())
        self.sa.env.infer_cupy(generate_random_paths(self.nparticles, 10))

    def infer(self, waypoints, probs, formation=True):
        '''
        formation: 
          True: return in format: {'sorted_pts':[[(direction), x, y], ...], 'sorted_values':[probs, ...]}
          False: ruturn in format: (best_f, best_path in index)
        '''
        if len(waypoints) != len(probs):
            raise ValueError("waypoints and probs should have the same length.")
        if len(waypoints) <= 1:
            raise ValueError("waypoints should have at least 2 elements: the starting point and at least one waypoint.")

        if len(waypoints[0]) == 3:
            # rotation is considered
            ignore_rot = False
        elif len(waypoints[0]) == 2:
            # rotation is ignored
            ignore_rot = True
        else:
            raise ValueError("invalid waypoint format.")
        
        dist_matrix = build_distance_matrix(waypoints, probs, ignore_rot=ignore_rot)
        self.sa.env.dist_matrix = dist_matrix
        self.sa.reset()

        f_best, path = self.sa.run()
        if formation:
            # return in format: {'sorted_pts':[[(direction), x, y], ...], 'sorted_values':[probs, ...]}
            sorted_pts = [waypoints[i] for i in path]
            sorted_values = [probs[i] for i in path]
            formatted = {'sorted_pts':sorted_pts, 'sorted_values':sorted_values}
            return formatted
        else:
            # ruturn in format: (best_f, best_path in index)
            return f_best, path
        


def main1():
    data = {
        # 'waypoints': [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 2, 1]], 
        'waypoints': [[0, 0], [1, 0]], 
        'information_gain': [0.1, 0.2],
         }
    sa1 = SA(nparticles=200, T_init=7.329, T_final=5.49, alpha=0.985, gamma=0.99, verbose=False)

    '''
    waypoints: [[x, y], ...] or [[direction, x, y], ...]
      ** first element is the starting point **
    information_gain: [float, ...]
    '''

    sttime = time.time()
    sorted = sa1.infer(data['waypoints'], data['information_gain'])
    edtime = time.time()
    print(sorted)
    print('time elapsed:',edtime-sttime)

if __name__ == '__main__':
    main1()

