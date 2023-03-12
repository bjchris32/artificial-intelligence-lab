# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
from queue import PriorityQueue

class Mst:
    def __init__(self, vertices):
        print("init")
        self.number_of_vertices = len(vertices)
        self.graph = []
        self.vertice_positions = []

        print("vertices = ", vertices)
        for vertex in vertices:
            self.vertice_positions.append(vertex)

        # print("vertice_positions = ", self.vertice_positions)

        # add edges
        for idx, vertex in enumerate(self.vertice_positions):
            starting_idx = idx
            # print("start idx = ", idx)
            for j in range(starting_idx, len(vertices)):
                # print(" connect to j  = ", j)

                if starting_idx != j:# and j + 1 <= len(vertices) - 1:
                    weight = self.get_manhattan_distance(idx, j)
                    self.addEdge(idx, j, weight)
        # print("len(vertices) == ", len(vertices))
        # print("self.graph edges = ", len(self.graph))

    def get_manhattan_distance(self, vertex1_idx, vertex2_idx):
        # print("get_manhattan_distance")
        # print("self.vertice_positions = ", self.vertice_positions)
        # print("vertex1_idx = ", vertex1_idx)
        # print("vertex2_idx = ", vertex2_idx)
        vertex1 = self.vertice_positions[vertex1_idx]
        vertex2 = self.vertice_positions[vertex2_idx]
        distance = (abs(vertex1[0] - vertex2[0]) + abs(vertex1[1] - vertex2[1]))
        # print("distance of vertex1 =", vertex1 , " and vertex2 =", vertex2 ," distance = ", distance)
        return distance

    def addEdge(self, u, v, weight):
        self.graph.append([u, v, weight])

    def find_parent(self, parent_map, i):
        # print("find_parent: parent_map[i] = ", parent_map[i],",i = ", i)
        if parent_map[i] != i:
            parent_map[i] = self.find_parent(parent_map, parent_map[i])
        return parent_map[i]
    
    # union by rank
    def make_union(self, parent_map, rank, subset1, subset2):
        # print("make_union for subset1 ", str(subset1), " and subset2 ", str(subset2))
        # parent_map[subset1] = subset2
        if rank[subset1] < rank[subset2]:
            parent_map[subset1] =subset2
        elif rank[subset1] > rank[subset2]:
            parent_map[subset2] =subset1
        else:
            parent_map[subset2] =subset1
            rank[subset1] += 1

    # def is_cycle(self):
    #     parent_map = [0] * self.number_of_vertices
    #     for i in range(self.number_of_vertices):
    #         parent_map[i] = i

    #     for i in self.graph:
    #         for j in self.graph[i]:
    #             subset1 = self.find_parent(parent_map, i)
    #             subset2 = self.find_parent(parent_map, j)
    #             if subset1 == subset2:
    #                 return True
    #             self.make_union(parent_map, subset1, subset2)

    def kruskal_mst(self):
        result = []
        # used for sorted edges
        i = 0
        # used for result
        e = 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent_map = []
        rank = []

        for node in range(self.number_of_vertices):
            parent_map.append(node)
            rank.append(0)
        # print("parent_map = ", parent_map)

        # take the edges less than number_of_vertices - 1
        while e < (self.number_of_vertices - 1):
            # print("self.graph[i] == ", self.graph[i])
            u, v, weight = self.graph[i]
            # print("u = ", u)
            # print("v = ", v)
            # print("weight = ", weight)

            # break
            i = i + 1
            subset1 = self.find_parent(parent_map, u)
            subset2 = self.find_parent(parent_map, v)
            # print("subset1 = ", subset1)
            # print("subset2 = ", subset2)

            # build the edge if they are not the same
            if subset1 != subset2:
                e = e + 1
                result.append([u, v, weight])
                self.make_union(parent_map, rank, subset1, subset2)
        return result


def calculate_heuristic_cost(maze, current_position):
    # assume there is only one waypoint
    waypoint = maze.waypoints[0]
    return (abs(waypoint[0] - current_position[0]) + abs(waypoint[1] - current_position[1]))

def calculate_current_cost(path):
    return len(path)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    explored_set = set()
    s = maze.start
    waypoints = maze.waypoints
    queue = []
    queue.append([s])

    while queue:
        current_path = queue.pop(0)
        # print("current_path = ", current_path)
        last_node = current_path[-1]

        # iterate all neighbors of the current s
        # if the neighbor has not been explored, mark it as explored and enqueue it
        for neighbor in maze.neighbors(*last_node):
            # print("neighbor = ", neighbor)
            if neighbor in waypoints:
                new_path = list(current_path)
                new_path.append(neighbor)
                # print("neighbor is in waypoints, return the current path with neighbor= ", new_path)
                return new_path
            # check the neighbor is not explored and not a wall
            if neighbor not in explored_set and maze.navigable(*neighbor):
                explored_set.add(neighbor)
                new_path = list(current_path)
                new_path.append(neighbor)
                queue.append(new_path)

    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    priority_queue = PriorityQueue()
    explored_set = set()
    s = maze.start
    waypoints = maze.waypoints
    priority_queue.put((0, [s]))

    while not priority_queue.empty():
        current_priority_path_pair = priority_queue.get()
        current_cost = current_priority_path_pair[0]
        # print("current_priority_path_pair == ", current_priority_path_pair)
        current_path = current_priority_path_pair[1]
        # print("current_priority_path_pair = ", current_priority_path_pair)
        last_node = current_path[-1]
        explored_set.add(last_node)

        if last_node in waypoints:
            new_path = list(current_path)
            return new_path

        # iterate all neighbors of the current s
        for neighbor in maze.neighbors(*last_node):
            if neighbor in explored_set:
                continue
            if not maze.navigable(*neighbor):
                continue
            heuristic_cost = calculate_heuristic_cost(maze, neighbor)
            current_cost = calculate_current_cost(current_path)
            total_cost = current_cost + heuristic_cost
            new_path = list(current_path)
            new_path.append(neighbor)

            priority_queue.put((total_cost, new_path))

    return []


def get_nearest_waypoint(maze, position):
    waypoints = maze.waypoints
    nearest_waypoint = waypoints[0]
    nearest_distance = (abs(nearest_waypoint[0] - position[0]) + abs(nearest_waypoint[1] - position[1]))
    # iterate all waypoints to get nearest waypoint heuristically
    for idx, waypoint in enumerate(waypoints):
        if idx == 0:
            continue
        possible_distance = (abs(waypoint[0] - position[0]) + abs(waypoint[1] - position[1]))
        if nearest_distance > possible_distance:
            nearest_waypoint = waypoint
            nearest_distance = possible_distance

    return nearest_waypoint

def get_vertice_positions(waypoints):
    vertice_positions = []
    for vertex in waypoints:
        vertice_positions.append(vertex)
    return vertice_positions

def dfs_mst(node, parent, graph, size, dp, weights):
    # only the node itself is in the subtree
    size[node] = 1 * weights[node][parent]
    # there is no path in the subtree
    dp[node] = 0
    for neighbor in graph[node]:
        if neighbor != parent:
            dfs_mst(neighbor, node, graph, size, dp, weights)
            size[node] += size[neighbor]
            dp[node] += (dp[neighbor] + size[neighbor])


# mst =  [[0, 1, 1], [2, 3, 1], [5, 7, 1], [6, 9, 1], [7, 8, 2], [0, 5, 3], [1, 4, 3], [2, 4, 3], [3, 6, 3]]
def path_sum(mst, N):
    # N = len(mst)
    graph = [0] * N
    weights = [0] * N
    for i in range(0, N):
        graph[i] = []
        weights[i] = [0] * N
    # print("graph = ", graph)
    # print("weights = ", weights)
    for pair in mst:
        vertex1 = pair[0]
        vertex2 = pair[1]
        weight = pair[2]
        graph[vertex1].append(vertex2)
        graph[vertex2].append(vertex1)
        weights[vertex1][vertex2] = weight
        weights[vertex2][vertex1] = weight
    # print("weights = ", weights)
    dp = [0] * N
    result = [0] * N
    size = [0] * N

    for r in range(0, N):
        dfs_mst(r, -1, graph, size, dp, weights)
        result[r] = dp[r]
    return result

# Your heuristic function h should be the sum of the distance from (x,y) to the nearest waypoint,
# plus the MST length for the waypoints in S.
def calculate_multiple_waypoints_heuristic_cost(maze, current_position, nearest_waypoint, mst, vertice_positions, mst_length_to_other_waypoints):
    # sum of the distance from (x,y) to the nearest waypoint
    # Q: could we pick the waypoint randomly at the beginning?
    # No -> pick the nearest
    # print("current_position = ", current_position, " go to nearest waypoint = ", nearest_waypoint)
    distance_to_nearest_waypoints = (abs(nearest_waypoint[0] - current_position[0]) + abs(nearest_waypoint[1] - current_position[1]))

    # TODO: MST length from nearest_waypoints to other waypoints
    # How to compute from the nearest waypoints to other waypoints
    nearest_waypoint_idx = vertice_positions.index(nearest_waypoint)
    distance_to_other_waypoints = mst_length_to_other_waypoints[nearest_waypoint_idx]
    # print("distance_to_other_waypoints = ", distance_to_other_waypoints)
    # print("mst = ", mst)
    # mst =  [[0, 1, 1], [2, 3, 1], [5, 7, 1], [6, 9, 1], [7, 8, 2], [0, 5, 3], [1, 4, 3], [2, 4, 3], [3, 6, 3]]
    # index_of_going_to_waypoint = vertice_positions[]
    
    return (distance_to_nearest_waypoints + distance_to_other_waypoints)

# TODO: build mst
def construct_mst(waypoints):
    print("construct mst")
    mst_graph = Mst(waypoints)
    kruskal_mst = mst_graph.kruskal_mst()

    # TODO: convert back to position map to distance?
    print("kruskal_mst = ", kruskal_mst)

    return kruskal_mst

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # use MST
    # obtain an estimate of the cost of reaching the rest of the unreached waypoints once we have reached one.

    # by constructing a graph where the vertices are the waypoints 
    # and each edge connecting w_i to w_j has weight manhattan_distance(w_i, w_j) for all pairs of vertices (w_i, w_j),
    # the MST represents the approximate lowest cost path that connects all the waypoints.
    # Since it strictly underestimates the cost of going through all the waypoints,
    # this is an admissable heuristic

    priority_queue = PriorityQueue()
    explored_set = set()
    s = maze.start
    waypoints = maze.waypoints
    total_waypoints = len(waypoints)
    # going_to_waypoint = get_nearest_waypoint(maze, s)
    nearest_waypoint = get_nearest_waypoint(maze, s)
    priority_queue.put((0, [s], nearest_waypoint))

    print("start from = ", s)

    mst = construct_mst(waypoints)

    # mst =  [[0, 1, 1], [2, 3, 1], [5, 7, 1], [6, 9, 1], [7, 8, 2], [0, 5, 3], [1, 4, 3], [2, 4, 3], [3, 6, 3]]
    # mst_graph = [0] * len(mst)

    mst_length_to_other_waypoints = path_sum(mst, len(waypoints))
    print("mst_length_to_other_waypoints = ", mst_length_to_other_waypoints)

    waypoint_counter = 0
    while not priority_queue.empty():
        current_priority_tuple = priority_queue.get()
        current_cost = current_priority_tuple[0]
        # print("current_priority_tuple == ", current_priority_tuple)
        current_path = current_priority_tuple[1]
        nearest_waypoint = current_priority_tuple[2]
        # print("current_priority_tuple = ", current_priority_tuple)
        last_node = current_path[-1]
        explored_set.add((last_node, nearest_waypoint))

        # # Q: how to check if it is nearest? Is it solved by priority queue?
        if last_node in waypoints:
            print("last_node = ", last_node)
            waypoint_counter += 1
            if waypoint_counter == 3:
                new_path = list(current_path)
                print("new_path with 3 waypoints = ", new_path)
                return new_path

            if waypoint_counter == total_waypoints:
                new_path = list(current_path)
                return new_path

        # iterate all neighbors of the current s
        for neighbor in maze.neighbors(*last_node):
            nearest_waypoint = get_nearest_waypoint(maze, neighbor)
            # print("explored_set = ", explored_set)
            # print("neighbor = ", neighbor, " , nearest_waypoint = ", nearest_waypoint)
            if (neighbor, nearest_waypoint) in explored_set:
                continue
            if not maze.navigable(*neighbor):
                continue
            # Sol1: iterate all waypoints and push to priority queue
            # -> it will traverse other waypoint and put into priority queue heuristically
            # for possible_nearest_waypoint in waypoints:
            #     vertice_positions = get_vertice_positions(waypoints)
            #     heuristic_cost = calculate_multiple_waypoints_heuristic_cost(maze, neighbor, nearest_waypoint, mst, vertice_positions, mst_length_to_other_waypoints)
            #     current_cost = calculate_current_cost(current_path)
            #     total_cost = current_cost + heuristic_cost
            #     new_path = list(current_path)
            #     new_path.append(neighbor)
            #     priority_queue.put((total_cost, new_path, nearest_waypoint))

            # Sol2: only take the nearest waypoint and push to priority queue
            # calculate which is the heuristic nearest waypoint?
            vertice_positions = get_vertice_positions(waypoints)
            heuristic_cost = calculate_multiple_waypoints_heuristic_cost(maze, neighbor, nearest_waypoint, mst, vertice_positions, mst_length_to_other_waypoints)
            current_cost = calculate_current_cost(current_path)
            total_cost = current_cost + heuristic_cost
            new_path = list(current_path)
            new_path.append(neighbor)
            priority_queue.put((total_cost, new_path, nearest_waypoint))

    return []

# Q: how to set the nearest waypoint?
# Find the first waypoint and check if it is smaller than past waypoints?