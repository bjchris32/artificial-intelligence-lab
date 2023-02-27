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

    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
