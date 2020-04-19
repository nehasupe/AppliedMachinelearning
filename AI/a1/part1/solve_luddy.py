#!/usr/local/bin/python3
# solve_luddy.py : Sliding tile puzzle solver
#
# Code by: Kelly Wheeler (kellwhee) and Neha Supe (nehasupe)
#
# Based on skeleton code by D. Crandall, September 2019
#
from queue import PriorityQueue
import sys

MOVES = { "R": (0, -1), "L": (0, 1), "D": (-1, 0), "U": (1,0) }
CIRCULAR_MOVES = { "R": (0, 3), "L": (0, -3), "D": (3, 0), "U": (-3, 0) }
LUDDY_MOVES = { "A": (2, 1), "B": (2, -1), "C": (-2, 1), "D": (-2, -1), "E": (1, 2), "F": (1, -2), "G": (-1, 2), "H": (-1, -2) }

#given the row and column number of an element, it returns the elements index in a list
def rowcol2ind(row, col):
    return row*4 + col

#given an index of an element in a list, it calculates its row and column number
def ind2rowcol(ind):
    return (int(ind/4), ind % 4)

#given a point, returns true if it fits in a 3x3 board, else false
def valid_index(row, col):
    return 0 <= row <= 3 and 0 <= col <= 3

#returns a new list with the variables at both index's switched
def swap_ind(list, ind1, ind2):
    return list[0:ind1] + (list[ind2],) + list[ind1+1:ind2] + (list[ind1],) + list[ind2+1:]

#returns a new list with the variables at both points switched
def swap_tiles(state, row1, col1, row2, col2):
    return swap_ind(state, *(sorted((rowcol2ind(row1,col1), rowcol2ind(row2,col2)))))

def printable_board(row):
    return [ '%3d %3d %3d %3d'  % (row[j:(j+4)]) for j in range(0, 16, 4) ]

# return a list of possible successor states
def successors(state):
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    return [ (swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c) \
             for (c, (i, j)) in MOVES.items() if valid_index(empty_row+i, empty_col+j) ]

# return a list of possible circular successor states
def circular_successors(state):
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    return [ (swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c) \
             for (c, (i, j)) in CIRCULAR_MOVES.items() if valid_index(empty_row+i, empty_col+j) ]

# return a list of possible luddy successor states
def luddy_successors(state):
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    return [ (swap_tiles(state, empty_row, empty_col, empty_row+i, empty_col+j), c) \
             for (c, (i, j)) in LUDDY_MOVES.items() if valid_index(empty_row+i, empty_col+j) ]

def all_successors(state, variant):
    succ = successors(state)
    if variant == "circular":
        #get circular successors
        succ = succ + circular_successors(state)
    elif variant == "luddy":
        #replace succ with luddy successors
        succ = luddy_successors(state)
    return succ

#calculates the permutation inversions of a board
def permutation_inversions(state):
    ind = 0
    count = 0
    for elem in state:
        for i in range(ind+1, len(state)):
            if state[i] > 0 and state[i] < elem:
                count += 1
        ind += 1
    return count

#checks if the initial board is solvable
def parity(state):
    (empty_row, empty_col) = ind2rowcol(state.index(0))
    inversions = permutation_inversions(state)
    if empty_row == 3 or empty_row == 1:
        if inversions % 2 == 0:
            return "solvable"
    elif inversions % 2 == 1:
        return "solvable"
    return "Inf"

# check if we've reached the goal
def is_goal(state):
    return sorted(state[:-1]) == list(state[:-1]) and state[-1]==0

#returns the Manhattan distance of the element to its goal state
def manhattan_distance(state, ind, variant):
    (row, col) = ind2rowcol(ind)
    goalInd = state[ind] - 1
    if state[ind] == 0:
        goalInd = 15#the 0 should be the last element in the list
    (goalRow, goalCol) = ind2rowcol(goalInd)
    dist = abs(row - goalRow) + abs(col - goalCol)
    if variant == "luddy":
        dist = dist / 3
    return dist

def manhattan_heuristic(state, variant):
    h = 0
    for row in range(4):
        for col in range(4):
            mDist = manhattan_distance(state, rowcol2ind(row, col), variant)
            h += mDist
    return h

def misplaced_tiles(state, variant):
    h = 0
    ind = 0
    for tile in state:
        if ind == 15:
            if tile != 0:
                h = h + 1
        elif tile != state[ind]:
            h = h + 1
        ind = ind + 1
    return h

# The solver! - using BFS right now
def solve(initial_board, variant):
    if parity(initial_board) == "Inf":
        return "Inf"
    fringe = PriorityQueue()
    fringe.put((0, (initial_board, "")))
    while fringe.qsize() > 0:
        (cost, (state, route_so_far)) = fringe.get()
        for (succ, move) in all_successors(state, variant):
            if is_goal(succ):
                return( route_so_far + move )
            heu = 0
            if variant == "luddy":
                heu = misplaced_tiles(succ, variant)
            else:
                heu = manhattan_heuristic(succ, variant)
            fringe.put((cost + heu, (succ, route_so_far + move ) ))
    return False

# test cases
if __name__ == "__main__":
    if(len(sys.argv) != 3):
        raise(Exception("Error: expected 2 arguments"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if((sys.argv[2] != "original") and (sys.argv[2] != "circular") and (sys.argv[2] != "luddy")):
        raise(Exception("Error: that is not one of the available variants"))

    if len(start_state) != 16:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state), sys.argv[2])
    
    if route == "Inf":
        print("Inf")
    else:
        print("Solution found in " + str(len(route)) + " moves:" + "\n" + route)

