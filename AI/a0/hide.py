#!/usr/local/bin/python3
#
# hide.py : a simple friend-hider
#
# Submitted by : nehasupe
#
# Based on skeleton code by D. Crandall and Z. Kachwala, 2019
#
# The problem to be solved is this:
# Given a campus map, find a placement of F friends so that no two can find one another.
#
import sys

#The logic for this code is based on what my classmate Manisha S Kumar hinted to me- 
#"if you find an F around the current position, the position is not valid and if there is an & then its a valid position"

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().split("\n")]

# returns number of friends on board when called, called from is_goal function
def count_friends(board):
    return sum([ row.count('F') for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ "".join(row) for row in board])

# Checks if (row,col) is one of the valid possible positions to place F by traversing the upper section of the position
def check_up(board, row, col):
    for r in range(row-1, -1,-1):
        if board[r][col] == 'F':
            return 0
        if board[r][col] =='&':
            return 1
    return 1

# Checks if (row,col) is one of the valid possible positions to place F by traversing the lower section of the position
def check_down(board, row, col):
    for r in range(row+1, len(board)):
        if board[r][col] == 'F':
            return 0
        if board[r][col] =='&':
            return 1
    return 1

# Checks if (row,col) is one of the valid possible positions to place F by traversing the left section of the position 
def check_left(board, row, col):
    for c in range(col-1, -1, -1):
        if board[row][c] == 'F':
            return 0
        if board[row][c] == '&':
            return 1
    return 1

# Checks if (row,col) is one of the valid possible positions to place F by traversing the right section of the position 
def check_right(board, row, col):
    for c in range(col+1,len(board[0])):
        if board[row][c] == 'F':
            return 0
        if board[row][c]=='&':
            return 1
    return 1

def valid_position(board, row, col):
    return True if (check_up(board, row, col) + check_down(board, row, col) + check_left(board, row, col) + check_right(board, row, col)) == 4 else False

# Add a friend to the board at the given position, and return a new board (doesn't change original)
def add_friend(board, row, col):

    return board[0:row] + [board[row][0:col] + ['F',] + board[row][col+1:]] + board[row+1:] 
    
# Get list of successors of given board state
def successors(board):
    return [ add_friend(board, r, c) for r in range(0, len(board)) for c in range(0,len(board[0])) if board[r][c] == '.' and valid_position(board, r, c) ]

# check if board is a goal state
def is_goal(board):#
    return count_friends(board) == K 

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    i=0
    while len(fringe) > 0:
        for s in successors(fringe.pop()):
            if is_goal(s):
                return(s)
            fringe.append(s)

    return False

# Main Function
if __name__ == "__main__":
    IUB_map=parse_map(sys.argv[1])

    # This is K, the number of friends
    K = int(sys.argv[2])
    luddy_loc=[[row_i,col_i] for col_i in range(len(IUB_map[0])) for row_i in range(len(IUB_map)) if IUB_map[row_i][col_i]=="@"]
    IUB_map[luddy_loc[0][0]][luddy_loc[0][1]] = "&"
    solution = solve(IUB_map)
    if solution:
        map = solution
        map[luddy_loc[0][0]][luddy_loc[0][1]] = "@"
        print(printable_board(map))
    else :
        print("None")
    
    

  
    
