#!/usr/local/bin/python3
#
# find_luddy.py : a simple maze solver
#
# Submitted by : Neha Supe
#
# Based on skeleton code by Z. Kachwala, 2019
#

import sys
import json
import math

# Parse the map from a given filename
def parse_map(filename):
	with open(filename, "r") as f:
		return [[char for char in line] for line in f.read().split("\n")]

# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
	return 0 <= pos[0] < n  and 0 <= pos[1] < m

# The path list converted to String
def path_string(path):
	path_string = " "
	for i in range(len(path)-1):
		if path[i][0]==path[i+1][0]+1 and path[i][1]==path[i+1][1]:
			path_string +='N'
		elif path[i][0]==path[i+1][0]-1 and path[i][1]==path[i+1][1]:
			path_string +='S'
		elif path[i][0]==path[i+1][0] and path[i][1]==path[i+1][1]-1:
			path_string +='E'
		elif path[i][0]==path[i+1][0] and path[i][1]==path[i+1][1]+1:
			path_string += 'W'
	return path_string

# Find the possible moves from position (row, col)
# Return only moves that are within the board and legal (i.e. on the sidewalk ".")
def moves(map, row, col):
	moves=((row+1,col), (row-1,col), (row,col-1), (row,col+1))
	end_loc = [(row_i,col_i) for col_i in range(len(map[0])) for row_i in range(len(map)) if map[row_i][col_i]=="@"]
	n=0
	valid_moves = [move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" )]

	
# Hueristic to arrange the valid nodes in decreasing value of their D, D is absolute distance between the node and the end location
	if len(valid_moves)>1:
		
		dx = [abs(valid_moves[i][0]-end_loc[0][0]) for i in range(len(valid_moves))]
		dy = [abs(valid_moves[i][1]-end_loc[0][1]) for i in range(len(valid_moves))]
		D = [dx[i] + dy[i] for i in range(len(dx))]
		for i in range(len(D)):
				for j in range(i+1,len(D)):
					if D[i] > D[j]:
						a =  D[i]
						b = valid_moves[i]
						D[i] = D[j]
						valid_moves[i]=valid_moves[j]
						D[j] = a
						valid_moves[j]==b
		valid_moves = valid_moves[::-1]

	return valid_moves

	
# Perform search on the map
def search1(IUB_map):
	you_loc=[(row_i,col_i) for col_i in range(len(IUB_map[0])) for row_i in range(len(IUB_map)) if IUB_map[row_i][col_i]=="#"][0]
	fringe=[(you_loc,0)]
	visited = []
	path = []
	while fringe:
		if fringe == []:
			return False, False
		(curr_move, curr_dist)=fringe.pop()
		if curr_move in path:
					n = len(path)
					for j in range(n-1,-1,-1):
						if curr_move==path[j]:
							path= path[0:j]
						else:
							break
					
		visited.append((curr_move, curr_dist))
#checks if the current node is already visited at a lesser cost distance
		for i in range(len(visited)-1,0,-1):
			if visited[i][0] == (curr_move) and visited[i][1]<curr_dist:

				if fringe == []:
					return False, False
				(curr_move, curr_dist)=fringe.pop()
				visited.append((curr_move, curr_dist))
				
		path.append(curr_move)
		end_loc = [(row_i,col_i) for col_i in range(len(IUB_map[0])) for row_i in range(len(IUB_map)) if IUB_map[row_i][col_i]=="@"]
		for mov in moves(IUB_map, *curr_move):
			if IUB_map[mov[0]][mov[1]]=="@":
				
				path.append(mov)
				final_path = path_string(path)
				return curr_dist+1, final_path
			else:
				fringe.append((mov, curr_dist + 1))


# Main Function
if __name__ == "__main__":
	IUB_map=parse_map(sys.argv[1])
	print("Shhhh... quiet while I navigate!")
	solution, path= search1(IUB_map)
	print("Here's the solution I found:")
	if solution== False:
		print("Inf")
	else:
		print(solution, path)
