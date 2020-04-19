B551 Assignment 0: Searching and Python 

Fall 2019 

  

Part 1: Finding your way 

  

According to the problem statement, the # is where we are and is our start position for the path, @ is the luddy hall which is our goal position, & are obstacles and . are the sidewalks. The given code loops infinitely to find a path from the start position # to goal position @. It finds the possible moves we can take to reach our goal by checking the esoteric symbols around the current position in the directions- up/North, down/South, left/West, Right/East. If a . Or @ is one the neighboring symbols then it is a possible move. It adds all the possible moves to the fringe. However, the code does not take into consideration that we might meet a dead end by being surrounded by & and keeps looping between those positions infinitely.  

To find the shortest possible path from # to @, first we need to make a change to stop the code running infinitely by keeping a track of the visited nodes. The initial code made use of a list as stack. The recently added node was popped from the stack and its successors were added to the stack. The fringe list stored the node as (row_number, column_number) and also stored the distance from the initial position.  

I continued with the same stack structure of fringe containing the node positions and total distance. I used visited list to keep a track of the nodes I am traversing along with the distance. No nodes are popped from this list and are useful for backtracking. I used another list to keep a track of the path. The path list contains just the positions of the nodes. First, added our initial position to the fringe. Then popped a node from the fringe and checked if the node is already present in our path list. If the node is present in our path list, the path list is sliced to this node.  

The current node postion is the appended to the visited list along with its distance. Then we check if the current node was already traversed through for a lesser cost distance. We iterate through the visited list and “if the current node position matches the visited node and it was visited at a lesser cost distance then we are going to pop another node from the fringe. The loop will iterate for all the visited nodes. After this loop, the current  node we have is appended to the path list. And this node is passed to a function to generate successor nodes. In this function, the possible moves for the current node are generated. If the number of valid moves is one then that move is returned back by the function. If there are more than 1 valid moves then for that I have considered a heuristic function which calculates the absolute distance from that valid move to the end location. The function calculates  

	D = dx + dy where dx = absolute(valid_move_row – end_location_row )  and 

										dy = absolute(valid_move_column – end_location_column ) 

The valid moves are arranged according to the value of D in decreasing order. And are returned back by the function. These valid moves are added to the fringe. So when the next node is popped from the fringe, the node is closer to the goal node than the other successors that were added for the same node.  

Initial state: The initial state for this problem is a map which is a board of N lines and M columns, total size N x M units. The map cells are marked with esoteric symbols- #, . , &, @, representing our position, sidewalk, obstacle, and luddy hall (goal position) respectively. 

Set of valid states: We can travel on the . sidewalks by moving in N, S, W, E directions. Valid state would be # on any of the . sidewalk  and @ and not on &. 

Successor function: The heuristic function is f(n) + g(n). Since we were expanding for the successors of the same current node, calculated g(n) = dx + dy where dx and dy are absolute coordinate distances from the successor node to end location. The node with the heuristic function value smaller value will be expanded first.  

Cost function: It is the number of cells or the distance units traversed by # to reach @. 

goal state definition: When we have reached @ by travelling the shortest path. 


Part 2: Hide-and-seek 

This problem is to place n number of F friends on the campus map. The initial code generates successor combinations by putting F in place of ‘. ’. The code iterates starting by generating boards putting 1 F in all ‘. ’, then another F in all ‘.’ till it generates the first board with n number of Fs on it. The first board with n number of F will be printed. The initial code does not check if the ‘.’ position is a valid position or not. It does not check if there is another F friend already present in the same row or column who can see this newly put F.  

State space: The state space is the collection of all possible board combinations with F placed in the valid position which satisfy all the conditions of not being seen by another friend. 

Goal state: The goal state is a state with any combination of n number of friends places on the board, who are not in the same row and column and if they are then they are hidden with the obstacle ‘&’ or ‘@’. (i.e they cannot see each other) 

Successor function: The successor function is to generate all possible combinations for 1 to n number of friends such that all the Fs in these combinations are at valid possible positions which satisfy the condition that any F on the board is not visible to another F on the board  

Cost function: The cost function is the number of combinations generated till all n friends are placed on board such that they satisfy the condition of not being seen by any other F on the board. 
