CS B551 - Assignment 1: Searching
Fall 2019

Code and report by: Kelly Wheeler (kellwhee) and Neha Supe (nehasupe)
________________________________________________________________________________________________________________________________________
Part 1: The Luddy puzzle
1) State space: Any arrangement of 15 numbered tiles and an empty tile on a 4x4 board

Successor function: Given by available actions (sliding tiles):
	Original and Circular: L, R, U, D
	Luddy: A, B, C, D, E, F, G, H

Edge weights: The number of steps the selected tile is from its goal location (Manhattan distance).

Goal state: In the list version of the board values are in order from 1 to 15 then 0 representing the empty tile after the 15.

Heuristic function: Our heuristic was us calculating the sum of Manhattan distances of every tile in the board to their goal position.  The formula for Manhattan distance when you have two points (x, y) and (a, b) is: |x-a| + |y – b|.
Argue heuristic is admissible: For a heuristic to be admissible it needs to never overestimate.  The Manhattan distance finds the distance from one point to another so it will return the minimum number of spaces (steps) to reach its goal location, because of that it will never overestimate but either be accurate or underestimate.

2) We used the A* Search algorithm to solve the puzzle.  We find the successors of the state we are in and if it is not the goal, we insert it into a priority queue.  If the fringe is empty, we return that there is no solution but if not we pop from the priority queue which gives us the item with the lowest cost.  Our cost was calculated by f(s) = g(s) + h(s).  g(s) is the cost of the path so far and h(s) is the admissible heuristic that we listed above.

3) After writing code for this problem, we tested the code against the given sample boards. For boardn, the code went into an infinite loop. We assumed it was taking long because it had to perform n moves but when we really tried to solve it on paper that we realized that the board is unsolvable. Hence, for that, we implemented a function to check parity of the board using permutation inversion. We referred: https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/ for implementing the concept of parity in our code.   



Part 2: Road trip!

1) State space: A list of locations that have direct paths to one another in order.

Successor function: Given by finding a location that has a direct path from the last item in the list.

Edge weights: the cost to get from one location to an adjacent one (1 segment, x miles etc.)

Goal state: A list of locations with the start state in the beginning and is followed by locations with direct paths, the goal location being the last item.

Heuristic function: We had different heuristics for each of the given cost functions:

	Segments: Always 1
	
	Distance: Length in miles provided between the two points
	
	Time: The time required to get from one point to the other assuming the user is consistently going the speed limit.  This is calculated by dividing the given distance in miles by the given speed limit.
	
	MPG: How many miles per gallon you get which is based on the given speed limit.  This is calculated by the provided formula where v is the speed limit.  MPG(v) = 400*(v/150)*(1-(v/150)^4
	
Argue heuristic is admissible:

	Segments: Because it is always 1, h(n) and h*(n) will always be 1, so it satisfies the requirement 0<=1<=1.
	
	Distance: Not admissible, it is dependent on the available paths.
	
	Time: Not admissible, it is dependent on the available paths.
	
	MPG: Not admissible, it is dependent on the available paths.
	
  
2) We used the A* Search algorithm to find the best path.  We find the cities with direct paths from the given city and if it is not the end city, we insert it into a priority queue.  If the fringe is empty, we return that there is no solution but if not, we pop from the priority queue which gives us the item with the lowest cost.  Our cost was calculated by f(s) = g(s) + h(s).  g(s) is the cost of the path so far and h(s) is the admissible heuristic that we listed above.

3) There were two main issues that came up when working on this.  The first was that we thought there was the need for an advanced heuristic, so we implemented Euclidean distance and kept trying to use it.  We never got the right output when using this, so we eventually decided to just implement it later, but as we went we discovered that there was no need at all so we finally deleted it all.  The other issue was with MPG and returning the total gas gallons at the end.  We were not getting the correct values and thought that our ‘v’ in the formula was incorrect.  We looked into all of these complicated formulas to try and get it to work but it would not.  After going to office hours we put ‘v’ back to our first interpretation which was correct and changed where we calculated the gas gallons.  Instead of keeping track of MPG and calculating gas gallons at the end, we just keep track of gas gallons and cut out the extra step.


Part 3: Choosing a team 
 
 
1) and 2) The task for this problem was to return a combination of a set of robots such that they fit in our budget while maximizing their total skills. By using the greedy approach, the code provided a solution which considered fractions of a robot for the final solution. However, if we considered whole robots for solution using the greedy approach, it did not result in giving us an optimal solution. So, our search problem here is to select whole robots which will give us maximum skills. 
We are using the upper bound as the heuristic for the implementation of solution to this problem. 
In our solution, we first order the robots in the decreasing order of their skills by cost ratio. We use the upper bound values for the combination of robots to decide which combination to explore next and which to prune. The upper bound values are calculated by adding the skills of selected explored robots and adding unexplored robots until we don’t overrun the budget.  
We consider if the first item should be included in the combination of robots. We create two nodes, one node will represent the upper bound when then this item is included in the combination and the other will represent the upper bound which is calculated without including this item. These nodes are added to the fringe.  
 Using the priority queue, we get the node with largest upper bound (we negated the upper bound values to use them in a priority queue, the priority queue will give the node with the smallest negative value which is actually the largest upper bound value in the fringe). This node is then explored to add the next robot.  
 
State space:  
The state space is the string of combinations of robots along with their upper bounds. 

Successor function: 
From the current node, the next robot in the sorted robot-skills list is considered. Upper bounds selecting this robot and not selecting this robot are calculated. This creates two nodes in the fringe. The successor function selects the node with the maximum upper bound (smallest value in the priority queue since the upper bounds are negated before adding to the fringe)   

Edge weights:  
The edge weights represent if we are selecting the ith item or not to calculate the upper bound 
If the edge weight is select(I) = 0 then this item is not to be considered in the combination of robots and an upper bound needs to be calculated without including this robot. 
If the edge weight select(I) = 1 then this robot is to be considered in the combination of robots and an upper bound needs to be calculated including this robot. 

Goal state:  
The goal state is reached when the fringe pops out the node which has the length of the string of explored nodes equal to the number of robots.  

Heuristic function:  
The heuristic function we have used here calculates the upper bound value considering the explored nodes as well as unexplored nodes. The skills of the selected explored nodes are added and if we still have some budget left to add entire robots from the unexplored sorted robot-skills list, these unexplored robots are added. Our function is admissible because it doesn’t explore all the possible combinations of robots.  
 
3) We used a binary string to represent selected and unselected objects to simplify the upper bound calculations. We are trying to maximize the upper bound values. We negated the values of upper bound in our nodes so we can use them in a priority queue. We faced issues with our new_skills function. We were previously making a mistake with the position of the nested ‘if’ loops which resulted in iteration only over items that were in the seletedItems (explored robots) list. It was fixed by exchanging the inner ‘if’ loop with the outer ‘if’ loop. 
 

