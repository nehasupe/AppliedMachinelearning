#!/usr/local/bin/python3
#
# choose_team.py : Choose a team of maximum skill under a fixed budget
#
# Code by: Kelly Wheeler (kellwhee) and Neha Supe (nehasupe)
#
# Based on skeleton code by D. Crandall, September 2019
#
import sys
from queue import PriorityQueue

# function loads robots, skills and cost into a dictionary from file
def load_people(filename):
    people={}
    with open(filename, "r") as file:
        for line in file:
            l = line.split()
            people[l[0]] = [ float(i) for i in l[1:] ] 
    return people


# calculates the weight of the explored robots
def new_weight(people, selectedItems):
    weights = 0
    for i in range(len(selectedItems)):
        (skill, w) = people[i][1]
        weights = weights + w* int(selectedItems[i])
    return weights

# calculates the upper bound value
def new_skills(people, budget, selectedItems):
    i = 0
    solution = 0

    for (person, (skill, cost)) in people:
        if i < len(selectedItems):
            if budget - cost > 0:        
                budget -= cost * int(selectedItems[i])
                solution = solution + skill * int(selectedItems[i])
        else:
            if budget - cost > 0:
                solution = solution + skill
                budget -= cost
        i = i + 1

    return -1 * solution

# creates two nodes which are to be added to the fringe
def add_node(people, budget, selectedItems, a):
    if a == 1:
        selectedItems = selectedItems+'1'
        return cal_node_values(people, budget, selectedItems)
    elif a == 0:
        selectedItems = selectedItems+'0'        
        return cal_node_values(people, budget, selectedItems)


def cal_node_values(people, budget, selectedItems):
    new_weight_so_far = new_weight(people, selectedItems)
    if new_weight_so_far <= budget:
        new_skills_so_far = new_skills(people, budget, selectedItems)
        return (new_skills_so_far,( new_weight_so_far, selectedItems))
    else:    
        return False

#
def approx_solve(people, budget):

    people = sorted(people.items(), key=lambda x: x[1][1]/x[1][0])
    count = 0
    for (person, (skill, cost)) in people:
        if cost < budget:
            count = count +1
    if count == 0:
        return 'Inf'
    solution = []
    fringe = PriorityQueue()
    selectedItems = ''
    fringe.put((0, (0, selectedItems)))
    while len(selectedItems) <= len(people):
        (skills_so_far, (weight_so_far, selectedItems)) = fringe.get()
        if len(selectedItems) == len(people):
            i = 0
            for (person, (skill, cost)) in people:
                if int(selectedItems[i]):
                    solution.append((person, 1.000000))
                i = i + 1

            return (skills_so_far, ( weight_so_far, solution))
            
        node = add_node(people, budget, selectedItems, 1)
        if node!=False:            
            fringe.put(node)
        node = add_node(people, budget, selectedItems, 0)
        if node!=False:            
            fringe.put(node)
           
    return possible_solution.get()

if __name__ == "__main__":

    if(len(sys.argv) != 3):
        raise Exception('Error: expected 2 command line arguments')

    budget = float(sys.argv[2])
    people = load_people(sys.argv[1])
    solution = approx_solve(people, budget)
    if solution != 'Inf':
        skills_so_far, ( weight_so_far, s) = solution
        print("Found a group with",len(s),"people costing",weight_so_far,"with total skill", -1 * skills_so_far)
        for (person, cost) in s:
           print(person, "%.6f"%cost)
    else:
        print('Inf')
    
