#!/usr/local/bin/python3

# route.py : Best route finder

#Code by: Kelly Wheeler (kellwhee) and Neha Supe (nehasupe)

from queue import PriorityQueue
import sys
import math

#find_attached_cities gets a list of lists of city names with data that all connect to the given city
def find_attached_cities(city, road_segments):
    closeCities = []
    for info in road_segments:
        if info[0] == city or info[1] == city:
            closeCities.append(info)
    return closeCities

#finds all successors and calculates the cost and data
def all_successors(city, cost_function, road_segments, city_gps):
    succ = []
    closeCities = find_attached_cities(city, road_segments)
    for cityInfo in closeCities:
        currentCity = cityInfo[0]
        if currentCity == city:
            currentCity = cityInfo[1]
        dist = float(cityInfo[2])
        seg = 1
        time = float(cityInfo[2]) / float(cityInfo[3])
        vDiv = float(cityInfo[3]) / 150
        powPart = (1 - vDiv)**4
        mpg = 400 * vDiv * powPart
        gallons = dist / mpg
        ls = [seg, dist, time, gallons]
        if cost_function == "segments":
            succ.append((currentCity, (seg, ls)))
        elif cost_function == "distance":
            succ.append((currentCity, (dist, ls)))
        elif cost_function == "time":
            succ.append((currentCity, (time, ls)))
        else: #mpg
            succ.append((currentCity, (mpg, ls)))
    return succ

# solver
#based on skeleton code by D. Crandall in part 1
def solve(start_city, end_city, cost_function, road_segments, city_gps):
    fringe = PriorityQueue()
    fringe.put(((0, [0, 0, 0, 0]), (start_city, start_city)))
    while fringe.qsize() > 0:
        ((cost, curData), (city, route_so_far)) = fringe.get()
        for (succ, (move, data)) in all_successors(city, cost_function, road_segments, city_gps):
            newData = [curData[0] + data[0], curData[1] + data[1], curData[2] + data[2], curData[3] + data[3]]
            if succ == end_city:
                return ((route_so_far + " " + succ, newData))
            fringe.put(((cost + move, newData), (succ, route_so_far + " " + succ)))
    return False

#test cases
#Based off skeleton code given for part1
if __name__  == "__main__":
    if(len(sys.argv) != 4):
        raise(Exception("Error: expected 3 arguments"))

    road_segments = []
    with open("road-segments.txt", 'r') as file:
        for line in file:
            road_segments.append((line.split()))

    city_gps = []
    with open("city-gps.txt", 'r') as file:
        for line in file:
            ls = line.split()
            city = ls[0]
            latitude = ls[1]
            longitude = ls[2]
            city_gps.append((city, (float(latitude), float(longitude))))

    start_city = sys.argv[1]
    end_city = sys.argv[2]
    cost_function = sys.argv[3]

    if((cost_function != "segments") and (cost_function != "distance") and (cost_function != "time") and (cost_function != "mpg")):
        raise(Exception("Error: that is not one of the available costs"))

    print("Solving...")
    
    (route, data) = solve(start_city, end_city, cost_function, road_segments, city_gps)

    solution = ""
    ind = 0
    for value in data:
        if ind == 1:
            solution = solution + str(int(value)) + " "
        else:
            solution = solution +  str(value) + " "
        ind = ind + 1
    
    solution += route

    print(solution)
