#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from gurobipy import Model, GRB, quicksum
import numpy as np


# In[2]:


# parse file to create adjacency dictionary
def create_adjacency_dict(file_path):
    adjacency = {}
    with open(file_path, 'r') as file:
        current_county = ""
        for line in file:
            # split line by tab and remove empty parts
            parts = [part for part in line.strip().split("\t") if part]

            # process new county entry or adjacent county
            if not line.startswith("\t"):
                # handle new county entry
                if parts:
                    current_county = parts[0]
                    adjacency[current_county] = []
                    # handle adjacent county on same line
                    if len(parts) >= 3:
                        adjacent_county = parts[2]
                        adjacency[current_county].append(adjacent_county)
            elif line.startswith("\t\t") and current_county:
                # handle adjacent county on new line
                if len(parts) >= 1:
                    adjacent_county = parts[0]
                    adjacency[current_county].append(adjacent_county)

        # remove quotation marks from county names
        adjacency = {county.replace('"', ''): [neighbor.replace('"', '') for neighbor in neighbors]
                     for county, neighbors in adjacency.items()}
    return adjacency

# set file path and create adjacency dictionary
file_path = "Country_Adj.txt"
adjacency = create_adjacency_dict(file_path)

# filter non-colorado counties from adjacency lists
adjacency = {county: [neighbor for neighbor in neighbors if "CO" in neighbor]
             for county, neighbors in adjacency.items()}

# modify colorado counties for consistency
colorado_adjacent = {county.replace(", CO", ", Colorado"): [neighbor.replace(", CO", ", Colorado") for neighbor in neighbors]
                     for county, neighbors in adjacency.items() if "CO" in county}

# create array of colorado county names
colorado_counties = list(colorado_adjacent.keys())

# load population data
population_data = pd.read_csv("Colorado_Population.csv", header=None)
county_names = population_data.iloc[0].tolist()
county_populations = population_data.iloc[1].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).tolist()

# map each county to its population
colorado_population = dict(zip(county_names, county_populations))

# include only counties in colorado_counties
colorado_population = {county: colorado_population[county] for county in colorado_counties if county in colorado_population}

# use generated colorado data for analysis
counties = colorado_counties
populations = colorado_population
adjacency = colorado_adjacent

# define number of counties and create adjacency matrix
N = len(colorado_counties)
adjacency_matrix = np.zeros((N, N), dtype=int)
county_index = {county: idx for idx, county in enumerate(colorado_counties)}

# populate adjacency matrix
for county, neighbors in colorado_adjacent.items():
    for neighbor in neighbors:
        i, j = county_index[county], county_index[neighbor]
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # bidirectional adjacency

# transform adjacency matrix to distance matrix using bfs
def transform_to_distance_matrix(adjacency_matrix):
    num_counties = len(adjacency_matrix)
    distance_matrix = np.full((num_counties, num_counties), np.inf)

    # bfs to update distance matrix
    def bfs(start):
        visited = [False] * num_counties
        queue = [(start, 0)]  # use list as queue
        visited[start] = True

        while queue:
            current, dist = queue.pop(0)  # pop from front of the queue
            distance_matrix[start][current] = dist
            for neighbor, is_adjacent in enumerate(adjacency_matrix[current]):
                if is_adjacent and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))  # append to end of the queue

    # perform bfs for each county
    for i in range(num_counties):
        bfs(i)

    # replace inf with -1 or any large number
    distance_matrix[distance_matrix == np.inf] = -1
    return distance_matrix

distance_matrix = transform_to_distance_matrix(adjacency_matrix)
distance_matrix += 1

# district modeling with gurobi
m = Model("Districts")
district_vars = m.addVars(counties, range(1, 9), vtype=GRB.BINARY, name="inDistrict")
total_counties = len(colorado_counties)  # 64 counties
num_districts = 8
total_population = sum(populations.values())
average_population_per_district = total_population / num_districts

# ensure each county is in exactly one district
for county in counties:
    m.addConstr(quicksum(district_vars[county, d] for d in range(1, 9)) == 1, f"OneDistrict_{county}")

# distance calculations for districts
district_distance_vars = m.addVars(range(1, 9), vtype=GRB.CONTINUOUS, name="DistrictDistance")
weight_population = 1
weight_distance = 1
focal_points = {d: colorado_counties[d-1] for d in range(1, 9)}
focal_distance_vars = m.addVars(range(1, 9), vtype=GRB.CONTINUOUS, name="FocalDistance")

# calculate distances to focal points
for d in range(1, 9):
    focal_idx = county_index[focal_points[d]]
    m.addConstr(
        focal_distance_vars[d] ==
        quicksum(distance_matrix[focal_idx][county_index[county]] * district_vars[county, d]
                 for county in counties),
        f"FocalDistance_{d}"
    )

# adjusted adjacency constraints
for county in counties:
    for d in range(1, 9):
        county_idx = county_index[county]
        adjacent_in_district = quicksum(
            adjacency_matrix[county_idx][county_index[neighbor]] * district_vars[neighbor, d]
            for neighbor in counties if neighbor != county
        )

        # relaxed adjacency constraint with slack variable
        slack_var = m.addVar(vtype=GRB.BINARY, name=f"slack_{county}_{d}")
        m.addConstr(
            adjacent_in_district + slack_var >= 1,
            f"Adjacency_{county}_{d}"
        )

# population difference variables
pop_diff_vars = m.addVars(range(1, 9), vtype=GRB.CONTINUOUS, name="PopDiff")

# objective function for population and distance
m.setObjective(
    quicksum(pop_diff_vars[d] * weight_population for d in range(1, 9)) +
    quicksum(focal_distance_vars[d] * weight_distance for d in range(1, 9)),
    GRB.MINIMIZE
)

# constraints for population differences
for d in range(1, 9):
    district_population = quicksum(populations[county] * district_vars[county, d] for county in counties)
    m.addConstr(pop_diff_vars[d] >= district_population - average_population_per_district, f"PopDiffPos_{d}")
    m.addConstr(pop_diff_vars[d] >= -(district_population - average_population_per_district), f"PopDiffNeg_{d}")

# add reference to variables in the model
m._vars = district_vars

# optimize model
m.optimize()

# check model status and print best solution
if m.SolCount > 0:
    # process feasible solution
    district_data = []
    district_populations = {}  # population of each district
    capitals_used = {}

    for d in range(1, num_districts + 1):
        district_population = 0
        capital_found = False
        for county in counties:
            if district_vars[county, d].X > 0.5:  # get solution value
                district_data.append({'County': county, 'District': d})
                district_population += populations[county]
                if not capital_found:
                    capitals_used[d] = county
                    capital_found = True
        
        district_populations[d] = district_population
        print(f"District {d} Population: {district_population}")  # print district population

    district_assignments = pd.DataFrame(district_data)

    # set pandas display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # print district assignments
    for index, row in district_assignments.iterrows():
        print(f"County: {row['County']}, District: {row['District']}")

    # print capitals for each district
    for district, capital in capitals_used.items():
        print(f"District {district}: Capital: {capital}")
else:
    print("No feasible solution was found within the time limit.")


# In[ ]:





# In[ ]:




