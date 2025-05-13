import csv
from pulp import GLPK
from pulp import PULP_CBC_CMD
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
from datetime import datetime
import pprint
import itertools
import random


def solve_knapsack_problem(capacity, values, weights):
    model = LpProblem(name="Knapsack", sense=LpMaximize)
    
    # Define variables
    variables = [LpVariable(name="X_"+str(i+1), lowBound=0, cat="Integer") for i in range(len(values))]
    
    # Define objective function
    model += lpSum([values[i] * variables[i] for i in range(len(values))])
    
    # Add weight constraint
    backpack_weight = lpSum([weights[i] * variables[i] for i in range(len(values))])
    model += (backpack_weight <= capacity, "Maximum weight")
    
    # Solve the problem using PULP_CBC_CMD solver
    model.solve(solver=PULP_CBC_CMD())
    
    # Display results
    selected_objects = []
    total_weight = 0
    count_per_type = [0] * len(values)  # Initialize counter for each object type to zero

    for i, var in enumerate(variables):
        if var.varValue > 0:
            selected_objects.append(i)
            total_weight += weights[i] * var.varValue
            count_per_type[i] += var.varValue  # Increment the counter for the corresponding object type

    print("Selected objects:", selected_objects)
    print("Total value:", model.objective.value())
    print("Total weight:", total_weight)

    for i, count in enumerate(count_per_type):
        print("Number of objects of type", i, ":", count)

def generate_problem(size):
    print(size)
    capacity = 10*size/2
    values, weights = [random.randint(1,20) for i in range(size)], [random.randint(1,20) for i in range(size)]
    
    return capacity, values, weights

# Problem data
capacity = 16
values = [4,10,15]
weights = [2,5,7]

size=40

#capacity, values, weights = generate_problem(size)

# Solve the knapsack problem
solve_knapsack_problem(capacity, values, weights)
