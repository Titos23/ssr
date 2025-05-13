"""Show of the limitations of linear programming for the bin packing-problem"""

import csv
from pulp import GLPK
from pulp import PULP_CBC_CMD
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
from datetime import datetime
import pprint
import itertools
import random


def generate_problem(size, bin_size):
    # print(size)

    objects = [random.randint(1, bin_size) for i in range(size)]
    return objects


def solve_linear_programming(bin_size, objects):
    S = len(objects)

    # 1. Initialise the problem and the binary decision variables (one for each item).
    model = LpProblem(name="Bin packing problem", sense=LpMinimize)
    bins = [LpVariable(name="Bin_"+str(y+1), cat="Binary") for y in range(S)]

    # bins_objects[y][i] = 1 if object i in bin y
    bins_objects = [[LpVariable(name="Object_"+str(i)+"in_bin_"+str(y), cat="Binary")
                    for i in range(S)]
                    for y in range(len(bins))]


    # 2. the bin loading constraint
    for y in range(len(bins)):
        load = sum([objects[i]*bins_objects[y][i] for i in range(S)])
        model += (load <= bin_size*bins[y] , "Bin_" + str(y) + "not overused")   # also forces the value of bins[y]

    # 3. each object assigned
    for i in range(S):
        assignments = sum([bins_objects[y][i] for y in range(len(bins))])
        model += (assignments >= 1, "object "+str(i) +"must be stored in one bin.")
        model += (assignments <= 1, "object "+str(i) +"can be stored only once.")


    # Solve
    model += lpSum(bins)

    # status = model.solve(solver=GLPK(msg=True, options=['--mipgap', '1e-10', "--tmlim", "300"],
    #                                  keepFiles=True))
    status = model.solve(solver=PULP_CBC_CMD(msg=True, timeLimit=1500, options=['--mipgap', '1e-10'],
                                     keepFiles=True))


    # Check the results:
    print("Bin, object weights, total weight")
    for y,b in enumerate(bins):
        if b.varValue >= 1:
            print("bin_" + str(y) + ":", end=" ")
            weight = 0
            for i,o in enumerate(bins_objects[y]):
                if o.varValue >= 1:
                    weight += objects[i]
                    print(objects[i], end=" ")
            print("\t total weight:", weight)


def best_fit(objects, bin_size):
    def best_bin(bins, o, bin_size):
        best = -1
        load = 0
        for i, b in enumerate(bins):
            if b + o <= bin_size and b + o > load:
                best, load = i, b + o

        return best


    bins = []
    for o in objects:
        best = best_bin(bins, o, bin_size)
        if best == -1:
            bins.append(o)
        else:
            bins[best] += o

    return bins


if __name__ == "__main__":
    # some tests.
    random.seed(0)
    s = 1000
    s = 15
    bin_size = 20
    objects = generate_problem(s, bin_size)
    solve_linear_programming(bin_size, objects)
    # objects.sort(reverse=True)
    # print(objects)
    # bins = best_fit(objects, bin_size)
    # print(len(bins))
    # print(bins)

