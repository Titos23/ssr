import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value
import numpy as np
import pandas as pd

# Parameters
n = 10
players = range(1, n + 1)
rounds = range(1, n)
ratings = {i: 2000 - 100 * (i - 1) for i in players}  # Ratings: 2000, 1900, ..., 1100

# LP Problem
prob = LpProblem("SRR_Fair_Scheduling", LpMinimize)

# Variables
x = {(i, j, r): LpVariable(f"x_{i}_{j}_{r}", cat=LpBinary)
     for i in players for j in players if i != j for r in rounds}
h = {(i, j, r): LpVariable(f"h_{i}_{j}_{r}", cat=LpBinary)
     for i in players for j in players if i != j for r in rounds}
b = {(i, r): LpVariable(f"b_{i}_{r}", cat=LpBinary)
     for i in players for r in rounds[1:]}
N = {i: LpVariable(f"N_{i}", lowBound=0) for i in players}

# Constraints
for i in players:
    for r in rounds:
        prob += lpSum(x[i, j, r] for j in players if j != i) == 1
for i in players:
    for j in players:
        if i != j:
            for r in rounds:
                prob += x[i, j, r] == x[j, i, r]
                prob += h[i, j, r] + h[j, i, r] == x[i, j, r]
                prob += h[i, j, r] <= x[i, j, r]
for i in players:
    for j in players:
        if i != j:
            prob += lpSum(x[i, j, r] for r in rounds) == 1
for i in players:
    for r in rounds[1:]:
        for j in players:
            for k in players:
                if j != i and k != i:
                    prob += h[i, j, r-1] + h[i, k, r] <= 1 + b[i, r]
                    prob += 2 - h[i, j, r-1] - h[i, k, r] <= 1 + b[i, r]
for i in players:
    prob += N[i] == lpSum(h[i, j, r] for j in players if j != i for r in rounds)

# Objective: Minimize breaks and home game variance
total_breaks = lpSum(b[i, r] for i in players for r in rounds[1:])
mean_N = lpSum(N[i] for i in players) / n
dev_N = {i: LpVariable(f"dev_N_{i}", lowBound=0) for i in players}
for i in players:
    prob += dev_N[i] >= N[i] - mean_N
    prob += dev_N[i] >= mean_N - N[i]

# Composite objective (equal weights for breaks and home game variance)
max_breaks = n * (n - 2)  # Approx max breaks
max_dev_N = n  # Approx max deviation
prob += (0.5) * (total_breaks / max_breaks) + (0.5) * (lpSum(dev_N[i] for i in players) / max_dev_N)

# Solve with 60-second timeout
prob.solve(solver=pulp.PULP_CBC_CMD(timeLimit=120, msg=1))

# Output results
if prob.status in [pulp.LpStatusOptimal, pulp.LpStatusNotSolved]:
    # Save schedule to CSV
    schedule_data = []
    for r in rounds:
        for i in players:
            for j in players:
                if i < j and value(x[i, j, r]) > 0.5:
                    home = f"P{i}" if value(h[i, j, r]) > 0.5 else f"P{j}"
                    away = f"P{j}" if value(h[i, j, r]) > 0.5 else f"P{i}"
                    schedule_data.append({"Round": r, "Home": home, "Away": away})
    pd.DataFrame(schedule_data).to_csv("schedule_n10.csv", index=False)

    # Print schedule
    for r in rounds:
        print(f"Round {r}:")
        for data in schedule_data:
            if data["Round"] == r:
                print(f"  {data['Home']} vs {data['Away']}")

    # Compute fairness metrics
    H_values = [sum(value(h[i, j, r]) * ratings[j] for j in players if j != i for r in rounds)
                for i in players]
    N_values = [sum(value(h[i, j, r]) for j in players if j != i for r in rounds)
                for i in players]
    breaks = sum(value(b[i, r]) for i in players for r in rounds[1:])
    print(f"Opponent Strength Variance: {np.var(H_values)}")
    print(f"Total Breaks: {breaks}")
    print(f"Home Games Variance: {np.var(N_values)}")
else:
    print("No feasible solution found within time limit.")