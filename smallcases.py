import pulp
import numpy as np
import pandas as pd
import math
import time

def solve_fair_tournament_scheduling(n):
    """
    Solve the Fair Round Robin Tournament scheduling problem using Integer Linear Programming
    with focus on the three fairness criteria:
    1. Opponent Ranking Balance
    2. Home/Away Alternation Pattern
    3. Homogeneity Between Players
    
    Parameters:
    n (int): Number of players (must be even)
    
    Returns:
    list: The tournament schedule
    """
    if n % 2 != 0:
        raise ValueError("Number of players must be even")
    
    print(f"Creating fair SRR tournament model for {n} players...")
    start_time = time.time()
    
    # Create model
    model = pulp.LpProblem(name="Fair_Round_Robin_Tournament", sense=pulp.LpMinimize)
    
    # Sets
    players = list(range(1, n+1))  # Players indexed by strength (1 = strongest)
    rounds = list(range(1, n))     # Rounds numbered 1 to n-1
    
    # Create a dictionary to store all variables
    vars_dict = {}
    
    # 1. Match Assignment Variables (x_i,j,r)
    x = {}
    for i in players:
        for j in players:
            if i < j:  # Only need variables for i < j due to symmetry
                for r in rounds:
                    x[i, j, r] = pulp.LpVariable(f"x_{i}_{j}_{r}", cat=pulp.LpBinary)
                    vars_dict[f"x_{i}_{j}_{r}"] = x[i, j, r]
    
    # 2. Home Advantage Variables (h_i,j)
    h = {}
    for i in players:
        for j in players:
            if i < j:  # Only need variables for i < j due to symmetry
                h[i, j] = pulp.LpVariable(f"h_{i}_{j}", cat=pulp.LpBinary)
                vars_dict[f"h_{i}_{j}"] = h[i, j]
    
    # 3. Home/Away Status Variables (y_i,r)
    y = {}
    for i in players:
        for r in rounds:
            y[i, r] = pulp.LpVariable(f"y_{i}_{r}", cat=pulp.LpBinary)
            vars_dict[f"y_{i}_{r}"] = y[i, r]
    
    # 4. Match-Home combined variables (w_i,j,r = 1 if i plays j in round r and i has home advantage)
    w = {}
    for i in players:
        for j in players:
            if i < j:
                for r in rounds:
                    w[i, j, r] = pulp.LpVariable(f"w_{i}_{j}_{r}", cat=pulp.LpBinary)
                    vars_dict[f"w_{i}_{j}_{r}"] = w[i, j, r]
                    
                    # Linearization constraints
                    model += w[i, j, r] <= x[i, j, r], f"w_leq_x_{i}_{j}_{r}"
                    model += w[i, j, r] <= h[i, j], f"w_leq_h_{i}_{j}_{r}"
                    model += w[i, j, r] >= x[i, j, r] + h[i, j] - 1, f"w_geq_x_plus_h_minus_1_{i}_{j}_{r}"
    
    # BASIC CONSTRAINTS
    
    # 1. Every pair of players meets exactly once
    for i in players:
        for j in players:
            if i < j:  # Process each pair only once
                model += pulp.lpSum(x[i, j, r] for r in rounds) == 1, f"meet_once_{i}_{j}"
    
    # 2. Each player plays exactly one match per round
    for i in players:
        for r in rounds:
            model += (
                pulp.lpSum(x[i, j, r] for j in players if j > i) + 
                pulp.lpSum(x[j, i, r] for j in players if j < i)
            ) == 1, f"one_match_per_round_{i}_{r}"
    
    # 3. Linking y with w and x (player i is at home in round r)
    for i in players:
        for r in rounds:
            # i plays at home in round r if:
            # a) i plays j in round r and i has home advantage (represented by w[i,j,r]), or
            # b) j plays i in round r and j does not have home advantage (represented by x[j,i,r] - w[j,i,r])
            model += y[i, r] == (
                pulp.lpSum(w[i, j, r] for j in players if j > i) + 
                pulp.lpSum(x[j, i, r] - w[j, i, r] for j in players if j < i)
            ), f"home_status_{i}_{r}"
    
    # FAIRNESS CRITERIA 1: OPPONENT RANKING BALANCE
    
    # 4. Balance of home matches (each player has roughly the same number of home matches)
    for i in players:
        min_home = math.floor((n-1)/2)
        max_home = math.ceil((n-1)/2)
        model += pulp.lpSum(y[i, r] for r in rounds) >= min_home, f"min_home_matches_{i}"
        model += pulp.lpSum(y[i, r] for r in rounds) <= max_home, f"max_home_matches_{i}"
    
    # 5. Balance of home matches against strong opponents
    for i in players:
        stronger_players = [j for j in players if j < i]  # Players stronger than i
        if stronger_players:  # Only add if there are stronger players
            min_strong_home = math.floor(len(stronger_players)/2)
            max_strong_home = math.ceil(len(stronger_players)/2)
            
            # For players j < i, i plays at home if x[j,i,r] = 1 and w[j,i,r] = 0
            # i.e., sum over all r of (x[j,i,r] - w[j,i,r])
            strong_home_sum = pulp.lpSum(
                pulp.lpSum(x[j, i, r] - w[j, i, r] for r in rounds) 
                for j in stronger_players
            )
            
            model += strong_home_sum >= min_strong_home, f"min_strong_home_{i}"
            model += strong_home_sum <= max_strong_home, f"max_strong_home_{i}"
    
    # 6. Balance of home matches against weak opponents
    for i in players:
        weaker_players = [j for j in players if j > i]  # Players weaker than i
        if weaker_players:  # Only add if there are weaker players
            min_weak_home = math.floor(len(weaker_players)/2)
            max_weak_home = math.ceil(len(weaker_players)/2)
            
            # For players j > i, i plays at home if w[i,j,r] = 1
            # i.e., sum over all r of w[i,j,r]
            weak_home_sum = pulp.lpSum(
                pulp.lpSum(w[i, j, r] for r in rounds) 
                for j in weaker_players
            )
            
            model += weak_home_sum >= min_weak_home, f"min_weak_home_{i}"
            model += weak_home_sum <= max_weak_home, f"max_weak_home_{i}"
    
    # FAIRNESS CRITERIA 2: ALTERNATION PATTERN
    
    # 7. Variables for consecutive home/away games (to minimize)
    z_consec = {}
    for i in players:
        for r in rounds[:-1]:  # Only up to n-2 for consecutive rounds
            z_consec[i, r] = pulp.LpVariable(f"z_consec_{i}_{r}", cat=pulp.LpBinary)
            vars_dict[f"z_consec_{i}_{r}"] = z_consec[i, r]
            
            # z_consec[i,r] = 1 if player i has same venue (both home or both away) in rounds r and r+1
            model += z_consec[i, r] >= y[i, r] + y[i, r+1] - 1, f"z_consec_home_home_{i}_{r}"
            model += z_consec[i, r] >= 1 - y[i, r] - y[i, r+1], f"z_consec_away_away_{i}_{r}"
    
    # 8. Variables for longer consecutive home/away streaks (to heavily penalize)
    if n > 4:  # Only relevant for sufficiently large n
        z_streak = {}
        for i in players:
            for r in range(1, n-2):  # Check for 3 consecutive rounds
                z_streak[i, r] = pulp.LpVariable(f"z_streak_{i}_{r}", cat=pulp.LpBinary)
                vars_dict[f"z_streak_{i}_{r}"] = z_streak[i, r]
                
                # z_streak[i,r] = 1 if player i has 3 consecutive home games
                model += z_streak[i, r] >= y[i, r] + y[i, r+1] + y[i, r+2] - 2, f"z_streak_home_{i}_{r}"
                # z_streak[i,r] = 1 if player i has 3 consecutive away games
                model += z_streak[i, r] >= 3 - y[i, r] - y[i, r+1] - y[i, r+2] - 2, f"z_streak_away_{i}_{r}"
    
    # FAIRNESS CRITERIA 3: HOMOGENEITY BETWEEN PLAYERS
    
    # 9. Calculate pattern differences between player pairs
    pattern_diff = {}
    for i in players:
        for j in players:
            if i < j:
                pattern_diff[i, j] = {}
                for r in rounds:
                    pattern_diff[i, j][r] = pulp.LpVariable(f"pattern_diff_{i}_{j}_{r}", cat=pulp.LpBinary)
                    vars_dict[f"pattern_diff_{i}_{j}_{r}"] = pattern_diff[i, j][r]
                    
                    # pattern_diff[i,j][r] = 1 if players i and j have different home/away status in round r
                    model += pattern_diff[i, j][r] >= y[i, r] - y[j, r], f"pattern_diff_1_{i}_{j}_{r}"
                    model += pattern_diff[i, j][r] >= y[j, r] - y[i, r], f"pattern_diff_2_{i}_{j}_{r}"
    
    # 10. Calculate total pattern differences between player pairs
    total_pattern_diff = {}
    for i in players:
        for j in players:
            if i < j:
                total_pattern_diff[i, j] = pulp.LpVariable(f"total_pattern_diff_{i}_{j}", cat=pulp.LpContinuous, lowBound=0)
                vars_dict[f"total_pattern_diff_{i}_{j}"] = total_pattern_diff[i, j]
                
                model += total_pattern_diff[i, j] == pulp.lpSum(pattern_diff[i, j][r] for r in rounds), f"total_pattern_diff_def_{i}_{j}"
    
    # 11. Maximum pattern difference between any player pair (to minimize)
    max_pattern_diff = pulp.LpVariable("max_pattern_diff", cat=pulp.LpContinuous, lowBound=0)
    vars_dict["max_pattern_diff"] = max_pattern_diff
    
    for i in players:
        for j in players:
            if i < j:
                model += max_pattern_diff >= total_pattern_diff[i, j], f"max_pattern_diff_geq_{i}_{j}"
    
    # ADDITIONAL FAIRNESS METRICS
    
    # 12. Strength-weighted home advantage
    SW = {}
    for i in players:
        SW[i] = pulp.LpVariable(f"SW_{i}", cat=pulp.LpContinuous, lowBound=0)
        vars_dict[f"SW_{i}"] = SW[i]
        
        # Calculate strength-weighted home advantage for player i
        # Sum over all opponents j the strength-weight (n-j) times whether i plays j at home
        model += SW[i] == (
            # For j > i, i plays j at home if w[i,j,r] = 1 summed over all rounds r
            pulp.lpSum((n-j) * pulp.lpSum(w[i, j, r] for r in rounds) for j in players if j > i) + 
            # For j < i, i plays j at home if x[j,i,r] = 1 and w[j,i,r] = 0 summed over all rounds r
            pulp.lpSum((n-j) * pulp.lpSum(x[j, i, r] - w[j, i, r] for r in rounds) for j in players if j < i)
        ), f"SW_def_{i}"
    
    # 13. Maximum difference in strength-weighted home advantage
    delta_max = pulp.LpVariable("delta_max", cat=pulp.LpContinuous, lowBound=0)
    vars_dict["delta_max"] = delta_max
    
    for i in players:
        for j in players:
            if i < j:
                model += delta_max >= SW[i] - SW[j], f"delta_max_geq_diff1_{i}_{j}"
                model += delta_max >= SW[j] - SW[i], f"delta_max_geq_diff2_{i}_{j}"
    
    # MULTI-OBJECTIVE FUNCTION
    
    # Weights for each fairness criterion
    alpha = 3.0  # Weight for consecutive home/away games (Criteria 2)
    beta = 3.0   # Weight for strength-weighted inequality (Criteria 1)
    gamma = 3.0  # Weight for pattern homogeneity between players (Criteria 3)
    
    # Heavy penalty for 3+ consecutive same venues if applicable
    delta = 5.0 if n > 4 else 0
    
    # Build the objective function
    objective = (
        alpha * pulp.lpSum(z_consec[i, r] for i in players for r in rounds[:-1]) +  # Minimize consecutive same venues
        beta * delta_max +  # Minimize strength-weighted inequality
        gamma * max_pattern_diff  # Minimize differences in home/away patterns between players
    )
    
    # Add penalty for 3+ consecutive same venues if applicable
    if n > 4:
        objective += delta * pulp.lpSum(z_streak[i, r] for i in players for r in range(1, n-2))
    
    model += objective
    
    # Solve the model with a time limit
    time_limit = 300  # 5 minutes
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
    model.solve(solver)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Check solution status
    print(f"Solution status: {pulp.LpStatus[model.status]}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusNotSolved:
        # Extract schedule from the solution
        schedule = extract_schedule(x, h, n, vars_dict)
        
        # Print the results
        print_schedule(schedule)
        
        return schedule
    else:
        print("No feasible solution found")
        return None


def extract_schedule(x, h, n, vars_dict):
    """
    Extract the tournament schedule from the solution variables
    """
    players = list(range(1, n+1))
    rounds = list(range(1, n))
    
    schedule = []
    
    for r in rounds:
        round_matches = []
        matched_players = set()
        
        for i in players:
            if i in matched_players:
                continue
                
            for j in players:
                if i < j and f"x_{i}_{j}_{r}" in vars_dict and pulp.value(vars_dict[f"x_{i}_{j}_{r}"]) > 0.5:
                    # Match between i and j in round r
                    if f"h_{i}_{j}" in vars_dict and pulp.value(vars_dict[f"h_{i}_{j}"]) > 0.5:
                        # i has home advantage
                        round_matches.append((i, j))
                    else:
                        # j has home advantage
                        round_matches.append((j, i))
                    
                    matched_players.add(i)
                    matched_players.add(j)
                    break
        
        schedule.append(round_matches)
    
    return schedule


def print_schedule(schedule):
    """
    Print the tournament schedule in a readable format
    """
    print("\nTournament Schedule:")
    for r, round_matches in enumerate(schedule, 1):
        print(f"Round {r}:")
        for home, away in round_matches:
            print(f"  Player {home} (home) vs Player {away} (away)")


def save_schedule_to_csv(schedule, filename):
    """
    Save the tournament schedule to a CSV file
    """
    data = []
    for r, round_matches in enumerate(schedule, 1):
        for home, away in round_matches:
            data.append({
                'Round': r,
                'Home': home,
                'Away': away
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Schedule saved to {filename}")


def main():
    print("Fair Round Robin Tournament Scheduler")
    print("=====================================")
    
    # Ask for the number of players
    while True:
        try:
            n = int(input("Enter the number of players (even number, 4-10 recommended): "))
            if n < 2:
                print("Number of players must be at least 2.")
            elif n % 2 != 0:
                print("Number of players must be even.")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")
    
    # Solve the problem
    print("\nSolving the tournament scheduling problem...")
    schedule = solve_fair_tournament_scheduling(n)
    
    if schedule is not None:
        # Ask if user wants to save the schedule
        save_option = input("\nDo you want to save the schedule? (y/n): ")
        if save_option.lower() == 'y':
            filename = input("Enter filename (default: schedule.csv): ") or "schedule.csv"
            save_schedule_to_csv(schedule, filename)


if __name__ == "__main__":
    main()