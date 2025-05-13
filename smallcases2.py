import pulp
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def solve_fair_tournament_scheduling(n_players, w1=1.0, w2=1.0, w3=1.0):
    """
    Solve the Fair Round Robin Tournament scheduling problem for n players.
    
    Args:
        n_players: Number of players in the tournament (must be even)
        w1: Weight for opponent ranking fairness
        w2: Weight for home/away alternation pattern
        w3: Weight for homogeneity between players
        
    Returns:
        DataFrame with the schedule and fairness metrics
    """
    if n_players % 2 != 0:
        raise ValueError("Number of players must be even")
    
    # Create the model
    model = LpProblem(name="Fair_Round_Robin_Tournament", sense=LpMinimize)
    
    # Parameters
    n_rounds = n_players - 1
    players = range(1, n_players + 1)
    rounds = range(1, n_rounds + 1)
    matches = list(itertools.combinations(players, 2))
    
    # Define strong and weak players
    strong_players = list(range(1, (n_players // 2) + 1))
    weak_players = list(range((n_players // 2) + 1, n_players + 1))
    
    # Decision Variables
    # x[i,j,r] = 1 if player i plays against j in round r with i at home
    x = {}
    for i, j in itertools.permutations(players, 2):
        for r in rounds:
            x[i, j, r] = LpVariable(f"x_{i}_{j}_{r}", cat='Binary')
    
    # h[i,r] = 1 if player i plays at home in round r
    h = {}
    for i in players:
        for r in rounds:
            h[i, r] = LpVariable(f"h_{i}_{r}", cat='Binary')
    
    # Fairness metrics variables
    hs = {i: LpVariable(f"hs_{i}", lowBound=0, cat='Integer') for i in players}  # strong opponents at home
    hw = {i: LpVariable(f"hw_{i}", lowBound=0, cat='Integer') for i in players}  # weak opponents at home
    breaks = {i: LpVariable(f"breaks_{i}", lowBound=0, cat='Integer') for i in players}  # consecutive home/away
    home_total = {i: LpVariable(f"home_total_{i}", lowBound=0, cat='Integer') for i in players}  # total home games
    
    # Objective variables
    delta_opp = LpVariable("delta_opp", lowBound=0)  # Opponent ranking fairness
    delta_breaks = LpVariable("delta_breaks", lowBound=0)  # Home/away alternation
    delta_home = LpVariable("delta_home", lowBound=0)  # Homogeneity between players
    
    # Constraint 1: Each pair of players meets exactly once
    for i, j in matches:
        model += (
            lpSum(x[i, j, r] + x[j, i, r] for r in rounds) == 1,
            f"pair_{i}_{j}_meets_once"
        )
    
    # Constraint 2: Each player plays exactly once per round
    for i in players:
        for r in rounds:
            model += (
                lpSum(x[i, j, r] + x[j, i, r] for j in players if j != i) == 1,
                f"player_{i}_plays_once_round_{r}"
            )
    
    # Constraint 3: Define home/away indicator
    for i in players:
        for r in rounds:
            model += (
                h[i, r] == lpSum(x[i, j, r] for j in players if j != i),
                f"home_indicator_{i}_{r}"
            )
    
    # Constraint 4a: Count strong opponents at home
    for i in players:
        model += (
            hs[i] == lpSum(x[i, j, r] for j in strong_players if j != i for r in rounds),
            f"strong_at_home_{i}"
        )
    
    # Constraint 4b: Count weak opponents at home
    for i in players:
        model += (
            hw[i] == lpSum(x[i, j, r] for j in weak_players if j != i for r in rounds),
            f"weak_at_home_{i}"
        )
    
    # Constraint 4c: Define opponent fairness metric
    for i in players:
        model += (delta_opp >= hs[i] - hw[i], f"opp_fairness_upper_{i}")
        model += (delta_opp >= hw[i] - hs[i], f"opp_fairness_lower_{i}")
    
    # Constraint 5a: Count breaks (consecutive home or away games)
    for i in players:
        # We need auxiliary binary variables for the product of h[i,r] and h[i,r+1]
        hh = {}
        aa = {}
        for r in range(1, n_rounds):
            hh[i, r] = LpVariable(f"hh_{i}_{r}", cat='Binary')  # both home
            aa[i, r] = LpVariable(f"aa_{i}_{r}", cat='Binary')  # both away
            
            # hh[i,r] = 1 iff h[i,r] = 1 and h[i,r+1] = 1
            model += (hh[i, r] <= h[i, r], f"hh_bound1_{i}_{r}")
            model += (hh[i, r] <= h[i, r+1], f"hh_bound2_{i}_{r}")
            model += (hh[i, r] >= h[i, r] + h[i, r+1] - 1, f"hh_bound3_{i}_{r}")
            
            # aa[i,r] = 1 iff h[i,r] = 0 and h[i,r+1] = 0
            model += (aa[i, r] <= 1 - h[i, r], f"aa_bound1_{i}_{r}")
            model += (aa[i, r] <= 1 - h[i, r+1], f"aa_bound2_{i}_{r}")
            model += (aa[i, r] >= 2 - h[i, r] - h[i, r+1] - 1, f"aa_bound3_{i}_{r}")
        
        # Count total breaks
        model += (
            breaks[i] == lpSum(hh[i, r] + aa[i, r] for r in range(1, n_rounds)),
            f"count_breaks_{i}"
        )
    
    # Constraint 5b: Define break fairness metric
    for i in players:
        model += (delta_breaks >= breaks[i], f"break_fairness_{i}")
    
    # Constraint 6a: Count total home games
    for i in players:
        model += (
            home_total[i] == lpSum(h[i, r] for r in rounds),
            f"total_home_{i}"
        )
    
    # Constraint 6b: Ensure balanced home/away distribution
    min_home = (n_rounds) // 2
    max_home = (n_rounds + 1) // 2
    for i in players:
        model += (
            home_total[i] >= min_home,
            f"min_home_{i}"
        )
        model += (
            home_total[i] <= max_home,
            f"max_home_{i}"
        )
    
    # Constraint 6c: Define homogeneity fairness metric
    for i in players:
        for j in players:
            if i < j:  # Only need to compare each pair once
                model += (delta_home >= home_total[i] - home_total[j], f"home_diff_upper_{i}_{j}")
                model += (delta_home >= home_total[j] - home_total[i], f"home_diff_lower_{i}_{j}")
    
    # Objective: Minimize weighted sum of fairness metrics
    model += w1 * delta_opp + w2 * delta_breaks + w3 * delta_home
    
    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=300))
    
    # Check if a solution was found
    if model.status != pulp.LpStatusOptimal:
        print(f"No optimal solution found. Status: {LpStatus[model.status]}")
        return None, None
    
    # Extract the solution
    schedule = []
    for r in rounds:
        round_matches = []
        for i, j in itertools.permutations(players, 2):
            if i < j and pulp.value(x[i, j, r]) == 1:
                round_matches.append((i, j))  # (home, away)
            elif i > j and pulp.value(x[i, j, r]) == 1:
                round_matches.append((i, j))  # (home, away)
        
        # Sort by match index
        round_matches.sort()
        for home, away in round_matches:
            schedule.append({
                'Round': r,
                'Home': home,
                'Away': away,
                'Home_Strength': home,  # Strength = player number (1 is strongest)
                'Away_Strength': away
            })
    
    # Create a DataFrame from the schedule
    schedule_df = pd.DataFrame(schedule)
    
    # Calculate fairness metrics
    fairness_metrics = {}
    for i in players:
        # Extract home/away pattern
        home_away_pattern = []
        for r in rounds:
            is_home = False
            for _, match in schedule_df[schedule_df['Round'] == r].iterrows():
                if match['Home'] == i:
                    is_home = True
                    break
            home_away_pattern.append('H' if is_home else 'A')
        
        # Count strong and weak opponents at home
        home_matches = schedule_df[schedule_df['Home'] == i]
        strong_at_home = sum(1 for _, row in home_matches.iterrows() 
                           if row['Away'] in strong_players and row['Away'] != i)
        weak_at_home = sum(1 for _, row in home_matches.iterrows() 
                         if row['Away'] in weak_players and row['Away'] != i)
        
        # Count breaks
        break_count = 0
        for k in range(len(home_away_pattern) - 1):
            if home_away_pattern[k] == home_away_pattern[k + 1]:
                break_count += 1
        
        fairness_metrics[i] = {
            'Home_Games': len(home_matches),
            'Away_Games': n_rounds - len(home_matches),
            'Pattern': ''.join(home_away_pattern),
            'Strong_at_Home': strong_at_home,
            'Weak_at_Home': weak_at_home,
            'Opponent_Fairness': abs(strong_at_home - weak_at_home),
            'Breaks': break_count
        }
    
    fairness_df = pd.DataFrame.from_dict(fairness_metrics, orient='index')
    fairness_df.index.name = 'Player'
    
    # Overall fairness metrics
    opponent_fairness = fairness_df['Opponent_Fairness'].max()
    break_fairness = fairness_df['Breaks'].max()
    home_fairness = fairness_df['Home_Games'].max() - fairness_df['Home_Games'].min()
    
    print(f"\nSolution found!")
    print(f"Opponent Fairness: {opponent_fairness} (lower is better)")
    print(f"Break Fairness: {break_fairness} (lower is better)")
    print(f"Home Distribution Fairness: {home_fairness} (lower is better)")
    print(f"Objective Value: {pulp.value(model.objective)}")
    
    return schedule_df, fairness_df

def display_schedule(schedule_df, fairness_df):
    """
    Display the tournament schedule in a readable format.
    """
    print("\n===== TOURNAMENT SCHEDULE =====")
    for round_num in sorted(schedule_df['Round'].unique()):
        round_matches = schedule_df[schedule_df['Round'] == round_num]
        print(f"\nRound {round_num}:")
        for _, match in round_matches.iterrows():
            print(f"  Player {match['Home']} (home) vs Player {match['Away']} (away)")
    
    print("\n===== FAIRNESS METRICS =====")
    print(fairness_df)
    
    # Summary statistics
    print("\n===== FAIRNESS SUMMARY =====")
    print(f"Opponent Fairness (max diff strong vs weak): {fairness_df['Opponent_Fairness'].max()}")
    print(f"Break Fairness (max consecutive H/A): {fairness_df['Breaks'].max()}")
    print(f"Home Distribution Fairness (max diff in home games): {fairness_df['Home_Games'].max() - fairness_df['Home_Games'].min()}")
    
    # Check if any player has 3+ consecutive home or away games
    bad_patterns = []
    for player, row in fairness_df.iterrows():
        pattern = row['Pattern']
        if 'HHH' in pattern or 'AAA' in pattern:
            bad_patterns.append(f"Player {player}: {pattern}")
    
    if bad_patterns:
        print("\nPlayers with 3+ consecutive home or away games:")
        for pattern in bad_patterns:
            print(f"  {pattern}")
    else:
        print("\nNo player has 3+ consecutive home or away games - good alternation!")

def visualize_schedule(schedule_df, fairness_df, n_players):
    """
    Visualize the home/away pattern for each player.
    """
    # Create a matrix for the visualization
    players = range(1, n_players + 1)
    rounds = range(1, n_players)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Visualize the home/away pattern
    # Extract patterns from fairness_df
    patterns = fairness_df['Pattern'].to_dict()
    pattern_matrix = np.zeros((n_players, n_players - 1))
    
    for i, p in enumerate(players):
        for j, char in enumerate(patterns[p]):
            pattern_matrix[i, j] = 1 if char == 'H' else -1
    
    # Plot the heatmap
    im1 = ax1.imshow(pattern_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Add labels and colorbar
    ax1.set_xticks(range(n_players - 1))
    ax1.set_xticklabels([f'R{r}' for r in rounds])
    ax1.set_yticks(range(n_players))
    ax1.set_yticklabels([f'P{p}' for p in players])
    ax1.set_title('Home (H) vs Away (A) Pattern')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Player')
    
    # Add text annotations
    for i in range(n_players):
        for j in range(n_players - 1):
            text = 'H' if pattern_matrix[i, j] == 1 else 'A'
            ax1.text(j, i, text, ha='center', va='center', color='black')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[-1, 1])
    cbar1.set_ticklabels(['Away', 'Home'])
    
    # 2. Visualize the strength of opponents at home
    opponent_matrix = np.zeros((n_players, 2))
    
    for i, p in enumerate(players):
        opponent_matrix[i, 0] = fairness_df.loc[p, 'Strong_at_Home']
        opponent_matrix[i, 1] = fairness_df.loc[p, 'Weak_at_Home']
    
    # Plot the heatmap
    im2 = ax2.imshow(opponent_matrix, cmap='YlGnBu', aspect='auto')
    
    # Add labels and colorbar
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Strong', 'Weak'])
    ax2.set_yticks(range(n_players))
    ax2.set_yticklabels([f'P{p}' for p in players])
    ax2.set_title('Strong vs Weak Opponents at Home')
    ax2.set_xlabel('Opponent Type')
    ax2.set_ylabel('Player')
    
    # Add text annotations
    for i in range(n_players):
        for j in range(2):
            ax2.text(j, i, int(opponent_matrix[i, j]), ha='center', va='center', color='black')
    
    # Add colorbar
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'tournament_schedule_{n_players}_players.png')
    plt.show()

def export_schedule(schedule_df, fairness_df, n_players):
    """
    Export the schedule and fairness metrics to CSV format.
    """
    schedule_filename = f'tournament_schedule_{n_players}_players.csv'
    fairness_filename = f'fairness_metrics_{n_players}_players.csv'
    
    schedule_df.to_csv(schedule_filename, index=False)
    fairness_df.to_csv(fairness_filename)
    
    print(f"Schedule exported to {schedule_filename}")
    print(f"Fairness metrics exported to {fairness_filename}")

# Run the solver for different numbers of players
# We can adjust weights to prioritize different fairness criteria
for n_players in [4, 6, 8]:
    print(f"\n\n{'='*50}")
    print(f"Solving for {n_players} players")
    print(f"{'='*50}")
    
    # Set weights for different fairness criteria
    # w1: Opponent ranking fairness
    # w2: Home/away alternation pattern
    # w3: Homogeneity between players
    w1, w2, w3 = 2.0, 1.5, 1.0
    
    schedule_df, fairness_df = solve_fair_tournament_scheduling(n_players, w1, w2, w3)
    
    if schedule_df is not None and fairness_df is not None:
        display_schedule(schedule_df, fairness_df)
        visualize_schedule(schedule_df, fairness_df, n_players)
        export_schedule(schedule_df, fairness_df, n_players)