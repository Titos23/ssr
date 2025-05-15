import pulp
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time

def solve_fair_tournament_scheduling(n_players, w1=1.0, w2=1.0, w3=1.0):
    """
    Solve the Fair Round Robin Tournament scheduling problem for n players.
    
    Args:
        n_players: Number of players in the tournament (must be even)
        w1: Weight for opponent ranking fairness
        w2: Weight for home/away alternation pattern
        w3: Weight for homogeneity between players
        
    Returns:
        schedule_df: DataFrame with the schedule
        fairness_df: DataFrame with fairness metrics
        compute_time: Computation time in seconds
        objective_value: Value of the objective function
    """
    if n_players % 2 != 0:
        raise ValueError("Number of players must be even")
    
    start_time = time.time()
    
    model = LpProblem(name="Fair_Round_Robin_Tournament", sense=LpMinimize)
    
    n_rounds = n_players - 1
    players = range(1, n_players + 1)
    rounds = range(1, n_rounds + 1)
    matches = list(itertools.combinations(players, 2))
    
    strong_players = list(range(1, (n_players // 2) + 1))
    weak_players = list(range((n_players // 2) + 1, n_players + 1))
    
    # Decision variables
    x = {}
    for i, j in itertools.permutations(players, 2):
        for r in rounds:
            x[i, j, r] = LpVariable(f"x_{i}_{j}_{r}", cat='Binary')
    
    h = {}
    for i in players:
        for r in rounds:
            h[i, r] = LpVariable(f"h_{i}_{r}", cat='Binary')
    
    # Auxiliary variables
    hs = {i: LpVariable(f"hs_{i}", lowBound=0, cat='Integer') for i in players}
    hw = {i: LpVariable(f"hw_{i}", lowBound=0, cat='Integer') for i in players}
    breaks = {i: LpVariable(f"breaks_{i}", lowBound=0, cat='Integer') for i in players}
    home_total = {i: LpVariable(f"home_total_{i}", lowBound=0, cat='Integer') for i in players}
    
    # Objective variables
    delta_opp = LpVariable("delta_opp", lowBound=0)
    delta_breaks = LpVariable("delta_breaks", lowBound=0)
    delta_home = LpVariable("delta_home", lowBound=0)
    
    # Each pair of players meets exactly once
    for i, j in matches:
        model += (
            lpSum(x[i, j, r] + x[j, i, r] for r in rounds) == 1,
            f"pair_{i}_{j}_meets_once"
        )
    
    # Each player plays exactly once per round
    for i in players:
        for r in rounds:
            model += (
                lpSum(x[i, j, r] + x[j, i, r] for j in players if j != i) == 1,
                f"player_{i}_plays_once_round_{r}"
            )
    
    # Define the home/away indicator
    for i in players:
        for r in rounds:
            model += (
                h[i, r] == lpSum(x[i, j, r] for j in players if j != i),
                f"home_indicator_{i}_{r}"
            )
    
    # Count strong opponents at home
    for i in players:
        model += (
            hs[i] == lpSum(x[i, j, r] for j in strong_players if j != i for r in rounds),
            f"strong_at_home_{i}"
        )
    
    # Count weak opponents at home
    for i in players:
        model += (
            hw[i] == lpSum(x[i, j, r] for j in weak_players if j != i for r in rounds),
            f"weak_at_home_{i}"
        )
    
    # Opponent fairness constraint
    for i in players:
        model += (delta_opp >= hs[i] - hw[i], f"opp_fairness_upper_{i}")
        model += (delta_opp >= hw[i] - hs[i], f"opp_fairness_lower_{i}")
    
    # Count breaks (consecutive home or away games)
    for i in players:
        hh = {}  # home followed by home
        aa = {}  # away followed by away
        for r in range(1, n_rounds):
            hh[i, r] = LpVariable(f"hh_{i}_{r}", cat='Binary')
            aa[i, r] = LpVariable(f"aa_{i}_{r}", cat='Binary')
            
            # Linearize hh[i, r] = h[i, r] AND h[i, r+1]
            model += (hh[i, r] <= h[i, r], f"hh_bound1_{i}_{r}")
            model += (hh[i, r] <= h[i, r+1], f"hh_bound2_{i}_{r}")
            model += (hh[i, r] >= h[i, r] + h[i, r+1] - 1, f"hh_bound3_{i}_{r}")
            
            # Linearize aa[i, r] = (1-h[i, r]) AND (1-h[i, r+1])
            model += (aa[i, r] <= 1 - h[i, r], f"aa_bound1_{i}_{r}")
            model += (aa[i, r] <= 1 - h[i, r+1], f"aa_bound2_{i}_{r}")
            model += (aa[i, r] >= 2 - h[i, r] - h[i, r+1] - 1, f"aa_bound3_{i}_{r}")
        
        # Total breaks = sum of consecutive home games + consecutive away games
        model += (
            breaks[i] == lpSum(hh[i, r] + aa[i, r] for r in range(1, n_rounds)),
            f"count_breaks_{i}"
        )
    
    # Break fairness constraint
    for i in players:
        model += (delta_breaks >= breaks[i], f"break_fairness_{i}")
    
    # Count total home games
    for i in players:
        model += (
            home_total[i] == lpSum(h[i, r] for r in rounds),
            f"total_home_{i}"
        )
    
    # Home/away balance constraints
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
    
    # Home game fairness constraint
    for i in players:
        for j in players:
            if i < j:
                model += (delta_home >= home_total[i] - home_total[j], f"home_diff_upper_{i}_{j}")
                model += (delta_home >= home_total[j] - home_total[i], f"home_diff_lower_{i}_{j}")
    
    # Objective function
    model += w1 * delta_opp + w2 * delta_breaks + w3 * delta_home
    
    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300))
    
    # Calculate computation time and get objective value
    compute_time = time.time() - start_time
    objective_value = pulp.value(model.objective)
    
    if model.status != pulp.LpStatusOptimal:
        print(f"No optimal solution found. Status: {LpStatus[model.status]}")
        print(f"Computation Time: {compute_time:.2f} seconds")
        return None, None, compute_time, None
    
    # Extract the solution
    schedule = []
    for r in rounds:
        round_matches = []
        for i, j in itertools.permutations(players, 2):
            if i < j and pulp.value(x[i, j, r]) == 1:
                round_matches.append((i, j))
            elif i > j and pulp.value(x[i, j, r]) == 1:
                round_matches.append((i, j))
        
        round_matches.sort()
        for home, away in round_matches:
            schedule.append({
                'Round': r,
                'Home': home,
                'Away': away,
                'Home_Strength': home,
                'Away_Strength': away
            })
    
    schedule_df = pd.DataFrame(schedule)
    
    # Calculate fairness metrics
    fairness_metrics = {}
    for i in players:
        # Determine home/away pattern
        home_away_pattern = []
        for r in rounds:
            is_home = False
            for _, match in schedule_df[schedule_df['Round'] == r].iterrows():
                if match['Home'] == i:
                    is_home = True
                    break
            home_away_pattern.append('H' if is_home else 'A')
        
        home_matches = schedule_df[schedule_df['Home'] == i]
        pattern = ''.join(home_away_pattern)
        
        # Count consecutive sequences
        break_count = 0
        for k in range(len(pattern) - 1):
            if pattern[k] == pattern[k + 1]:
                break_count += 1
        
        # Count strong and weak opponents at home
        strong_at_home = sum(1 for _, row in home_matches.iterrows() 
                            if row['Away'] in strong_players)
        weak_at_home = sum(1 for _, row in home_matches.iterrows() 
                          if row['Away'] in weak_players)
        
        # Store all metrics
        fairness_metrics[i] = {
            'Home_Games': len(home_matches),
            'Away_Games': n_rounds - len(home_matches),
            'Pattern': pattern,
            'Breaks': break_count,
            'Strong_at_Home': strong_at_home,
            'Weak_at_Home': weak_at_home,
            'Opponent_Fairness': abs(strong_at_home - weak_at_home)
        }
    
    fairness_df = pd.DataFrame.from_dict(fairness_metrics, orient='index')
    fairness_df.index.name = 'Player'
    
    return schedule_df, fairness_df, compute_time, objective_value

def display_schedule(schedule_df, fairness_df, compute_time, n_players, objective_value):
    """
    Write the tournament schedule and metrics to a single file with sections for different player counts.
    Only override the metrics section for a specific player count if the computation time is better than previous runs
    for the same number of players.
    """
    metrics_file = 'tournament_metrics.txt'
    best_times = {}
    current_section = None
    file_content = []
    
    # Try to read existing metrics file to extract best times for each player count
    try:
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                file_content.append(line.rstrip())
                if "TOURNAMENT METRICS FOR" in line:
                    # Extract the player count - safely get the number from the line
                    try:
                        parts = line.split()
                        # Find the part that contains only digits
                        for part in parts:
                            if part.isdigit():
                                player_count = int(part)
                                current_section = player_count
                                break
                    except (ValueError, IndexError):
                        # If we can't parse the player count, just continue
                        pass
                elif "Computation Time:" in line and current_section is not None:
                    time_str = line.split(':')[1].split('seconds')[0].strip()
                    try:
                        best_times[current_section] = float(time_str)
                    except ValueError:
                        # If we can't parse the time, just continue
                        pass
    except FileNotFoundError:
        # File doesn't exist yet
        pass
    
    # Get the best time for current player count, default to infinity if not found
    best_time = best_times.get(n_players, float('inf'))
    
    # If new compute time is better than existing time for this player count, update the section
    if compute_time < best_time:
        # Create the new content for this player count
        new_section = []
        new_section.append(f"===== TOURNAMENT METRICS FOR {n_players} PLAYERS =====")
        new_section.append(f"Computation Time: {compute_time:.2f} seconds")
        new_section.append(f"Objective Value: {objective_value:.5f}")
        
        # Summary metrics
        new_section.append("")
        new_section.append("----- FAIRNESS SUMMARY -----")
        new_section.append(f"Maximum Consecutive Sequences: {fairness_df['Breaks'].max()}")
        new_section.append(f"Home Distribution Fairness (max diff in home games): {fairness_df['Home_Games'].max() - fairness_df['Home_Games'].min()}")
        
        # Check for bad patterns
        bad_patterns = []
        for player, row in fairness_df.iterrows():
            pattern = row['Pattern']
            if 'HHH' in pattern or 'AAA' in pattern:
                bad_patterns.append(f"Player {player}: {pattern}")
        
        if bad_patterns:
            new_section.append("")
            new_section.append("Players with 3+ consecutive home or away games:")
            for pattern in bad_patterns:
                new_section.append(f"  {pattern}")
        else:
            new_section.append("")
            new_section.append("No player has 3+ consecutive home or away games - good alternation!")
        
        # Fairness metrics table
        new_section.append("")
        new_section.append("----- PLAYER METRICS -----")
        new_section.append(f"{'Player':<8} {'Home Games':<12} {'Away Games':<12} {'Pattern':<{n_players}} {'Consecutive':<12}")
        new_section.append('-' * (8 + 12 + 12 + n_players + 12))
        
        for player, row in fairness_df.iterrows():
            new_section.append(f"{player:<8} {row['Home_Games']:<12} {row['Away_Games']:<12} {row['Pattern']:<{n_players}} {row['Breaks']:<12}")
        
        # Schedule table
        new_section.append("")
        new_section.append("----- TOURNAMENT SCHEDULE -----")
        new_section.append(f"{'Round':<8} {'Home':<8} {'Away':<8}")
        new_section.append('-' * 24)
        
        for _, match in schedule_df.iterrows():
            new_section.append(f"{match['Round']:<8} {match['Home']:<8} {match['Away']:<8}")
        
        new_section.append("")
        new_section.append("NOTE: Metrics updated because new compute time is better than previous best.")
        new_section.append("")  # Empty line after section
        
        # If file already exists, replace the section for this player count
        if file_content:
            # Find the section for this player count and replace it
            i = 0
            section_found = False
            while i < len(file_content):
                if f"TOURNAMENT METRICS FOR {n_players} PLAYERS" in file_content[i]:
                    # Found the start of the section to replace
                    section_start = i
                    section_found = True
                    # Find the end of this section (next section or end of file)
                    section_end = len(file_content)
                    for j in range(i + 1, len(file_content)):
                        if "TOURNAMENT METRICS FOR" in file_content[j]:
                            section_end = j
                            break
                    
                    # Replace this section with the new content
                    file_content[section_start:section_end] = new_section
                    break
                i += 1
            
            # If section wasn't found, add it to the end
            if not section_found:
                file_content.extend(new_section)
        else:
            # First time creating the file
            file_content = new_section
        
        # Write the updated content back to the file
        with open(metrics_file, 'w') as f:
            f.write('\n'.join(file_content))
        
        print(f"Metrics updated in {metrics_file} for {n_players} players (better compute time: {compute_time:.2f}s vs previous {best_time:.2f}s)")
    else:
        print(f"Metrics not updated for {n_players} players. Current compute time ({compute_time:.2f}s) not better than existing ({best_time:.2f}s)")
    
    return metrics_file

def visualize_schedule(schedule_df, fairness_df, n_players):
    """
    Create a simplified visualization focused only on the home/away pattern.
    """
    players = range(1, n_players + 1)
    rounds = range(1, n_players)
    
    # Create figure with specific dimensions
    plt.figure(figsize=(12, 8))
    
    # ===== HOME/AWAY PATTERN VISUALIZATION =====
    
    # Get patterns and create pattern matrix
    patterns = fairness_df['Pattern'].to_dict()
    pattern_matrix = np.zeros((n_players, n_players - 1))
    
    for i, p in enumerate(players):
        for j, char in enumerate(patterns[p]):
            pattern_matrix[i, j] = 1 if char == 'H' else -1
    
    # Display the pattern matrix with a custom colormap
    colors = ['#0072B2', '#D55E00']  # Blue for Away, Red for Home
    cmap_pattern = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    im = plt.imshow(pattern_matrix, cmap=cmap_pattern, aspect='auto', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(n_players):
        for j in range(n_players - 1):
            text = 'H' if pattern_matrix[i, j] == 1 else 'A'
            plt.text(j, i, text, ha='center', va='center', color='white', 
                    fontweight='bold', fontsize=12)
    
    # Add strength indicators on y-axis
    strong_players = list(range(1, (n_players // 2) + 1))
    for i, p in enumerate(players):
        strength_text = "Strong" if p in strong_players else "Weak"
        
        # Add colored square for player strength
        color = "#D55E00" if p in strong_players else "#0072B2"
        plt.gca().add_patch(plt.Rectangle((-1.4, i-0.4), 0.8, 0.8, 
                                  facecolor=color, alpha=0.7))
        plt.text(-1.0, i, strength_text[0], ha='center', va='center', 
                color='white', fontweight='bold')
    
    # Add labels and title
    plt.title('Fair Tournament Schedule - Home (H) vs Away (A) Pattern', fontsize=16, fontweight='bold')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Player', fontsize=12)
    
    # Set ticks
    plt.xticks(range(n_players - 1), [f'Round {r}' for r in rounds])
    plt.yticks(range(n_players), [f'Player {p}' for p in players])
    
    # Add colorbar
    cbar = plt.colorbar(im, ticks=[-1, 1])
    cbar.set_ticklabels(['Away', 'Home'])
    
    plt.tight_layout()
    
    # Save and close
    vis_filename = f'tournament_schedule_{n_players}_players.png'
    plt.savefig(vis_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Home/Away pattern visualization saved to {vis_filename}")

def export_schedule(schedule_df, n_players):
    """
    Export the schedule to CSV format.
    """
    schedule_filename = f'tournament_schedule_{n_players}_players.csv'
    schedule_df.to_csv(schedule_filename, index=False)
    print(f"Schedule exported to {schedule_filename}")

# Main execution
if __name__ == "__main__":
    # Ask user for the number of players
    while True:
        try:
            n_players = int(input("Enter the number of players (must be even): "))
            if n_players % 2 != 0:
                print("Number of players must be even. Please try again.")
                continue
            if n_players < 4:
                print("Number of players must be at least 4. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    print(f"\n\n{'='*50}")
    print(f"Solving for {n_players} players")
    print(f"{'='*50}")

    # Weights for different fairness criteria
    w1, w2, w3 = 2.0, 1.5, 1.0

    # Solve the tournament scheduling problem
    schedule_df, fairness_df, compute_time, objective_value = solve_fair_tournament_scheduling(n_players, w1, w2, w3)

    if schedule_df is not None and fairness_df is not None:
        # Display and save metrics
        metrics_file = display_schedule(schedule_df, fairness_df, compute_time, n_players, objective_value)
        print(f"Metrics saved/updated in {metrics_file}")
        
        # Ask about CSV export
        export_choice = input("Do you want to export the schedule to a CSV file? (yes/no): ").lower()
        if export_choice in ['yes', 'y']:
            export_schedule(schedule_df, n_players)
        
        # Ask about visualization
        vis_choice = input("Do you want to generate a visualization of the schedule? (yes/no): ").lower()
        if vis_choice in ['yes', 'y']:
            visualize_schedule(schedule_df, fairness_df, n_players)