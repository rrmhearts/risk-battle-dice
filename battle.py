"""
Risk Battle Probability Calculator Library

Calculates exact probabilities for Risk game battles including:
- Normal battles (1-3 attackers vs 1-2 defenders)
- Capital defense bonuses (extra die for defender)
- Balanced Blitz mode (adjusts final battle outcome probabilities)
"""

from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt


class RiskBattle:
    """Calculate Risk battle probabilities."""
    
    # Balanced Blitz configuration constants
    BALANCE_POWER = 0.3  # Exponent for outcome probability adjustment
    
    @staticmethod
    def roll_probabilities(n_dice):
        """Generate all possible outcomes for n dice rolls."""
        return list(product(range(1, 7), repeat=n_dice))
    
    @staticmethod
    def compare_dice(attacker_dice, defender_dice):
        """
        Compare dice rolls and determine losses.
        Returns (attacker_losses, defender_losses)
        """
        # Sort dice in descending order
        att_sorted = sorted(attacker_dice, reverse=True)
        def_sorted = sorted(defender_dice, reverse=True)
        
        attacker_losses = 0
        defender_losses = 0
        
        # Compare dice pairwise (highest vs highest, etc.)
        comparisons = min(len(att_sorted), len(def_sorted))
        for i in range(comparisons):
            if att_sorted[i] > def_sorted[i]:
                defender_losses += 1
            else:
                attacker_losses += 1
        
        return attacker_losses, defender_losses
    
    @classmethod
    def single_round_probabilities(cls, n_attackers, n_defenders, capital_defense=False):
        """
        Calculate probabilities for a single round of combat.
        
        Args:
            n_attackers: Number of attacking armies (1-3 dice used)
            n_defenders: Number of defending armies (1-2 dice used, or 1-3 if capital)
            capital_defense: If True, defender gets an extra die
        
        Returns:
            dict: {(attacker_losses, defender_losses): probability}
        """
        # Determine number of dice
        attacker_dice = min(3, n_attackers)
        if capital_defense:
            defender_dice = min(3, n_defenders)
        else:
            defender_dice = min(2, n_defenders)
        
        # Generate all possible rolls
        attacker_rolls = cls.roll_probabilities(attacker_dice)
        defender_rolls = cls.roll_probabilities(defender_dice)
        
        # Count outcomes
        outcomes = defaultdict(int)
        total = 0
        
        for att_roll in attacker_rolls:
            for def_roll in defender_rolls:
                result = cls.compare_dice(att_roll, def_roll)
                outcomes[result] += 1
                total += 1
        
        # Convert to probabilities
        probabilities = {k: v / total for k, v in outcomes.items()}
        return probabilities
    
    @classmethod
    def battle_probabilities(cls, n_attackers, n_defenders, capital_defense=False, 
                           max_rounds=100):
        """
        Calculate probabilities for complete battle until one side is eliminated.
        Uses dynamic programming.
        
        Args:
            n_attackers: Initial number of attacking armies
            n_defenders: Initial number of defending armies
            capital_defense: If True, defender gets capital bonus
            max_rounds: Maximum rounds to calculate (prevents infinite loops)
        
        Returns:
            dict: {
                'attacker_wins': probability,
                'defender_wins': probability,
                'attacker_survivors': {count: probability},
                'defender_survivors': {count: probability}
            }
        """
        # Memoization for dynamic programming
        memo = {}
        
        def dp(att, dfd):
            """Recursively calculate probabilities from state (att, dfd)."""
            if att == 0:
                return {'attacker_wins': 0.0, 'defender_wins': 1.0,
                       'attacker_survivors': {0: 1.0}, 'defender_survivors': {dfd: 1.0}}
            if dfd == 0:
                return {'attacker_wins': 1.0, 'defender_wins': 0.0,
                       'attacker_survivors': {att: 1.0}, 'defender_survivors': {0: 1.0}}
            
            if (att, dfd) in memo:
                return memo[(att, dfd)]
            
            # Get single round probabilities (NO Balanced Blitz adjustment here)
            round_probs = cls.single_round_probabilities(att, dfd, capital_defense)

            result = {
                'attacker_wins': 0.0,
                'defender_wins': 0.0,
                'attacker_survivors': defaultdict(float),
                'defender_survivors': defaultdict(float)
            }
            
            # Aggregate over all possible outcomes
            for (att_loss, def_loss), prob in round_probs.items():
                new_att = att - att_loss
                new_dfd = dfd - def_loss
                
                sub_result = dp(new_att, new_dfd)
                
                result['attacker_wins'] += prob * sub_result['attacker_wins']
                result['defender_wins'] += prob * sub_result['defender_wins']
                
                for survivors, survivor_prob in sub_result['attacker_survivors'].items():
                    result['attacker_survivors'][survivors] += prob * survivor_prob
                for survivors, survivor_prob in sub_result['defender_survivors'].items():
                    result['defender_survivors'][survivors] += prob * survivor_prob
            
            # Convert defaultdict to regular dict
            result['attacker_survivors'] = dict(result['attacker_survivors'])
            result['defender_survivors'] = dict(result['defender_survivors'])
            
            memo[(att, dfd)] = result
            return result
        
        return dp(n_attackers, n_defenders)


def apply_balanced_blitz(outcome_probabilities):
    """
    Apply Balanced Blitz algorithm to outcome probabilities.
    
    Balanced Blitz adjusts probabilities to push high-probability outcomes 
    closer to 100% and low-probability outcomes closer to 0%.
    
    Based on RISK: Global Domination's Balanced Blitz algorithm which:
    - Raises each outcome probability to a power (< 1.0)
    - Renormalizes so probabilities sum to 1.0
    - This compresses the middle and expands extremes
    
    Args:
        outcome_probabilities: dict of {outcome: probability}
    
    Returns:
        dict: adjusted probabilities
    """
    # Apply power transformation to each probability
    adjusted = {}
    for outcome, prob in outcome_probabilities.items():
        # Raise to power < 1 to push extremes
        adjusted[outcome] = prob ** RiskBattle.BALANCE_POWER
    
    # Renormalize so probabilities sum to 1.0
    total = sum(adjusted.values())
    normalized = {k: v / total for k, v in adjusted.items()}
    
    return normalized


def battle_probabilities_balanced_blitz(n_attackers, n_defenders, capital_defense=False):
    """
    Calculate battle probabilities using Balanced Blitz algorithm.
    
    This simulates RISK: Global Domination's Balanced Blitz mode 
    which adjusts FINAL battle outcome probabilities to make high-probability wins more likely
    and low-probability wins less likely. The adjustment is applied to the complete battle
    results, NOT to individual combat rounds.
    
    Args:
        n_attackers: Initial number of attacking armies
        n_defenders: Initial number of defending armies
        capital_defense: If True, defender gets capital bonus
    
    Returns:
        dict: same format as battle_probabilities but with adjusted final probabilities
    """
    # First, calculate the true random battle probabilities
    true_random_result = RiskBattle.battle_probabilities(n_attackers, n_defenders, capital_defense)
    
    # Extract just the win/loss probabilities for Balanced Blitz adjustment
    outcome_probs = {
        'attacker_wins': true_random_result['attacker_wins'],
        'defender_wins': true_random_result['defender_wins']
    }
    
    # Apply Balanced Blitz adjustment to the FINAL battle outcomes
    adjusted_outcomes = apply_balanced_blitz(outcome_probs)
    
    # Calculate the adjustment ratios
    attacker_ratio = adjusted_outcomes['attacker_wins'] / true_random_result['attacker_wins'] if true_random_result['attacker_wins'] > 0 else 0
    defender_ratio = adjusted_outcomes['defender_wins'] / true_random_result['defender_wins'] if true_random_result['defender_wins'] > 0 else 0
    
    # Apply the same ratios to the survivor distributions
    # This maintains consistency between win probabilities and survivor counts
    adjusted_attacker_survivors = {}
    adjusted_defender_survivors = {}
    
    # Adjust attacker survivor probabilities
    for survivors, prob in true_random_result['attacker_survivors'].items():
        if survivors > 0:  # Attacker wins
            adjusted_attacker_survivors[survivors] = prob * attacker_ratio
        else:  # Attacker loses (0 survivors)
            adjusted_attacker_survivors[survivors] = prob * defender_ratio
    
    # Adjust defender survivor probabilities
    for survivors, prob in true_random_result['defender_survivors'].items():
        if survivors > 0:  # Defender wins
            adjusted_defender_survivors[survivors] = prob * defender_ratio
        else:  # Defender loses (0 survivors)
            adjusted_defender_survivors[survivors] = prob * attacker_ratio
    
    return {
        'attacker_wins': adjusted_outcomes['attacker_wins'],
        'defender_wins': adjusted_outcomes['defender_wins'],
        'attacker_survivors': adjusted_attacker_survivors,
        'defender_survivors': adjusted_defender_survivors
    }


def print_single_round_analysis(n_attackers, n_defenders, capital=False):
    """Print analysis of a single round of combat."""
    print(f"\n{'='*70}")
    print(f"SINGLE ROUND: {n_attackers} attackers vs {n_defenders} defenders" + 
          (f" (CAPITAL)" if capital else ""))
    print(f"{'='*70}")
    
    probs = RiskBattle.single_round_probabilities(n_attackers, n_defenders, capital)
    
    for (att_loss, def_loss), prob in sorted(probs.items()):
        print(f"  Attacker loses {att_loss}, Defender loses {def_loss}: {prob:.4f} ({prob*100:.2f}%)")


def print_full_battle_analysis(n_attackers, n_defenders, capital=False):
    """Print analysis of a complete battle."""
    print(f"\n{'='*70}")
    print(f"FULL BATTLE: {n_attackers} attackers vs {n_defenders} defenders" + 
          (f" (CAPITAL)" if capital else ""))
    print(f"{'='*70}")
    
    result = RiskBattle.battle_probabilities(n_attackers, n_defenders, capital)
    
    print(f"\nOverall Probabilities:")
    print(f"  Attacker wins: {result['attacker_wins']:.4f} ({result['attacker_wins']*100:.2f}%)")
    print(f"  Defender wins: {result['defender_wins']:.4f} ({result['defender_wins']*100:.2f}%)")
    
    print(f"\nAttacker survivors distribution:")
    for survivors in sorted(result['attacker_survivors'].keys(), reverse=True):
        prob = result['attacker_survivors'][survivors]
        if prob > 0.001:
            print(f"  {survivors} survivors: {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\nDefender survivors distribution:")
    for survivors in sorted(result['defender_survivors'].keys(), reverse=True):
        prob = result['defender_survivors'][survivors]
        if prob > 0.001:
            print(f"  {survivors} survivors: {prob:.4f} ({prob*100:.2f}%)")


def print_balanced_blitz_comparison(n_attackers, n_defenders, capital=False):
    """Print comparison of True Random vs Balanced Blitz."""
    print(f"\n{'='*70}")
    print(f"BALANCED BLITZ COMPARISON: {n_attackers} attackers vs {n_defenders} defenders" +
          (f" (CAPITAL)" if capital else ""))
    print(f"{'='*70}")
    
    result_tr = RiskBattle.battle_probabilities(n_attackers, n_defenders, capital)
    result_bb = battle_probabilities_balanced_blitz(n_attackers, n_defenders, capital)
    
    print(f"\nAttacker Win Probability:")
    print(f"  True Random:    {result_tr['attacker_wins']:.4f} ({result_tr['attacker_wins']*100:.2f}%)")
    print(f"  Balanced Blitz: {result_bb['attacker_wins']:.4f} ({result_bb['attacker_wins']*100:.2f}%)")
    
    diff = result_bb['attacker_wins'] - result_tr['attacker_wins']
    direction = "HIGHER" if diff > 0 else "LOWER"
    print(f"  Difference:     {diff:+.4f} ({diff*100:+.2f}%) - Balanced Blitz is {direction}")
    
    # Show which mode is better based on initial odds
    if result_tr['attacker_wins'] > 0.75:
        print(f"  Recommendation: BLITZ (high odds favor attacker more)")
    elif result_tr['attacker_wins'] < 0.25:
        print(f"  Recommendation: MANUAL ROLL (low odds hurt attacker more in blitz)")
    else:
        print(f"  Recommendation: Player choice (odds are in middle range)")
    
    # Additional verification that probabilities sum to 1.0
    tr_sum = result_tr['attacker_wins'] + result_tr['defender_wins']
    bb_sum = result_bb['attacker_wins'] + result_bb['defender_wins']
    print(f"\nProbability sum verification:")
    print(f"  True Random sum:    {tr_sum:.6f}")
    print(f"  Balanced Blitz sum: {bb_sum:.6f}")


def plot_probability_vs_troops():
    """Plot how attacker win probability changes with troop numbers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Risk Battle Probabilities vs Troop Numbers', fontsize=14, fontweight='bold')
    
    # Plot 1: Fixed defenders (5), varying attackers
    ax = axes[0, 0]
    defenders = 5
    attackers_range = range(1, 21)
    
    normal_probs = []
    capital_probs = []
    
    for att in attackers_range:
        result_normal = RiskBattle.battle_probabilities(att, defenders, False)
        result_capital = RiskBattle.battle_probabilities(att, defenders, True)
        normal_probs.append(result_normal['attacker_wins'])
        capital_probs.append(result_capital['attacker_wins'])
    
    ax.plot(attackers_range, normal_probs, 'b-o', label='Normal Defense', linewidth=2)
    ax.plot(attackers_range, capital_probs, 'r-s', label='Capital Defense', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax.set_xlabel('Number of Attackers', fontsize=9)
    ax.set_ylabel('Attacker Win Probability', fontsize=9)
    ax.set_title(f'Attackers vs {defenders} Defenders', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Plot 2: Fixed attackers (10), varying defenders
    ax = axes[0, 1]
    attackers = 10
    defenders_range = range(1, 16)
    
    normal_probs = []
    capital_probs = []
    
    for dfd in defenders_range:
        result_normal = RiskBattle.battle_probabilities(attackers, dfd, False)
        result_capital = RiskBattle.battle_probabilities(attackers, dfd, True)
        normal_probs.append(result_normal['attacker_wins'])
        capital_probs.append(result_capital['attacker_wins'])
    
    ax.plot(defenders_range, normal_probs, 'b-o', label='Normal Defense', linewidth=2)
    ax.plot(defenders_range, capital_probs, 'r-s', label='Capital Defense', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax.set_xlabel('Number of Defenders', fontsize=9)
    ax.set_ylabel('Attacker Win Probability', fontsize=9)
    ax.set_title(f'{attackers} Attackers vs Varying Defenders', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Plot 3: Heatmap of win probabilities (Normal)
    ax = axes[1, 0]
    att_range = range(1, 11)
    def_range = range(1, 11)
    
    probs_matrix = []
    for dfd in def_range:
        row = []
        for att in att_range:
            result = RiskBattle.battle_probabilities(att, dfd, False)
            row.append(result['attacker_wins'])
        probs_matrix.append(row)
    
    im = ax.imshow(probs_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(att_range)))
    ax.set_yticks(range(len(def_range)))
    ax.set_xticklabels(att_range)
    ax.set_yticklabels(def_range)
    ax.set_xlabel('Number of Attackers', fontsize=11)
    ax.set_ylabel('Number of Defenders', fontsize=11)
    ax.set_title('Normal Defense Win Probability Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attacker Win Probability')
    
    # Plot 4: Heatmap of win probabilities (Capital)
    ax = axes[1, 1]
    probs_matrix = []
    for dfd in def_range:
        row = []
        for att in att_range:
            result = RiskBattle.battle_probabilities(att, dfd, True)
            row.append(result['attacker_wins'])
        probs_matrix.append(row)
    
    im = ax.imshow(probs_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(att_range)))
    ax.set_yticks(range(len(def_range)))
    ax.set_xticklabels(att_range)
    ax.set_yticklabels(def_range)
    ax.set_xlabel('Number of Attackers', fontsize=11)
    ax.set_ylabel('Number of Defenders', fontsize=11)
    ax.set_title('Capital Defense Win Probability Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Attacker Win Probability')
    
    plt.tight_layout()
    plt.show()


def plot_balanced_blitz_comparison():
    """Plot comparison between True Random and Balanced Blitz probabilities."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('True Random vs Balanced Blitz Comparison', fontsize=14, fontweight='bold')
    
    # Plot 1: Fixed defenders (5), varying attackers - comparison
    ax = axes[0, 0]
    defenders = 5
    attackers_range = range(1, 21)
    
    tr_probs = []
    bb_probs = []
    
    for att in attackers_range:
        result_tr = RiskBattle.battle_probabilities(att, defenders, False)
        result_bb = battle_probabilities_balanced_blitz(att, defenders, False)
        tr_probs.append(result_tr['attacker_wins'])
        bb_probs.append(result_bb['attacker_wins'])
    
    ax.plot(attackers_range, tr_probs, 'b-o', label='True Random', linewidth=2)
    ax.plot(attackers_range, bb_probs, 'r-s', label='Balanced Blitz', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.75, color='green', linestyle=':', alpha=0.5, label='75% threshold')
    ax.axhline(y=0.25, color='orange', linestyle=':', alpha=0.5, label='25% threshold')
    ax.set_xlabel('Number of Attackers', fontsize=9)
    ax.set_ylabel('Attacker Win Probability', fontsize=9)
    ax.set_title(f'Attackers vs {defenders} Defenders', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    
    # Plot 2: Difference plot (Balanced Blitz - True Random)
    ax = axes[0, 1]
    differences = [bb - tr for bb, tr in zip(bb_probs, tr_probs)]
    
    colors = ['green' if d > 0 else 'red' for d in differences]
    ax.bar(attackers_range, differences, color=colors, alpha=0.6)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Number of Attackers', fontsize=9)
    ax.set_ylabel('Probability Difference (BB - TR)', fontsize=9)
    ax.set_title(f'Balanced Blitz Advantage vs {defenders} Defenders', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Multiple defender counts comparison
    ax = axes[1, 0]
    attackers = 10
    defenders_range = range(1, 16)
    
    tr_probs = []
    bb_probs = []
    
    for dfd in defenders_range:
        result_tr = RiskBattle.battle_probabilities(attackers, dfd, False)
        result_bb = battle_probabilities_balanced_blitz(attackers, dfd, False)
        tr_probs.append(result_tr['attacker_wins'])
        bb_probs.append(result_bb['attacker_wins'])
    
    ax.plot(defenders_range, tr_probs, 'b-o', label='True Random', linewidth=2)
    ax.plot(defenders_range, bb_probs, 'r-s', label='Balanced Blitz', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.75, color='green', linestyle=':', alpha=0.5, label='75% threshold')
    ax.axhline(y=0.25, color='orange', linestyle=':', alpha=0.5, label='25% threshold')
    ax.set_xlabel('Number of Defenders', fontsize=9)
    ax.set_ylabel('Attacker Win Probability', fontsize=9)
    ax.set_title(f'{attackers} Attackers vs Varying Defenders', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    
    # Plot 4: Heatmap of probability differences
    ax = axes[1, 1]
    att_range = range(1, 11)
    def_range = range(1, 11)
    
    diff_matrix = []
    for dfd in def_range:
        row = []
        for att in att_range:
            result_tr = RiskBattle.battle_probabilities(att, dfd, False)
            result_bb = battle_probabilities_balanced_blitz(att, dfd, False)
            diff = result_bb['attacker_wins'] - result_tr['attacker_wins']
            row.append(diff)
        diff_matrix.append(row)
    
    im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(att_range)))
    ax.set_yticks(range(len(def_range)))
    ax.set_xticklabels(att_range)
    ax.set_yticklabels(def_range)
    ax.set_xlabel('Number of Attackers', fontsize=9)
    ax.set_ylabel('Number of Defenders', fontsize=9)
    ax.set_title('BB Advantage Heatmap (Green=Better for Attacker)', fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='Probability Difference')
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("RISK BATTLE PROBABILITY CALCULATOR")
    print("="*70)
    
    # Single round examples
    print("\n" + "="*70)
    print("SINGLE ROUND PROBABILITIES")
    print("="*70)
    
    print_single_round_analysis(3, 2, capital=False)
    print_single_round_analysis(3, 2, capital=True)
    print_single_round_analysis(1, 1, capital=False)
    print_single_round_analysis(2, 1, capital=False)
    
    # Full battle examples
    print("\n" + "="*70)
    print("COMPLETE BATTLE PROBABILITIES")
    print("="*70)
    
    print_full_battle_analysis(5, 3, capital=False)
    print_full_battle_analysis(5, 3, capital=True)
    print_full_battle_analysis(10, 7, capital=False)
    print_full_battle_analysis(10, 7, capital=True)
    print_full_battle_analysis(3, 3, capital=False)
    
    # Generate probability graphs
    print("\n" + "="*70)
    print("GENERATING PROBABILITY GRAPHS...")
    print("="*70)
    plot_probability_vs_troops()
    
    # Balanced Blitz comparisons
    print("\n" + "="*70)
    print("BALANCED BLITZ vs TRUE RANDOM ANALYSIS")
    print("="*70)
    
    print_balanced_blitz_comparison(5, 3, capital=False)
    print_balanced_blitz_comparison(10, 7, capital=False)
    print_balanced_blitz_comparison(2, 1, capital=False)  # Low odds for attacker
    print_balanced_blitz_comparison(15, 5, capital=False)  # High odds for attacker
    print_balanced_blitz_comparison(4, 1, capital=False)  # Very high odds
    
    # Generate Balanced Blitz comparison graphs
    print("\n" + "="*70)
    print("GENERATING BALANCED BLITZ COMPARISON GRAPHS...")
    print("="*70)
    plot_balanced_blitz_comparison()