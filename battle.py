"""
Risk Battle Probability Calculator Library

Calculates exact probabilities for Risk game battles including:
- Normal battles (1-3 attackers vs 1-2 defenders)
- Capital defense bonuses (extra die for defender)
"""

from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt


class RiskBattle:
    """Calculate Risk battle probabilities."""
    
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
            
            # Get single round probabilities
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