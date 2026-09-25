"""
Interactive Demo Script for Texas Hold'em ML Model.
Demonstrates scenario predictions, action probabilities, and statistical analysis.
"""

import json
import argparse
from src.predictor import HoldemPredictor


def display_prediction(title, result):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {title}")
    print("=" * 70)
    print(f"Hole Cards      : {result['hole_cards']}")
    print(f"Community Cards : {result['community_cards']}")
    print(f"Pot Size        : {result['hand_analysis']['pot']} chips")
    print(f"To Call         : {result['hand_analysis']['to_call']} chips")
    print(f"Stack Size      : {result['hand_analysis']['stack']} chips")
    print("-" * 70)
    print("STATISTICAL & STRATEGIC ANALYSIS (via texas_hold_em_utils):")
    analysis = result['hand_analysis']
    print(f"  Expected Win Rate        : {analysis['expected_win_rate']*100:.1f}%")
    print(f"  2-Player Win Rate        : {analysis['expected_2_player_win_rate']*100:.1f}%")
    print(f"  Percentile Rank          : {analysis['percentile_rank']:.1f}th percentile")
    print(f"  Sklansky Hand Group      : Group {analysis['sklansky_group']}")
    print(f"  Ideal Kelly Bet Fraction : {analysis['ideal_kelly_max']*100:.1f}%")
    print(f"  Pot Odds                 : {analysis['pot_odds']*100:.1f}%")
    if analysis['outs_1card'] > 0:
        print(f"  Outs (1-card draw)       : {analysis['outs_1card']} outs")
    print("-" * 70)
    print("MODEL RECOMMENDED DECISION:")
    print(f"  Selected Action          : {result['recommended_action'].upper()} ({result['action_type'].upper()})")
    print(f"  Recommended Chip Bet     : {result['recommended_bet_amount']} chips")
    print("\nACTION PROBABILITY DISTRIBUTION:")
    for action, prob in result['action_probabilities'].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {action:<12} : {prob*100:6.2f}%  [{bar}]")
    print("=" * 70 + "\n")


def run_demo_scenarios(model_path="saved_models/holdem_policy_net.pt"):
    predictor = HoldemPredictor(model_path=model_path)

    scenarios = [
        {
            "title": "1. Preflop Premium Suited Connectors (A-K Suited in Late Position)",
            "hole_cards": ["As", "Ks"],
            "community_cards": [],
            "pot": 50,
            "all_day": 20,
            "round_bet": 0,
            "stack": 1000,
            "position": 4,
            "player_ct": 6
        },
        {
            "title": "2. Flop Nut Flush & Royal Straight Draw (As Ks on Qs Js 2h Board)",
            "hole_cards": ["As", "Ks"],
            "community_cards": ["Qs", "Js", "2h"],
            "pot": 150,
            "all_day": 40,
            "round_bet": 0,
            "stack": 980,
            "position": 2,
            "player_ct": 4
        },
        {
            "title": "3. Turn Full House (Q-Q on Qh Js Jc 5d Board)",
            "hole_cards": ["Qs", "Qd"],
            "community_cards": ["Qh", "Js", "Jc", "5d"],
            "pot": 350,
            "all_day": 80,
            "round_bet": 0,
            "stack": 900,
            "position": 1,
            "player_ct": 3
        },
        {
            "title": "4. Preflop Pocket Tens Short Stack facing a Raise",
            "hole_cards": ["10c", "10d"],
            "community_cards": [],
            "pot": 120,
            "all_day": 80,
            "round_bet": 20,
            "stack": 250,
            "position": 1,
            "player_ct": 6
        },
        {
            "title": "5. River Weak Ace Facing a Large Bet (Ah 5h on Kd 10s 8c 2h Jd)",
            "hole_cards": ["Ah", "5h"],
            "community_cards": ["Kd", "10s", "8c", "2h", "Jd"],
            "pot": 400,
            "all_day": 200,
            "round_bet": 0,
            "stack": 800,
            "position": 0,
            "player_ct": 2
        }
    ]

    for s in scenarios:
        res = predictor.predict_scenario(
            hole_cards=s['hole_cards'],
            community_cards=s['community_cards'],
            pot=s['pot'],
            all_day=s['all_day'],
            round_bet=s['round_bet'],
            stack=s['stack'],
            position=s['position'],
            player_ct=s['player_ct']
        )
        display_prediction(s['title'], res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Texas Hold'em scenarios using ML model")
    parser.add_argument("--model", type=str, default="saved_models/holdem_policy_net.pt", help="Path to model weights")
    args = parser.parse_args()

    run_demo_scenarios(model_path=args.model)
