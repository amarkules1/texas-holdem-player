"""
Tournament Benchmark Script.
Evaluates MLHoldemPlayer against texas_hold_em_utils baseline players:
  - KellyMaxProportionPlayer
  - LimpPlayer
  - AllInPreFlopPlayer
  - SimplePlayer
"""

import os
import torch
import numpy as np
import pandas as pd

from texas_hold_em_utils.game import Game
from texas_hold_em_utils.deck import Deck
from src.model import HoldemPolicyValueNet
from src.player_agent import MLHoldemPlayer
from texas_hold_em_utils.player import SimplePlayer, LimpPlayer, KellyMaxProportionPlayer, AllInPreFlopPlayer


def run_poker_tournament(num_hands=1000, model_path="saved_models/holdem_policy_net.pt"):
    print("==========================================================================")
    print(f"       TEXAS HOLD'EM ML MODEL TOURNAMENT BENCHMARK ({num_hands} HANDS)    ")
    print("==========================================================================")

    model = HoldemPolicyValueNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded trained model weights from {model_path}")
    else:
        print("Using initialized model (untrained/default weights).")
    model.eval()

    ml_player = MLHoldemPlayer(0, chips=1000, model=model, deterministic=True)
    kelly_player = KellyMaxProportionPlayer(1, chips=1000)
    limp_player = LimpPlayer(2, chips=1000)
    allin_player = AllInPreFlopPlayer(3, chips=1000)
    simple_player = SimplePlayer(4, chips=1000)

    players = [ml_player, kelly_player, limp_player, allin_player, simple_player]
    player_names = ["MLHoldemPlayer", "KellyMaxPlayer", "LimpPlayer", "AllInPreFlopPlayer", "SimplePlayer"]

    game = Game(num_players=len(players), big_blind=20, starting_chips=1000)
    game.players = players

    hand_wins = {i: 0 for i in range(len(players))}
    chips_gained = {i: 0.0 for i in range(len(players))}

    for hand in range(1, num_hands + 1):
        # Reset stacks if bankrupt
        for p in game.players:
            if p.chips < 20:
                p.chips = 1000

        start_stacks = [p.chips for p in game.players]

        game.deck = Deck()
        game.deck.shuffle()
        game.community_cards = []
        game.pot = 0
        game.all_day = 0
        game.round = 0
        for p in game.players:
            p.in_round = True
            p.round_bet = 0
            p.hand_of_two.cards = []

        game.dealer_position = (game.dealer_position + 1) % len(game.players)

        try:
            game.run_round()
        except Exception:
            continue

        for i, p in enumerate(game.players):
            delta = p.chips - start_stacks[i]
            chips_gained[i] += delta
            if delta > 0:
                hand_wins[i] += 1

        if hand % 200 == 0 or hand == num_hands:
            print(f"Hand {hand:4d}/{num_hands} completed | ML Stacks: {ml_player.chips} | Kelly: {kelly_player.chips} | Limp: {limp_player.chips}")

    print("\n==========================================================================")
    print("                        TOURNAMENT RESULTS SUMMARY                        ")
    print("==========================================================================")

    results_data = []
    for i in range(len(players)):
        total_chips = chips_gained[i]
        win_rate = (hand_wins[i] / num_hands) * 100.0
        bb_per_100 = (total_chips / 20.0) / (num_hands / 100.0)
        results_data.append({
            "Player": player_names[i],
            "Final Chips": players[i].chips,
            "Total Net Chips": total_chips,
            "Win Rate %": f"{win_rate:.1f}%",
            "BB / 100 Hands": f"{bb_per_100:+.2f}"
        })

    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    print("==========================================================================\n")


if __name__ == "__main__":
    run_poker_tournament(num_hands=500)
