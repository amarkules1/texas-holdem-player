"""
HoldemPredictor - High Level API for Texas Hold'em Strategy Prediction.
Provides move recommendations and probability distributions for any poker scenario.
"""

import os
import torch
from texas_hold_em_utils.card import Card
from .feature_extractor import FeatureExtractor, parse_card
from .model import HoldemPolicyValueNet


class HoldemPredictor:
    """
    High-level interface to evaluate Texas Hold'em scenarios using the trained ML model.
    """

    def __init__(self, model_path=None):
        self.model = HoldemPolicyValueNet()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Loaded trained weights from {model_path}")
        self.model.eval()

    def predict_scenario(
        self,
        hole_cards,
        community_cards=None,
        pot=100,
        all_day=20,
        round_bet=0,
        stack=1000,
        position=0,
        player_ct=6,
        big_blind=20,
        temperature=1.0
    ):
        """
        Evaluates a game scenario and returns move recommendations and probabilities.

        :param hole_cards: List of 2 hole cards e.g. ["As", "Ks"] or ["A of Spades", "K of Spades"]
        :param community_cards: List of 0 to 5 community cards e.g. ["Qs", "Js", "2h"]
        :param pot: Current total pot size in chips (default 100)
        :param all_day: Highest bet in current hand across all rounds (default 20)
        :param round_bet: Player's existing bet in current round (default 0)
        :param stack: Player's remaining stack size in chips (default 1000)
        :param position: Player's seat position 0..player_ct-1 (default 0)
        :param player_ct: Total players at table (default 6)
        :param big_blind: Big blind chip size (default 20)
        :param temperature: Softmax temperature for probability smoothing
        :return: dict with action recommendation, probability distribution, and hand analysis
        """
        if community_cards is None:
            community_cards = []

        h_cards = [parse_card(c) for c in hole_cards]
        c_cards = [parse_card(c) for c in community_cards]

        round_num = 0
        if len(c_cards) == 3:
            round_num = 1
        elif len(c_cards) == 4:
            round_num = 2
        elif len(c_cards) == 5:
            round_num = 3

        to_call = max(0, all_day - round_bet)

        # 1. Extract features using texas_hold_em_utils
        features = FeatureExtractor.extract_features(
            hole_cards=h_cards,
            community_cards=c_cards,
            round_num=round_num,
            pot=pot,
            all_day=all_day,
            round_bet=round_bet,
            chips=stack,
            big_blind=big_blind,
            position=position,
            player_ct=player_ct
        )
        state_tensor = torch.tensor(features, dtype=torch.float32)

        # 2. Build action mask
        mask = [1.0, 1.0, 1.0, 1.0, 1.0]
        if to_call == 0:
            mask[0] = 0.0  # Cannot fold if checking is free
        if stack <= to_call:
            mask[2] = 0.0  # Cannot raise small
            mask[3] = 0.0  # Cannot raise med
            mask[4] = 0.0  # Cannot raise all-in
        if stack == 0:
            mask = [0.0, 1.0, 0.0, 0.0, 0.0]
        action_mask = torch.tensor(mask, dtype=torch.float32)

        # 3. Network Prediction
        action_idx, action_name, prob_dict = self.model.predict_action(
            state_tensor,
            action_mask=action_mask,
            deterministic=True,
            temperature=temperature
        )

        with torch.no_grad():
            _, value_est = self.model(state_tensor)
            state_val = float(value_est.item())

        # 4. Determine bet amount and game action
        if action_idx == 0:
            action_type = "fold"
            bet_amt = 0
        elif action_idx == 1:
            if to_call == 0:
                action_type = "check"
                bet_amt = 0
            else:
                action_type = "call"
                bet_amt = min(to_call, stack)
        elif action_idx == 2:
            action_type = "raise"
            raise_inc = max(big_blind, pot // 2)
            target_total = all_day + raise_inc
            needed_chips = target_total - round_bet
            bet_amt = min(stack, needed_chips)
        elif action_idx == 3:
            action_type = "raise"
            raise_inc = max(big_blind * 2, pot)
            target_total = all_day + raise_inc
            needed_chips = target_total - round_bet
            bet_amt = min(stack, needed_chips)
        elif action_idx == 4:
            action_type = "raise"
            bet_amt = stack

        # 5. Hand Analysis Breakdown
        win_rate = float(features[0])
        win_rate_2p = float(features[1])
        percentile = float(features[2]) * 100.0
        ideal_kelly = float(features[3])
        sklansky_grp = int(round(features[4] * 9.0))
        outs_1card = int(round(features[5] * 47.0))
        pot_odds = float(features[7])

        return {
            "hole_cards": [str(c) for c in h_cards],
            "community_cards": [str(c) for c in c_cards],
            "recommended_action": action_name,
            "action_type": action_type,
            "recommended_bet_amount": bet_amt,
            "action_probabilities": prob_dict,
            "state_value_estimate": state_val,
            "hand_analysis": {
                "expected_win_rate": round(win_rate, 4),
                "expected_2_player_win_rate": round(win_rate_2p, 4),
                "percentile_rank": round(percentile, 2),
                "ideal_kelly_max": round(ideal_kelly, 4),
                "sklansky_group": sklansky_grp,
                "outs_1card": outs_1card,
                "pot_odds": round(pot_odds, 4),
                "to_call": to_call,
                "pot": pot,
                "stack": stack
            }
        }
