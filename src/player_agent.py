"""
MLHoldemPlayer - Machine Learning Powered Texas Hold'em Player Agent.
Subclasses texas_hold_em_utils.player.Player and uses HoldemPolicyValueNet for move decisions.
"""

import numpy as np
import torch
from texas_hold_em_utils.player import Player
from .feature_extractor import FeatureExtractor
from .model import HoldemPolicyValueNet


class MLHoldemPlayer(Player):
    """
    ML Agent Player that evaluates game state using FeatureExtractor
    and selects optimal moves using HoldemPolicyValueNet.
    """

    def __init__(self, position, chips=1000, model=None, deterministic=False, temperature=1.0):
        super().__init__(position, chips)
        self.deterministic = deterministic
        self.temperature = temperature
        
        if model is None:
            self.model = HoldemPolicyValueNet()
        else:
            self.model = model
            
        self.model.eval()

    def get_action_mask(self, to_call, chips):
        """
        Creates a valid action mask [fold, check_call, raise_small, raise_med, raise_allin].
        1 = valid, 0 = invalid move.
        """
        mask = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # 1. If to_call == 0 (free to check), folding is strictly dominated (mask fold out)
        if to_call == 0:
            mask[0] = 0.0
            
        # 2. If remaining chips are less than or equal to to_call, player cannot raise
        if chips <= to_call:
            mask[2] = 0.0
            mask[3] = 0.0
            mask[4] = 0.0
            
        # 3. If chips == 0 (already all-in), only check_call is possible
        if chips == 0:
            mask = [0.0, 1.0, 0.0, 0.0, 0.0]
            
        return torch.tensor(mask, dtype=torch.float32)

    def decide(self, round_num, pot, all_day, big_blind, community_cards, player_ct):
        """
        Calculates move probabilities and selects action according to ML Policy model.

        :param round_num: 0 for pre-flop, 1 for flop, 2 for turn, 3 for river
        :param pot: the current pot
        :param all_day: the current highest bet (including all rounds)
        :param big_blind: the big blind for the game
        :param community_cards: the community cards (list of 0 to 5 cards)
        :param player_ct: number of players in the game
        :return: a tuple of action string ("fold", "check", "call", "raise") and bet amount
        """
        to_call = max(0, all_day - self.round_bet)

        # 1. Extract 24-dim features using texas_hold_em_utils
        features = FeatureExtractor.extract_features(
            hole_cards=self.hand_of_two.cards,
            community_cards=community_cards,
            round_num=round_num,
            pot=pot,
            all_day=all_day,
            round_bet=self.round_bet,
            chips=self.chips,
            big_blind=big_blind,
            position=self.position,
            player_ct=player_ct
        )
        state_tensor = torch.tensor(features, dtype=torch.float32)
        action_mask = self.get_action_mask(to_call, self.chips)

        # 2. Predict action and probabilities using Neural Network
        action_idx, action_name, prob_dict = self.model.predict_action(
            state_tensor,
            action_mask=action_mask,
            deterministic=self.deterministic,
            temperature=self.temperature
        )

        # 3. Execute chosen action into game chip amounts
        if action_idx == 0:  # FOLD
            if to_call == 0:
                return "check", 0
            return "fold", self.fold()

        elif action_idx == 1:  # CHECK / CALL
            if to_call == 0:
                return "check", 0
            else:
                call_amt = min(to_call, self.chips)
                return "call", self.bet(call_amt)

        elif action_idx == 2:  # RAISE SMALL (0.5x Pot / Min-raise)
            raise_inc = max(big_blind, pot // 2)
            target_total = all_day + raise_inc
            needed_chips = target_total - self.round_bet
            actual_raise = min(self.chips, needed_chips)
            if actual_raise <= to_call:
                call_amt = min(to_call, self.chips)
                return "call", self.bet(call_amt)
            return "raise", self.bet(actual_raise)

        elif action_idx == 3:  # RAISE MEDIUM (1.0x Pot)
            raise_inc = max(big_blind * 2, pot)
            target_total = all_day + raise_inc
            needed_chips = target_total - self.round_bet
            actual_raise = min(self.chips, needed_chips)
            if actual_raise <= to_call:
                call_amt = min(to_call, self.chips)
                return "call", self.bet(call_amt)
            return "raise", self.bet(actual_raise)

        elif action_idx == 4:  # RAISE LARGE / ALL-IN
            actual_raise = self.chips
            if actual_raise <= to_call:
                call_amt = min(to_call, self.chips)
                return "call", self.bet(call_amt)
            return "raise", self.bet(actual_raise)

        return "check", 0

    def get_move_probabilities(self, round_num, pot, all_day, big_blind, community_cards, player_ct):
        """
        Public method to query the exact move probabilities for the current state.
        :return: dict mapping action names to probabilities.
        """
        to_call = max(0, all_day - self.round_bet)
        features = FeatureExtractor.extract_features(
            hole_cards=self.hand_of_two.cards,
            community_cards=community_cards,
            round_num=round_num,
            pot=pot,
            all_day=all_day,
            round_bet=self.round_bet,
            chips=self.chips,
            big_blind=big_blind,
            position=self.position,
            player_ct=player_ct
        )
        state_tensor = torch.tensor(features, dtype=torch.float32)
        action_mask = self.get_action_mask(to_call, self.chips)

        _, _, prob_dict = self.model.predict_action(
            state_tensor,
            action_mask=action_mask,
            deterministic=True
        )
        return prob_dict
