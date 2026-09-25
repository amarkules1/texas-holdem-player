"""
Feature Extractor for Texas Hold'em Game States
Uses texas_hold_em_utils to compute rich statistical, probabilistic, and strategic features.
"""

import numpy as np
from texas_hold_em_utils.card import Card
from texas_hold_em_utils.hands import HandOfFive, HandOfTwo
from texas_hold_em_utils.relative_ranking import get_hand_rank_details
from texas_hold_em_utils.outs_counter import get_one_card_outs, get_two_card_outs
from texas_hold_em_utils.sklansky import sklansky_rank


def parse_card(card):
    """Converts string representations or Card objects to Card objects."""
    if isinstance(card, Card):
        return card
    c_str = str(card).strip()
    if ' of ' in c_str:
        return Card().from_name(c_str)
    
    suit_map = {'s': 'Spades', 'h': 'Hearts', 'd': 'Diamonds', 'c': 'Clubs'}
    rank_map = {'T': '10'}
    
    suit_char = c_str[-1].lower()
    rank_str = c_str[:-1].upper()
    rank_str = rank_map.get(rank_str, rank_str)
    suit_str = suit_map.get(suit_char, suit_char)
    
    return Card().from_str(rank_str, suit_str)


class FeatureExtractor:
    """
    Extracts a 24-dimensional normalized feature vector from any Texas Hold'em game state.
    """
    FEATURE_DIM = 24
    FEATURE_NAMES = [
        "win_rate", "win_rate_2p", "percentile", "ideal_kelly_max",
        "sklansky_rank_norm", "outs_1card_norm", "outs_2card_norm",
        "pot_odds", "spr", "bet_to_stack_ratio",
        "call_in_bb", "stack_in_bb", "pot_in_bb",
        "relative_position", "active_player_ratio",
        "is_preflop", "is_flop", "is_turn", "is_river",
        "is_suited", "is_pair",
        "high_card_rank_norm", "low_card_rank_norm", "hand_category_norm"
    ]

    @staticmethod
    def extract_features(
        hole_cards,
        community_cards=None,
        round_num=0,
        pot=20.0,
        all_day=20.0,
        round_bet=0.0,
        chips=1000.0,
        big_blind=20.0,
        position=0,
        player_ct=6,
        active_players=None,
        sample_size=200
    ):
        """
        Extracts features for a player's decision point.

        :param hole_cards: List of 2 cards (Card objects or str representations)
        :param community_cards: List of 0-5 community cards (Card objects or str representations)
        :param round_num: 0 (preflop), 1 (flop), 2 (turn), 3 (river)
        :param pot: Total current pot size in chips
        :param all_day: Highest bet in current hand across rounds
        :param round_bet: Amount already bet by player in current round
        :param chips: Player's remaining stack size
        :param big_blind: Big blind size
        :param position: Player's seat position (0 to player_ct - 1)
        :param player_ct: Total number of players at the table
        :param active_players: Number of players currently active in the round
        :param sample_size: Monte Carlo sample size for postflop estimation
        :return: numpy array of shape (24,), dtype float32
        """
        if community_cards is None:
            community_cards = []
        if active_players is None:
            active_players = player_ct

        # Parse cards
        h_cards = [parse_card(c) for c in hole_cards]
        c_cards = [parse_card(c) for c in community_cards]

        to_call = max(0.0, float(all_day - round_bet))
        pot = max(1.0, float(pot))
        chips = max(0.0, float(chips))
        big_blind = max(1.0, float(big_blind))

        # 1. Hand Rank Details from texas_hold_em_utils
        try:
            comm_arg = c_cards if len(c_cards) >= 3 else None
            rank_details = get_hand_rank_details(
                h_cards,
                community_cards=comm_arg,
                player_count=max(2, player_ct),
                sample_size=sample_size
            )
            win_rate = float(rank_details.get("expected_win_rate", 0.2))
            win_rate_2p = float(rank_details.get("expected_2_player_win_rate", 0.5))
            percentile = float(rank_details.get("percentile", 50.0)) / 100.0
            ideal_kelly_max = float(rank_details.get("ideal_kelly_max", 0.0))
        except Exception:
            win_rate = 1.0 / max(2, player_ct)
            win_rate_2p = 0.5
            percentile = 0.5
            ideal_kelly_max = 0.0

        # 2. Sklansky Rank
        try:
            sk_rank = sklansky_rank(h_cards[0], h_cards[1])
            sklansky_rank_norm = float(sk_rank) / 9.0
        except Exception:
            sklansky_rank_norm = 0.5

        # 3. Outs Calculation
        outs_1card_norm = 0.0
        outs_2card_norm = 0.0
        if len(c_cards) in [3, 4]:
            try:
                outs_1 = get_one_card_outs(h_cards, c_cards)
                outs_1card_norm = min(1.0, len(outs_1) / 47.0)
            except Exception:
                outs_1card_norm = 0.0
        if len(c_cards) == 3:
            try:
                outs_2 = get_two_card_outs(h_cards, c_cards)
                outs_2card_norm = min(1.0, len(outs_2) / 47.0)
            except Exception:
                outs_2card_norm = 0.0

        # 4. Pot & Stack Metrics
        pot_odds = to_call / (pot + to_call + 1e-5)
        spr = float(np.tanh(chips / (pot + 1.0) / 10.0))
        bet_to_stack_ratio = min(1.0, to_call / (chips + 1e-5))
        call_in_bb = float(np.tanh(to_call / big_blind / 10.0))
        stack_in_bb = float(np.tanh(chips / big_blind / 100.0))
        pot_in_bb = float(np.tanh(pot / big_blind / 50.0))

        # 5. Table Position Metrics
        rel_pos = float(position) / max(1.0, float(player_ct - 1))
        act_ratio = float(active_players) / max(1.0, float(player_ct))

        # 6. Round One-Hot
        is_preflop = 1.0 if round_num == 0 else 0.0
        is_flop = 1.0 if round_num == 1 else 0.0
        is_turn = 1.0 if round_num == 2 else 0.0
        is_river = 1.0 if round_num == 3 else 0.0

        # 7. Card Specific Properties
        r1, r2 = h_cards[0].rank, h_cards[1].rank
        is_suited = 1.0 if h_cards[0].suit == h_cards[1].suit else 0.0
        is_pair = 1.0 if r1 == r2 else 0.0
        high_card_rank_norm = max(r1, r2) / 12.0
        low_card_rank_norm = min(r1, r2) / 12.0

        # 8. Made Hand Category (0.0 to 1.0)
        hand_category_norm = 0.0
        if len(c_cards) == 5:
            try:
                h5 = HandOfFive(h_cards, c_cards)
                hand_category_norm = float(h5.hand_rank) / 9.0
            except Exception:
                hand_category_norm = 0.0

        features = np.array([
            win_rate,
            win_rate_2p,
            percentile,
            ideal_kelly_max,
            sklansky_rank_norm,
            outs_1card_norm,
            outs_2card_norm,
            pot_odds,
            spr,
            bet_to_stack_ratio,
            call_in_bb,
            stack_in_bb,
            pot_in_bb,
            rel_pos,
            act_ratio,
            is_preflop,
            is_flop,
            is_turn,
            is_river,
            is_suited,
            is_pair,
            high_card_rank_norm,
            low_card_rank_norm,
            hand_category_norm
        ], dtype=np.float32)

        return features
