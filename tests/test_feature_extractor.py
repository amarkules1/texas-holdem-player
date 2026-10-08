"""
Unit tests for FeatureExtractor module with expanded dimensions.
"""

import numpy as np
import pytest
from src.feature_extractor import FeatureExtractor, parse_card
from texas_hold_em_utils.card import Card


def test_parse_card():
    c1 = parse_card("As")
    assert c1.get_rank_str() == "A"
    assert c1.get_suit_str() == "Spades"

    c2 = parse_card("10c")
    assert c2.get_rank_str() == "10"
    assert c2.get_suit_str() == "Clubs"

    c3 = parse_card(Card().from_ints(11, 0))
    assert c3.get_rank_str() == "K"


def test_extract_features_preflop():
    feats = FeatureExtractor.extract_features(
        hole_cards=["As", "Ks"],
        community_cards=[],
        round_num=0,
        pot=40,
        all_day=20,
        round_bet=0,
        chips=1000,
        big_blind=20,
        position=2,
        player_ct=6
    )

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (28,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Preflop AKs should have high win rate and percentile
    win_rate = feats[0]
    percentile = feats[2]
    assert win_rate > 0.25
    assert percentile > 0.90

    # Check player count dimensions
    player_ct_norm = feats[24]
    assert np.isclose(player_ct_norm, (6 - 2) / 8.0)
    assert feats[27] == 1.0  # is_short_handed for 6 players


def test_extract_features_varying_player_counts():
    # 2 players (heads up)
    feats_2p = FeatureExtractor.extract_features(
        hole_cards=["Ah", "Kh"],
        community_cards=[],
        player_ct=2
    )
    assert feats_2p.shape == (28,)
    assert feats_2p[24] == 0.0  # (2-2)/8
    assert feats_2p[26] == 1.0  # is_heads_up

    # 10 players (full table)
    feats_10p = FeatureExtractor.extract_features(
        hole_cards=["Ah", "Kh"],
        community_cards=[],
        player_ct=10
    )
    assert feats_10p.shape == (28,)
    assert feats_10p[24] == 1.0  # (10-2)/8
    assert feats_10p[26] == 0.0  # not heads up
    assert feats_10p[27] == 0.0  # not short-handed (> 6)


def test_extract_features_flop():
    feats = FeatureExtractor.extract_features(
        hole_cards=["As", "Ks"],
        community_cards=["Qs", "Js", "2h"],
        round_num=1,
        pot=120,
        all_day=30,
        round_bet=0,
        chips=970,
        big_blind=20,
        position=1,
        player_ct=4
    )

    assert feats.shape == (28,)
    assert not np.isnan(feats).any()
    outs_1card = feats[5] * 47.0
    assert outs_1card > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
