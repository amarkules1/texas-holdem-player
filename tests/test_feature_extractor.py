"""
Unit tests for FeatureExtractor module.
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
    assert feats.shape == (24,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Preflop AKs should have high win rate and percentile
    win_rate = feats[0]
    percentile = feats[2]
    assert win_rate > 0.25
    assert percentile > 0.90


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

    assert feats.shape == (24,)
    assert not np.isnan(feats).any()
    outs_1card = feats[5] * 47.0
    assert outs_1card > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
