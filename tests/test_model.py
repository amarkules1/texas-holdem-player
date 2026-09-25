"""
Unit tests for HoldemPolicyValueNet and HoldemPredictor.
"""

import numpy as np
import torch
import pytest
from src.model import HoldemPolicyValueNet
from src.predictor import HoldemPredictor


def test_model_forward():
    net = HoldemPolicyValueNet()
    dummy_x = torch.randn(4, 24)
    logits, val = net(dummy_x)

    assert logits.shape == (4, 5)
    assert val.shape == (4,)


def test_model_action_masking():
    net = HoldemPolicyValueNet()
    dummy_x = torch.randn(1, 24)
    # Mask out actions 0, 2, 3, 4 -> only action 1 is valid
    mask = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])

    probs = net.get_action_probs(dummy_x, action_mask=mask)
    probs_np = probs.detach().numpy()[0]

    assert np.isclose(probs_np[1], 1.0, atol=1e-5)
    assert np.isclose(probs_np[0], 0.0, atol=1e-5)


def test_predictor_scenario():
    predictor = HoldemPredictor()
    res = predictor.predict_scenario(
        hole_cards=["Ah", "Kh"],
        community_cards=["Qh", "Jh", "3c"],
        pot=200,
        all_day=50,
        stack=1000,
        position=3,
        player_ct=6
    )

    assert "recommended_action" in res
    assert "action_probabilities" in res
    assert "hand_analysis" in res
    assert len(res["action_probabilities"]) == 5
    probs_sum = sum(res["action_probabilities"].values())
    assert np.isclose(probs_sum, 1.0, atol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
