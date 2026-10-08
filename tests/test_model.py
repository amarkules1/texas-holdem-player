"""
Unit tests for HoldemPolicyValueNet and HoldemPredictor with expanded dimensions.
"""

import numpy as np
import torch
import pytest
from src.model import HoldemPolicyValueNet
from src.predictor import HoldemPredictor


def test_model_forward_expanded_dimensions():
    # New default: input_dim=28, hidden_dim=256
    net = HoldemPolicyValueNet(input_dim=28, hidden_dim=256)
    dummy_x = torch.randn(4, 28)
    logits, val = net(dummy_x)

    assert logits.shape == (4, 5)
    assert val.shape == (4,)


def test_model_backward_compatibility_dimension_adaptation():
    # If 24-dim tensor passed to 28-dim model, it should auto-pad without crashing
    net_28 = HoldemPolicyValueNet(input_dim=28, hidden_dim=256)
    x_24 = torch.randn(2, 24)
    logits, val = net_28(x_24)
    assert logits.shape == (2, 5)

    # If 28-dim tensor passed to 24-dim model, it should auto-slice without crashing
    net_24 = HoldemPolicyValueNet(input_dim=24, hidden_dim=128)
    x_28 = torch.randn(2, 28)
    logits, val = net_24(x_28)
    assert logits.shape == (2, 5)


def test_model_action_masking():
    net = HoldemPolicyValueNet()
    dummy_x = torch.randn(1, 28)
    # Mask out actions 0, 2, 3, 4 -> only action 1 is valid
    mask = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])

    probs = net.get_action_probs(dummy_x, action_mask=mask)
    probs_np = probs.detach().numpy()[0]

    assert np.isclose(probs_np[1], 1.0, atol=1e-5)
    assert np.isclose(probs_np[0], 0.0, atol=1e-5)


def test_predictor_scenario_varying_players():
    predictor = HoldemPredictor()

    for p_count in [2, 4, 6, 9, 10]:
        res = predictor.predict_scenario(
            hole_cards=["Ah", "Kh"],
            community_cards=["Qh", "Jh", "3c"],
            pot=200,
            all_day=50,
            stack=1000,
            position=min(p_count - 1, 3),
            player_ct=p_count
        )

        assert "recommended_action" in res
        assert "action_probabilities" in res
        assert "hand_analysis" in res
        assert len(res["action_probabilities"]) == 5
        probs_sum = sum(res["action_probabilities"].values())
        assert np.isclose(probs_sum, 1.0, atol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
