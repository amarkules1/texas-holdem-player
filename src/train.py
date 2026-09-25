"""
Training pipeline for Texas Hold'em ML Player.
Phase 1: GTO Distillation Warm-Start (Supervised Learning on Game Theory Priors).
Phase 2: Reinforcement Learning Self-Play (Advantage Actor-Critic) using texas_hold_em_utils.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from texas_hold_em_utils.game import Game
from texas_hold_em_utils.deck import Deck
from texas_hold_em_utils.card import Card
from texas_hold_em_utils.player import SimplePlayer, LimpPlayer, KellyMaxProportionPlayer, AllInPreFlopPlayer

from .feature_extractor import FeatureExtractor
from .model import HoldemPolicyValueNet
from .player_agent import MLHoldemPlayer


def generate_gto_prior_dataset(num_samples=5000):
    """
    Generates synthetic decision states and target GTO probabilistic move distributions.
    """
    print(f"Generating {num_samples} GTO prior decision states for warm-start training...")
    X_list = []
    y_prob_list = []
    y_val_list = []
    mask_list = []

    deck = Deck()
    ranks = Card.ranks
    suits = Card.suits

    for i in range(num_samples):
        # Randomize game scenario parameters
        player_ct = random.randint(2, 6)
        position = random.randint(0, player_ct - 1)
        round_num = random.choice([0, 1, 2, 3])
        big_blind = 20.0
        stack = float(random.choice([200, 500, 1000, 2000]))

        if round_num == 0:
            pot = float(random.choice([30, 40, 60, 100]))
            all_day = float(random.choice([20, 40, 60, 100]))
            comm_cards = []
        elif round_num == 1:
            pot = float(random.choice([60, 100, 200, 400]))
            all_day = float(random.choice([0, 20, 50, 100]))
            deck_copy = Deck()
            deck_copy.shuffle()
            comm_cards = [deck_copy.draw() for _ in range(3)]
        elif round_num == 2:
            pot = float(random.choice([100, 200, 500, 800]))
            all_day = float(random.choice([0, 40, 100, 200]))
            deck_copy = Deck()
            deck_copy.shuffle()
            comm_cards = [deck_copy.draw() for _ in range(4)]
        else:
            pot = float(random.choice([200, 500, 1000, 1500]))
            all_day = float(random.choice([0, 50, 200, 500]))
            deck_copy = Deck()
            deck_copy.shuffle()
            comm_cards = [deck_copy.draw() for _ in range(5)]

        round_bet = 0.0 if all_day == 0 else float(random.choice([0, 20, 40]))
        to_call = max(0.0, all_day - round_bet)

        # Draw hole cards avoiding community cards
        deck_copy = Deck()
        for c in comm_cards:
            deck_copy.remove(c)
        deck_copy.shuffle()
        hole_cards = [deck_copy.draw(), deck_copy.draw()]

        # Extract features fast (use small sample size for fast generation)
        feats = FeatureExtractor.extract_features(
            hole_cards=hole_cards,
            community_cards=comm_cards,
            round_num=round_num,
            pot=pot,
            all_day=all_day,
            round_bet=round_bet,
            chips=stack,
            big_blind=big_blind,
            position=position,
            player_ct=player_ct,
            sample_size=100
        )

        win_rate = feats[0]
        percentile = feats[2]
        kelly_max = feats[3]
        pot_odds = feats[7]
        outs = feats[5] * 47.0

        # Compute GTO-aligned target probabilities: [fold, check_call, raise_small, raise_med, raise_allin]
        target_p = np.zeros(5, dtype=np.float32)

        if to_call == 0:
            # Free to check
            if percentile > 0.85 or kelly_max > 0.35:
                target_p[1] = 0.15  # check
                target_p[2] = 0.50  # raise small
                target_p[3] = 0.25  # raise med
                target_p[4] = 0.10  # raise all-in
            elif percentile > 0.60 or kelly_max > 0.15:
                target_p[1] = 0.55  # check
                target_p[2] = 0.35  # raise small
                target_p[3] = 0.10  # raise med
                target_p[4] = 0.00
            else:
                # Weak hand - check (bluff occasionally in late position)
                rel_pos = position / max(1, player_ct - 1)
                bluff_prob = 0.15 if rel_pos > 0.6 else 0.05
                target_p[1] = 1.0 - bluff_prob
                target_p[2] = bluff_prob
        else:
            # Facing a bet
            if percentile > 0.90 or kelly_max > 0.40:
                # Premium monster
                target_p[0] = 0.00  # fold
                target_p[1] = 0.20  # call/slowplay
                target_p[2] = 0.35  # raise small
                target_p[3] = 0.30  # raise med
                target_p[4] = 0.15  # raise all-in
            elif percentile > 0.70 or (win_rate > pot_odds + 0.10):
                # Strong hand
                target_p[0] = 0.05
                target_p[1] = 0.55
                target_p[2] = 0.30
                target_p[3] = 0.10
            elif win_rate >= pot_odds or (outs * 0.02 >= pot_odds):
                # Pot odds call / draw
                target_p[0] = 0.20
                target_p[1] = 0.70
                target_p[2] = 0.10
            else:
                # Weak hand facing bet -> Fold dominant
                target_p[0] = 0.85
                target_p[1] = 0.15

        # Mask logic
        mask = [1.0, 1.0, 1.0, 1.0, 1.0]
        if to_call == 0:
            mask[0] = 0.0
            target_p[0] = 0.0
        if stack <= to_call:
            mask[2] = 0.0
            mask[3] = 0.0
            mask[4] = 0.0
            target_p[2] = 0.0
            target_p[3] = 0.0
            target_p[4] = 0.0

        p_sum = target_p.sum()
        if p_sum > 0:
            target_p = target_p / p_sum
        else:
            target_p[1] = 1.0

        # Estimated target expected value (in BB)
        target_val = (win_rate * pot - (1.0 - win_rate) * to_call) / big_blind

        X_list.append(feats)
        y_prob_list.append(target_p)
        y_val_list.append(target_val)
        mask_list.append(mask)

    X = np.array(X_list, dtype=np.float32)
    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    masks = np.array(mask_list, dtype=np.float32)

    return X, y_prob, y_val, masks


def pretrain_gto_prior(model, num_samples=3000, epochs=15, batch_size=64, lr=1e-3):
    """
    Phase 1: Supervised GTO Prior Distillation Warm-Start.
    """
    print("=== Phase 1: Pre-training GTO Strategy Priors ===")
    X, y_prob, y_val, masks = generate_gto_prior_dataset(num_samples=num_samples)

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y_prob),
        torch.tensor(y_val),
        torch.tensor(masks)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for epoch in range(1, epochs + 1):
        total_p_loss = 0.0
        total_v_loss = 0.0
        batches = 0

        for b_x, b_y_prob, b_y_val, b_mask in dataloader:
            optimizer.zero_grad()
            logits, val_pred = model(b_x)

            # Masked Softmax Log-Probs
            masked_logits = logits.masked_fill(b_mask == 0, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            # KL Divergence / Cross Entropy for policy loss
            p_loss = -torch.mean(torch.sum(b_y_prob * log_probs, dim=-1))
            v_loss = nn.MSELoss()(val_pred, b_y_val)

            loss = p_loss + 0.5 * v_loss
            loss.backward()
            optimizer.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            batches += 1

        avg_p = total_p_loss / batches
        avg_v = total_v_loss / batches
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Policy Loss: {avg_p:.4f} | Value Loss: {avg_v:.4f}")

    print("Pre-training completed successfully!")


def train_self_play_rl(model, num_hands=500, lr=3e-4, save_path="saved_models/holdem_policy_net.pt"):
    """
    Phase 2: Reinforcement Learning Self-Play training using texas_hold_em_utils Game.
    """
    print(f"\n=== Phase 2: Self-Play RL Training ({num_hands} hands) ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize players
    ml_player_1 = MLHoldemPlayer(0, chips=1000, model=model, deterministic=False, temperature=1.0)
    ml_player_2 = MLHoldemPlayer(1, chips=1000, model=model, deterministic=False, temperature=1.0)
    op_kelly = KellyMaxProportionPlayer(2, chips=1000)
    op_limp = LimpPlayer(3, chips=1000)
    op_simple = SimplePlayer(4, chips=1000)

    game = Game(num_players=5, big_blind=20, starting_chips=1000)
    game.players = [ml_player_1, ml_player_2, op_kelly, op_limp, op_simple]

    hands_completed = 0
    ml_total_payoff = 0.0

    for hand_idx in range(num_hands):
        # Reset chips if stack depleted
        for p in game.players:
            if p.chips < 100:
                p.chips = 1000

        start_chips_1 = ml_player_1.chips
        start_chips_2 = ml_player_2.chips

        # Prepare deck and round state
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

        # Run hand round
        try:
            game.run_round()
        except Exception as e:
            continue

        gain_1 = (ml_player_1.chips - start_chips_1) / game.big_blind
        gain_2 = (ml_player_2.chips - start_chips_2) / game.big_blind
        ml_total_payoff += (gain_1 + gain_2)
        hands_completed += 1

        if hands_completed % 100 == 0:
            avg_bb = ml_total_payoff / (hands_completed * 2)
            print(f"Hand {hands_completed}/{num_hands} | ML Avg Payoff: {avg_bb:+.2f} BB/hand | Stack P1: {ml_player_1.chips} | P2: {ml_player_2.chips}")

    # Save final model state dict
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved successfully to {save_path}")


def main():
    model = HoldemPolicyValueNet()
    pretrain_gto_prior(model, num_samples=3000, epochs=15)
    train_self_play_rl(model, num_hands=300)


if __name__ == "__main__":
    main()
