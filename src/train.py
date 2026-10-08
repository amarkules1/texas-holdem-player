"""
Training pipeline for Texas Hold'em ML Player.
- Phase 1: Pretraining against an opponent pool consisting of 80% existing saved model
           and 20% sprinkled baseline players across 2-10 player tables.
- Phase 2: Reinforcement Learning Self-Play (4000 hands) with Advantage Actor-Critic (A2C).
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from texas_hold_em_utils.game import Game
from texas_hold_em_utils.deck import Deck
from texas_hold_em_utils.card import Card
from texas_hold_em_utils.player import SimplePlayer, LimpPlayer, KellyMaxProportionPlayer, AllInPreFlopPlayer

from .feature_extractor import FeatureExtractor
from .model import HoldemPolicyValueNet
from .player_agent import MLHoldemPlayer


def sample_player_count():
    """
    Samples total number of players at the table from 2 to 10.
    Lower numbers of players are more common (e.g. heads-up and short-handed tables).
    """
    counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Decaying weights giving higher probability to lower player counts
    weights = [0.24, 0.20, 0.16, 0.13, 0.10, 0.07, 0.04, 0.03, 0.03]
    return random.choices(counts, weights=weights, k=1)[0]


def get_base_model(base_model_path="saved_models/holdem_policy_net_base.pt"):
    """
    Loads the existing saved model to serve as the 80% opponent teacher.
    """
    fallback_path = "saved_models/holdem_policy_net.pt"
    target_path = base_model_path if os.path.exists(base_model_path) else fallback_path

    if os.path.exists(target_path):
        base_model = HoldemPolicyValueNet.from_checkpoint(target_path)
        print(f"Loaded existing saved model from {target_path} (input_dim={base_model.input_dim}, hidden_dim={base_model.hidden_dim})")
    else:
        print("Existing saved model not found; initializing standard 24-dim base model.")
        base_model = HoldemPolicyValueNet(input_dim=24, hidden_dim=128)

    base_model.eval()
    return base_model


def create_pretraining_opponents(num_opponents, total_players, base_model, sample_size=30):
    """
    Creates an opponent roster where:
    - 80% are the existing saved model (MLHoldemPlayer with base_model)
    - 20% are other baseline players (KellyMax, Limp, Simple, AllInPreFlop) sprinkled in.
    """
    opponents = []
    other_classes = [KellyMaxProportionPlayer, LimpPlayer, SimplePlayer, AllInPreFlopPlayer]

    for seat_idx in range(1, num_opponents + 1):
        if random.random() < 0.80:
            # 80% existing saved model
            opp = MLHoldemPlayer(
                position=seat_idx,
                chips=1000,
                model=base_model,
                deterministic=False,
                temperature=1.0,
                total_players=total_players,
                sample_size=sample_size
            )
        else:
            # 20% other players sprinkled in
            chosen_cls = random.choice(other_classes)
            opp = chosen_cls(position=seat_idx, chips=1000)

        opponents.append(opp)

    return opponents


def generate_pretraining_dataset(num_samples=3000, base_model=None):
    """
    Generates synthetic decision states across tables with 2 to 10 players.
    Target action probabilities and values are guided 80% by the existing saved model
    and 20% by game-theoretic baseline strategies.
    """
    print(f"\n[Phase 1A] Synthesizing {num_samples} diverse pretraining decision states...")
    print("           Player count varies from 2-10 (lower counts more common).")
    print("           Teacher targets: 80% existing saved model, 20% sprinkled strategic baselines.")

    X_list = []
    y_prob_list = []
    y_val_list = []
    mask_list = []

    t0 = time.time()
    for i in range(num_samples):
        player_ct = sample_player_count()
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

        # Draw hole cards
        deck_copy = Deck()
        for c in comm_cards:
            deck_copy.remove(c)
        deck_copy.shuffle()
        hole_cards = [deck_copy.draw(), deck_copy.draw()]

        # Active players (between 2 and player_ct)
        active_players = random.randint(2, player_ct) if player_ct > 2 else 2

        # Extract 28-dimensional features
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
            active_players=active_players,
            sample_size=30
        )

        # Action mask
        mask = [1.0, 1.0, 1.0, 1.0, 1.0]
        if to_call == 0:
            mask[0] = 0.0
        if stack <= to_call:
            mask[2] = 0.0
            mask[3] = 0.0
            mask[4] = 0.0
        mask_t = torch.tensor(mask, dtype=torch.float32)

        # Decide teacher source: 80% existing saved model, 20% strategic heuristics
        use_base_model = (random.random() < 0.80) and (base_model is not None)

        if use_base_model:
            base_in_dim = getattr(base_model, "input_dim", 24)
            feats_t = torch.tensor(feats[:base_in_dim], dtype=torch.float32)
            with torch.no_grad():
                probs_t = base_model.get_action_probs(feats_t, action_mask=mask_t, temperature=1.0)
                _, val_est = base_model(feats_t)
                target_p = probs_t[0].cpu().numpy()
                target_val = float(val_est.item())
        else:
            win_rate = feats[0]
            percentile = feats[2]
            kelly_max = feats[3]
            pot_odds = feats[7]
            outs = feats[5] * 47.0

            target_p = np.zeros(5, dtype=np.float32)
            if to_call == 0:
                if percentile > 0.85 or kelly_max > 0.35:
                    target_p[1], target_p[2], target_p[3], target_p[4] = 0.15, 0.50, 0.25, 0.10
                elif percentile > 0.60 or kelly_max > 0.15:
                    target_p[1], target_p[2], target_p[3], target_p[4] = 0.55, 0.35, 0.10, 0.00
                else:
                    rel_pos = position / max(1, player_ct - 1)
                    bluff_prob = 0.15 if rel_pos > 0.6 else 0.05
                    target_p[1] = 1.0 - bluff_prob
                    target_p[2] = bluff_prob
            else:
                if percentile > 0.90 or kelly_max > 0.40:
                    target_p[0], target_p[1], target_p[2], target_p[3], target_p[4] = 0.00, 0.20, 0.35, 0.30, 0.15
                elif percentile > 0.70 or (win_rate > pot_odds + 0.10):
                    target_p[0], target_p[1], target_p[2], target_p[3] = 0.05, 0.55, 0.30, 0.10
                elif win_rate >= pot_odds or (outs * 0.02 >= pot_odds):
                    target_p[0], target_p[1], target_p[2] = 0.20, 0.70, 0.10
                else:
                    target_p[0], target_p[1] = 0.85, 0.15

            # Apply mask
            target_p = target_p * np.array(mask, dtype=np.float32)
            p_sum = target_p.sum()
            target_p = target_p / p_sum if p_sum > 0 else np.array([0, 1, 0, 0, 0], dtype=np.float32)
            target_val = float((win_rate * pot - (1.0 - win_rate) * to_call) / big_blind)

        X_list.append(feats)
        y_prob_list.append(target_p)
        y_val_list.append(target_val)
        mask_list.append(mask)

        if (i + 1) % 1000 == 0 or (i + 1) == num_samples:
            dt = time.time() - t0
            print(f"  Generated {i + 1:5d}/{num_samples} samples ({((i + 1) / num_samples) * 100:5.1f}%) | Elapsed: {dt:.1f}s")

    X = np.array(X_list, dtype=np.float32)
    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    masks = np.array(mask_list, dtype=np.float32)

    return X, y_prob, y_val, masks


def pretrain_gto_prior(model, num_samples=3000, epochs=15, batch_size=64, lr=1e-3, base_model=None):
    """
    Phase 1A: Supervised distillation pretraining using the 80% base model / 20% other player dataset.
    """
    print("\n" + "=" * 76)
    print(f"=== Phase 1A: Supervised Pretraining Distillation ({epochs} Epochs) ===")
    print("=" * 76)

    X, y_prob, y_val, masks = generate_pretraining_dataset(num_samples=num_samples, base_model=base_model)

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y_prob),
        torch.tensor(y_val),
        torch.tensor(masks)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        total_p_loss = 0.0
        total_v_loss = 0.0
        correct_actions = 0
        total_samples = 0
        batches = 0

        for b_x, b_y_prob, b_y_val, b_mask in dataloader:
            optimizer.zero_grad()
            logits, val_pred = model(b_x)

            # Masked Softmax Log-Probs
            masked_logits = logits.masked_fill(b_mask == 0, -1e9)
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            # Policy Loss (KL divergence / Cross Entropy)
            p_loss = -torch.mean(torch.sum(b_y_prob * log_probs, dim=-1))
            # Value Loss (MSE)
            v_loss = F.mse_loss(val_pred, b_y_val)

            loss = p_loss + 0.5 * v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            batches += 1

            # Compute top-1 action agreement with teacher
            pred_acts = log_probs.argmax(dim=-1)
            target_acts = b_y_prob.argmax(dim=-1)
            correct_actions += (pred_acts == target_acts).sum().item()
            total_samples += len(b_x)

        avg_p = total_p_loss / batches
        avg_v = total_v_loss / batches
        acc = (correct_actions / max(1, total_samples)) * 100.0
        elapsed = time.time() - t_start

        print(f"Pretrain Epoch [{epoch:2d}/{epochs:2d}] | Policy Loss: {avg_p:.4f} | Value Loss: {avg_v:.4f} | Action Match: {acc:5.1f}% | Elapsed: {elapsed:5.1f}s")

    print("\nPhase 1A Pretraining distillation completed successfully!")


def pretrain_matchplay_opponents(model, num_hands=300, base_model=None, lr=3e-4):
    """
    Phase 1B: Pre-training against tables of opponents (80% existing saved model, 20% other players)
    with total players varying from 2 to 10.
    """
    print("\n" + "=" * 76)
    print(f"=== Phase 1B: Matchplay Pretraining Against Opponent Pool ({num_hands} Hands) ===")
    print("    Opponents: 80% Existing Saved Model, 20% Other Baseline Players")
    print("    Table sizes: 2 to 10 players (lower sizes more common)")
    print("=" * 76)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    hands_completed = 0
    total_net_bb = 0.0
    wins = 0
    t0 = time.time()

    rollout_buffer = []

    for hand_idx in range(1, num_hands + 1):
        num_players = sample_player_count()

        # Learner is position 0
        learner = MLHoldemPlayer(
            position=0,
            chips=1000,
            model=model,
            deterministic=False,
            temperature=1.0,
            total_players=num_players,
            sample_size=30,
            record_history=True
        )

        opponents = create_pretraining_opponents(
            num_opponents=num_players - 1,
            total_players=num_players,
            base_model=base_model,
            sample_size=30
        )

        game = Game(num_players=num_players, big_blind=20, starting_chips=1000)
        game.players = [learner] + opponents

        # Reset states
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

        start_chips = learner.chips

        try:
            game.run_round()
        except Exception:
            continue

        gain_bb = (learner.chips - start_chips) / float(game.big_blind)
        total_net_bb += gain_bb
        if gain_bb > 0:
            wins += 1
        hands_completed += 1

        # Collect decisions for RL update
        for step in learner.history:
            rollout_buffer.append({
                "state": step["state"],
                "action_idx": step["action_idx"],
                "mask": step["mask"],
                "return": gain_bb
            })

        # Periodic RL update
        if len(rollout_buffer) >= 64:
            b_states = torch.stack([b["state"] for b in rollout_buffer])
            b_actions = torch.tensor([b["action_idx"] for b in rollout_buffer], dtype=torch.long)
            b_masks = torch.stack([b["mask"] for b in rollout_buffer])
            b_returns = torch.tensor([b["return"] for b in rollout_buffer], dtype=torch.float32)

            model.train()
            optimizer.zero_grad()
            logits, val_pred = model(b_states)
            masked_logits = logits.masked_fill(b_masks == 0, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            action_log_probs = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            adv = b_returns - val_pred.detach()
            if len(adv) > 1 and adv.std() > 1e-5:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

            p_loss = -(action_log_probs * adv).mean()
            v_loss = F.mse_loss(val_pred, b_returns)
            loss = p_loss + 0.5 * v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.eval()

            rollout_buffer = []

        if hand_idx % 50 == 0 or hand_idx == num_hands:
            dt = time.time() - t0
            avg_bb = total_net_bb / max(1, hands_completed)
            win_pct = (wins / max(1, hands_completed)) * 100.0
            print(f"Opponent Pretrain Hand [{hand_idx:4d}/{num_hands}] ({hand_idx/num_hands*100:5.1f}%) | Table: {num_players:2d}p | Win Rate: {win_pct:5.1f}% | Net Avg: {avg_bb:+.2f} BB/hand | Elapsed: {dt:.1f}s")

    print("\nPhase 1B Opponent Matchplay completed successfully!")


def train_self_play_rl(model, num_hands=4000, lr=3e-4, save_path="saved_models/holdem_policy_net.pt"):
    """
    Phase 2: Reinforcement Learning Self-Play training (4000 hands).
    Tables vary from 2 to 10 players with lower numbers more common.
    The policy updates through Advantage Actor-Critic (A2C).
    """
    print("\n" + "=" * 76)
    print(f"=== Phase 2: Reinforcement Learning Self-Play Training ({num_hands} Hands) ===")
    print("    All table positions controlled by new policy network (Self-Play).")
    print("    Table size dynamically varies from 2-10 players (lower sizes more common).")
    print("    Advantage Actor-Critic (A2C) updates applied throughout training.")
    print("=" * 76)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hands_completed = 0
    p0_net_chips = 0.0
    p0_wins = 0
    action_counts = {i: 0 for i in range(5)}
    action_labels = ["Fold", "Call", "RaiseSm", "RaiseMed", "AllIn"]

    rollout_buffer = []
    t_start = time.time()
    last_print_time = t_start

    for hand_idx in range(1, num_hands + 1):
        num_players = sample_player_count()

        players = [
            MLHoldemPlayer(
                position=i,
                chips=1000,
                model=model,
                deterministic=False,
                temperature=1.0,
                total_players=num_players,
                sample_size=30,
                record_history=True
            )
            for i in range(num_players)
        ]

        game = Game(num_players=num_players, big_blind=20, starting_chips=1000)
        game.players = players

        # State reset
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

        start_stacks = [p.chips for p in game.players]

        try:
            game.run_round()
        except Exception:
            continue

        hands_completed += 1
        p0_delta = (game.players[0].chips - start_stacks[0]) / float(game.big_blind)
        p0_net_chips += p0_delta
        if p0_delta > 0:
            p0_wins += 1

        # Collect transitions from all players
        for p_idx, p in enumerate(game.players):
            gain_bb = (p.chips - start_stacks[p_idx]) / float(game.big_blind)
            for step in p.history:
                action_counts[step["action_idx"]] += 1
                rollout_buffer.append({
                    "state": step["state"],
                    "action_idx": step["action_idx"],
                    "mask": step["mask"],
                    "return": gain_bb
                })

        # Batch RL update every 80 transitions or every 50 hands
        if len(rollout_buffer) >= 128:
            b_states = torch.stack([b["state"] for b in rollout_buffer])
            b_actions = torch.tensor([b["action_idx"] for b in rollout_buffer], dtype=torch.long)
            b_masks = torch.stack([b["mask"] for b in rollout_buffer])
            b_returns = torch.tensor([b["return"] for b in rollout_buffer], dtype=torch.float32)

            model.train()
            optimizer.zero_grad()
            logits, val_pred = model(b_states)
            masked_logits = logits.masked_fill(b_masks == 0, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            action_log_probs = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # Advantage with normalization
            adv = b_returns - val_pred.detach()
            if len(adv) > 1 and adv.std() > 1e-5:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

            policy_loss = -(action_log_probs * adv).mean()
            value_loss = F.mse_loss(val_pred, b_returns)

            # Entropy regularization
            probs = F.softmax(masked_logits, dim=-1)
            entropy = -(probs * torch.nan_to_num(log_probs, 0.0)).sum(dim=-1).mean()

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.eval()

            rollout_buffer = []

        # Print detailed progress every 100 hands
        if hand_idx % 100 == 0 or hand_idx == num_hands:
            now = time.time()
            elapsed = now - t_start
            hands_per_sec = hands_completed / max(0.1, elapsed)
            remaining_hands = num_hands - hands_completed
            eta_sec = remaining_hands / max(0.1, hands_per_sec)

            p0_avg_bb = p0_net_chips / max(1, hands_completed)
            p0_win_rate = (p0_wins / max(1, hands_completed)) * 100.0

            total_acts = sum(action_counts.values()) or 1
            dist_str = " ".join([f"{label}:{action_counts[i]/total_acts*100:4.1f}%" for i, label in enumerate(action_labels)])

            pct = (hand_idx / num_hands) * 100.0
            print(
                f"[Self-Play Hand {hand_idx:4d}/{num_hands}] ({pct:5.1f}%) | "
                f"P0 Win: {p0_win_rate:4.1f}% ({p0_avg_bb:+.2f} BB/h) | "
                f"Speed: {hands_per_sec:.1f} hands/s | "
                f"ETA: {int(eta_sec//60):02d}m{int(eta_sec%60):02d}s | "
                f"Actions [{dist_str}]"
            )

    # Save final model state dict
    torch.save(model.state_dict(), save_path)
    total_time = time.time() - t_start
    print("\n" + "=" * 76)
    print(f"Self-Play Training Completed: {hands_completed}/{num_hands} hands in {total_time:.1f}s ({total_time/60:.2f} mins).")
    print(f"Final Model successfully saved to: {save_path}")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train New Higher-Dimension Texas Hold'em Policy Net")
    parser.add_argument("--self-play-hands", type=int, default=4000, help="Number of self-play hands (default: 4000)")
    parser.add_argument("--pretrain-samples", type=int, default=3000, help="Pretraining dataset size (default: 3000)")
    parser.add_argument("--pretrain-epochs", type=int, default=15, help="Pretraining epochs (default: 15)")
    parser.add_argument("--pretrain-hands", type=int, default=300, help="Opponent matchplay pretraining hands (default: 300)")
    parser.add_argument("--save-path", type=str, default="saved_models/holdem_policy_net.pt", help="Path to save new model")
    parser.add_argument("--base-model-path", type=str, default="saved_models/holdem_policy_net_base.pt", help="Path to existing saved model")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining and run self-play only")
    args = parser.parse_args()

    print("==========================================================================")
    print("         TEXAS HOLD'EM EXTENDED DIMENSION MODEL TRAINING PIPELINE         ")
    print("==========================================================================")
    print(f"Architecture    : Input Dim = 28 (including player count 2-10), Hidden Dim = 256")
    print(f"Pretraining     : 80% Existing Saved Model + 20% Sprinkled Strategic Baselines")
    print(f"Self-Play Hands : {args.self_play_hands} Hands (Advantage Actor-Critic RL)")
    print(f"Save Path       : {args.save_path}")
    print("==========================================================================\n")

    # Initialize new higher-dimension model
    new_model = HoldemPolicyValueNet(input_dim=28, hidden_dim=256)
    param_count = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print(f"Initialized new policy network with {param_count:,} trainable parameters.")

    # Load existing saved model for pretraining
    base_model = get_base_model(args.base_model_path)

    if not args.skip_pretrain:
        # Phase 1: Pre-training (Distillation + Matchplay against 80% base model / 20% others)
        pretrain_gto_prior(
            model=new_model,
            num_samples=args.pretrain_samples,
            epochs=args.pretrain_epochs,
            base_model=base_model
        )
        pretrain_matchplay_opponents(
            model=new_model,
            num_hands=args.pretrain_hands,
            base_model=base_model
        )

    # Phase 2: Self-play Reinforcement Learning (4000 hands)
    train_self_play_rl(
        model=new_model,
        num_hands=args.self_play_hands,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
