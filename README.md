# Texas Hold'em Machine Learning Player & Strategy Engine

An advanced Machine Learning framework for playing **Texas Hold'em**, built on top of the PyPI library [`texas-hold-em-utils`](https://pypi.org/project/texas-hold-em-utils/).

This package implements a **Deep Actor-Critic Policy Network** (`HoldemPolicyValueNet`) trained via a 2-phase pipeline combining **GTO Prior Distillation** and **Self-Play Reinforcement Learning (PPO/A2C)**.

For any given game state, the model evaluates statistical metrics, drawing odds, and pot geometry to output **recommended game moves and their probability distributions**.

---

## 🌟 Key Features

1. **Integrated Feature Extraction (`src/feature_extractor.py`)**:
   - Uses `texas_hold_em_utils` to compute 24-dimensional normalized state representations:
     - Monte Carlo expected win rate & 2-player win rate (`get_hand_rank_details`).
     - Hand percentile rank & ideal Kelly criterion bet fraction.
     - Sklansky hand group (1–9).
     - Drawing outs (1-card and 2-card outs via `outs_counter`).
     - Made hand classification (`HandOfFive`).
     - Pot odds, stack-to-pot ratio (SPR), position relative to dealer, and round stage.

2. **Probabilistic Policy Model (`src/model.py`)**:
   - PyTorch `HoldemPolicyValueNet` architecture with LayerNorm, GELU activations, and valid action masking.
   - Outputs discrete action probability distribution $P(\text{action})$ over 5 moves:
     - `FOLD`
     - `CHECK_CALL`
     - `RAISE_SMALL` (0.5x Pot / Min-raise)
     - `RAISE_MED` (1.0x Pot)
     - `RAISE_ALLIN` (2.0x Pot / All-In)
   - Outputs state value estimate $V(s)$ (expected chip return).

3. **High-Level Scenario Predictor API (`src/predictor.py`)**:
   - Single-function call `predict_scenario(...)` for evaluating custom hands and game situations.
   - Accepts flexible card notations (e.g. `["As", "Ks"]` or `["A of Spades", "K of Spades"]`).

4. **Self-Play Training Framework (`src/train.py`)**:
   - Phase 1: Pre-training GTO Strategy Priors via supervised KL-divergence distillation.
   - Phase 2: Self-play Reinforcement Learning against diverse opponent agents (`KellyMaxProportionPlayer`, `LimpPlayer`, `SimplePlayer`, `AllInPreFlopPlayer`).

5. **Tournament Benchmarking (`run_tournament.py`)**:
   - Evaluates the trained ML model in multi-table tournament hands against baseline strategies.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install texas-hold-em-utils torch numpy pandas scipy scikit-learn pytest
```

### 2. Predict Move & Probabilities for any Scenario

```python
from src.predictor import HoldemPredictor

predictor = HoldemPredictor(model_path="saved_models/holdem_policy_net.pt")

# Evaluate a Flop scenario with As Ks on Qs Js 2h board
result = predictor.predict_scenario(
    hole_cards=["As", "Ks"],
    community_cards=["Qs", "Js", "2h"],
    pot=150,
    all_day=40,
    round_bet=0,
    stack=980,
    position=2,
    player_ct=4
)

print("Recommended Move:", result["recommended_action"])
print("Recommended Bet :", result["recommended_bet_amount"], "chips")
print("Move Probabilities:")
for action, prob in result["action_probabilities"].items():
    print(f"  {action:<12}: {prob * 100:6.2f}%")
```

### 3. Run Interactive Scenario Demo

```bash
python demo.py
```

### 4. Train Model (GTO Pretraining + Self-Play RL)

```bash
python -m src.train
```

### 5. Run Tournament Benchmark

```bash
python run_tournament.py
```

---

## 📂 Project Structure

```
c:\dev\python_projects\texas-holdem-player\
├── src\
│   ├── __init__.py           # Package exports
│   ├── feature_extractor.py  # 24-dim feature vector generation via texas_hold_em_utils
│   ├── model.py              # PyTorch HoldemPolicyValueNet (Actor-Critic Neural Net)
│   ├── player_agent.py       # MLHoldemPlayer subclassing texas_hold_em_utils Player
│   ├── train.py              # GTO Distillation & Reinforcement Learning pipeline
│   └── predictor.py          # High-level scenario prediction API
├── tests\
│   ├── test_feature_extractor.py
│   └── test_model.py
├── demo.py                   # Demo scenario runner displaying probabilities
├── run_tournament.py         # Tournament benchmarking against baseline players
├── saved_models\
│   └── holdem_policy_net.pt  # Trained model weights
└── README.md
```
