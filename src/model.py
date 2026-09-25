"""
PyTorch Policy-Value Neural Network Architecture for Texas Hold'em
Outputs action probabilities and state value estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HoldemPolicyValueNet(nn.Module):
    """
    Actor-Critic Deep Policy Network for Texas Hold'em.
    Inputs: 24-dimensional game state feature vector.
    Outputs:
      - Policy Logits over 5 discrete actions:
          0: FOLD
          1: CHECK_CALL
          2: RAISE_SMALL
          3: RAISE_MED
          4: RAISE_ALLIN
      - Value Estimate V(s) (expected payoff / chip return)
    """
    ACTION_NAMES = ["fold", "check_call", "raise_small", "raise_med", "raise_allin"]
    NUM_ACTIONS = 5

    def __init__(self, input_dim=24, hidden_dim=128):
        super(HoldemPolicyValueNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared Backbone
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.ln3 = nn.LayerNorm(64)

        # Policy & Value Heads
        self.policy_head = nn.Linear(64, self.NUM_ACTIONS)
        self.value_head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass.
        :param x: FloatTensor of shape (B, input_dim) or (input_dim,)
        :return: (policy_logits, value)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = F.gelu(self.ln1(self.fc1(x)))
        h = F.gelu(self.ln2(self.fc2(h)))
        h = F.gelu(self.ln3(self.fc3(h)))

        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)

        return logits, value

    def get_action_probs(self, x, action_mask=None, temperature=1.0):
        """
        Computes action probability distribution with masking for illegal moves.

        :param x: FloatTensor of shape (B, input_dim) or (input_dim,)
        :param action_mask: FloatTensor of shape (B, 5) or (5,), where 1=valid, 0=invalid move
        :param temperature: Softmax temperature parameter
        :return: Tensor of action probabilities (B, 5) summing to 1.0 per row
        """
        logits, _ = self.forward(x)
        
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / temperature

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # Mask out invalid actions by setting logits to very large negative number
            masked_logits = logits.masked_fill(action_mask == 0, -1e9)
        else:
            masked_logits = logits

        probs = F.softmax(masked_logits, dim=-1)
        return probs

    def predict_action(self, x, action_mask=None, deterministic=False, temperature=1.0):
        """
        Selects an action based on policy probabilities.

        :param x: FloatTensor feature vector
        :param action_mask: FloatTensor valid actions mask
        :param deterministic: If True, returns argmax action; else samples according to probs
        :param temperature: Softmax temperature
        :return: tuple of (action_index, action_name, action_probs_dict)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if action_mask is not None and not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32)

            probs_tensor = self.get_action_probs(x, action_mask=action_mask, temperature=temperature)
            probs = probs_tensor[0].cpu().numpy()

            if deterministic:
                action_idx = int(probs.argmax())
            else:
                # Sample action from probability distribution
                probs_clean = probs / probs.sum()
                action_idx = int(np.random.choice(len(probs_clean), p=probs_clean))

            action_name = self.ACTION_NAMES[action_idx]
            prob_dict = {name: float(prob) for name, prob in zip(self.ACTION_NAMES, probs)}

            return action_idx, action_name, prob_dict
