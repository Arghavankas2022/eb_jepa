## CellEncoder and CellPredictor for the Cell-JEPA model.
##
## Architecture overview:
##   CellEncoder:  gene expression [n_genes] → latent embedding [embed_dim=256]
##                 3-layer MLP with BatchNorm + ReLU hidden layers
##
##   CellPredictor: latent embedding at time t → predicted embedding at time t+1
##                  3-layer MLP with BatchNorm + ReLU hidden layers
##                  No action input (cell fate is determined by internal state only)

import torch
import torch.nn as nn

# Simple MLP backbone
class MLP(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers=3):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Cell encoder mapping gene expression directly to latent space
class CellEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_dim):
        super().__init__()
        self.backbone = MLP(in_dim, embed_dim, hidden_dim) #in_dim : 45525 --> out: 256
        self.embed_dim = embed_dim
        self.out_dim = embed_dim

    def forward(self, x):
        # x: [B, C, T, H, W] or [B, C]
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            # Flatten spatial dims and process each time-step independently
            # [B, C, T, 1, 1] → [B, T, C] → [B*T, C] → MLP → [B*T, D]
            x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, -1)
            z = self.backbone(x_flat)
            # Reshape back to [B, D, T, 1, 1]
            D = z.shape[-1]
            return z.view(B, T, D).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)     
        elif x.dim() == 2:
            z = self.backbone(x)
            return z
        return x

# Predictor mapping current state to next state
class CellPredictor(nn.Module):
    def __init__(self, embed_dim, pred_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, embed_dim)  
        )
        self.is_rnn = False 
        self.context_length = 0

    def forward(self, state, action=None):
        # state: [B, D, T, 1, 1]
        # For MLP predictor, we typically take the last state if T > 1
        x = state[:, :, -1].flatten(1)
        out = self.net(x)
        # return [B, D, 1, 1, 1]
        return out.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
