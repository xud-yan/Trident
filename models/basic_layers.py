import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,            # Input dimension.
            latent_dim,         # Hidden layer dimension.
            out_dim,            # Output dimension.
            num_layers=2,       # Number of layers (incl. input & output).
            bias=True,          # Bias term in Linear layers.
            batchnorm=True,     # Use BatchNorm.
            layernorm=False,    # Use LayerNorm.
            dropout=0,          
            end_relu=False,     # Use ReLU at the end.
            drop_input=0,       # Dropout at input.
            drop_output=0,       # Dropout at output.
            final_linear_bias=True
        ):
        super(MLP, self).__init__()
        mod = []

        if drop_input > 0:
            mod.append(nn.Dropout(drop_input))

        mod.append(nn.Linear(inp_dim, latent_dim, bias=bias))
        if batchnorm:
            mod.append(nn.BatchNorm1d(latent_dim))
        if layernorm:
            mod.append(nn.LayerNorm(latent_dim))
        mod.append(nn.ReLU(True))

        for L in range(num_layers-2):
            mod.append(nn.Linear(latent_dim, latent_dim, bias=bias))
            if batchnorm:
                mod.append(nn.BatchNorm1d(latent_dim))
            if layernorm:
                mod.append(nn.LayerNorm(latent_dim))
            mod.append(nn.ReLU(True))
        
        if dropout > 0:
            mod.append(nn.Dropout(dropout))

        mod.append(nn.Linear(latent_dim, out_dim, bias=final_linear_bias))

        if end_relu:
            mod.append(nn.ReLU(True))

        if drop_output > 0:
            mod.append(nn.Dropout(drop_output))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class MLP_For_Remain(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP_For_Remain, self).__init__()
        self.remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.remain_map(x)