import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    A pure PyTorch implementation of the Mamba (State Space Model) block.
    This serves as a template and fallback for architectures without CUDA compilation access.
    
    Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    """
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        
        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters (Discretization & Selective Scan)
        # s_x, s_y, etc. are used to project inputs to Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2, bias=False)  # dt, B, C
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A parameter (State transition parameter)
        # S4D real initialization
        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        
        # D parameter (Skip connection / Feed-forward term)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: Output sequence tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Input projection & split into branches
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        # 2. Convolution branch
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_branch)[:, :, :seq_len]  # Apply conv and trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x_act = F.silu(x_conv)
        
        # 3. Selective SSM
        # Project to dt, B, C
        proj_val = self.x_proj(x_act)  # (B, L, 1 + 2*d_state)
        dt, B, C = torch.split(proj_val, [1, self.d_state, self.d_state], dim=-1)
        
        # Discretize dt
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Compute dynamic discretization of A (S4D style)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Perform selective scan (sequential loop for simplicity and pure PyTorch support)
        # h: (B, d_inner, d_state)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        y = torch.zeros(batch_size, seq_len, self.d_inner, device=x.device)
        
        for t in range(seq_len):
            # dt_t: (B, d_inner)
            dt_t = dt[:, t, :]
            # B_t: (B, d_state)
            B_t = B[:, t, :]
            # C_t: (B, d_state)
            C_t = C[:, t, :]
            # x_t: (B, d_inner)
            x_t = x_act[:, t, :]
            
            # Discretized A_bar: (B, d_inner, d_state)
            # A_bar_t = exp(dt_t * A)
            A_bar_t = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            
            # Discretized B_bar: (B, d_inner, d_state)
            # B_bar_t = dt_t * B_t
            B_bar_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # State update
            h = A_bar_t * h + B_bar_t * x_t.unsqueeze(-1)
            
            # Output generation: y_t = C_t * h + D * x_t
            # h: (B, d_inner, d_state)
            # C_t: (B, d_state)
            y_t = torch.einsum("bds,bs->bd", h, C_t)
            y[:, t, :] = y_t
            
        # Add residual connection for SSM branch
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)
        
        # 4. Multiplicative gate with z branch
        z_act = F.silu(z_branch)
        y = y * z_act
        
        # 5. Output projection
        out = self.out_proj(y)
        return out
