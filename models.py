import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)
    
class LearnableSimilarity(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(1, 1, bias=False)  # 학습 가능한 가중치 매개변수
    def forward(self, x, y):
        similarity = torch.sum(x * y, dim=-1)  # 내적
        similarity = similarity.unsqueeze(-1)  # (B, S, L, 1)
        weighted_similarity = self.linear(similarity)  # (B, S, L, 1)
        return weighted_similarity

class Adaptive_Attention(nn.Module):
    def __init__(self, d_model, max_shift=20, extra_shifts=None, daily_cycle=288, horizon_offset=1):
        super().__init__()
        self.d_model = d_model
        self.max_shift = max_shift
        self.learnable_similarity = LearnableSimilarity(d_model)

        if extra_shifts is None:
            extra_shifts = []
        default_daily_shift = -(daily_cycle - horizon_offset)
        extra_shifts.append(default_daily_shift)
        self.shifts = sorted(set(list(range(-self.max_shift, 1)) + extra_shifts))

    def forward(self, x):
        B, T, L, D = x.shape
        valid_shifts = [s for s in self.shifts if 0 < abs(s) < T]
        out_all = []

        for b in range(B):
            x_b = x[b:b+1]  # (1, T, L, D)
            x_shifted = []
            for s in valid_shifts:
                pad_len = abs(s)
                pad = torch.zeros(1, pad_len, L, D, device=x.device)
                if s < 0:
                    shifted = torch.cat([x_b[:, pad_len:], pad], dim=1)
                else:
                    shifted = torch.cat([pad, x_b[:, :-pad_len]], dim=1)
                x_shifted.append(shifted)  # (1, T, L, D)
            if not x_shifted:
                out_all.append(x_b)
                continue

            x_shifted = torch.stack(x_shifted, dim=1)  # (1, S, T, L, D)
            S = x_shifted.shape[1]

            target = x_b  # (1, T, L, D)
            target_exp = target.unsqueeze(1).unsqueeze(4)  # (1, 1, T, L, 1, D)
            x_shifted_exp = x_shifted.unsqueeze(3)         # (1, S, T, 1, L, D)

            sim = self.learnable_similarity(target_exp, x_shifted_exp)  # (1, S, T, L, L)
            sim_avg = sim.mean(dim=2)  # (1, S, L, L)
            # attention weights: (1, L, S*L)
            attn_weights = F.softmax(sim_avg.view(1, S * L, L), dim=1)  # (1, S*L, L)
            attn_weights = attn_weights.view(1, L, S * L)  # (1, L, S*L)
            # reshape input for weighted sum
            x_shifted_flat = x_shifted.permute(0, 2, 1, 3, 4).reshape(1, T, S * L, D)  # (1, T, S*L, D)
            # attention application
            output = []
            for l in range(L):
                weight = attn_weights[0, l].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, S*L, 1)
                fused = torch.sum(x_shifted_flat * weight, dim=2)  # (1, T, D)
                output.append(fused.unsqueeze(2))  # (1, T, 1, D)
            out_all.append(torch.cat(output, dim=2))  # (1, T, L, D)
        return torch.cat(out_all, dim=0)  # (B, T, L, D)

class Tramba(nn.Module):
    def __init__(self, d_model=32, in_features=5, pred_len=1, nhead=1, num_links=366,
                 decoder_type="direct", teacher_forcing=False):
        super().__init__()
        assert decoder_type in ["direct", "ar"]
        self.pred_len = pred_len
        self.d_model = d_model
        self.decoder_type = decoder_type
        self.teacher_forcing = teacher_forcing 


        self.embedding = nn.Linear(in_features, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model)
        self.agg_layer = nn.Linear(d_model, d_model)
        self.adaptive_emb = nn.Linear(d_model, d_model)

        self.spatial_attn = Adaptive_Attention(d_model, max_shift=20)
        self.norm1 = nn.LayerNorm(d_model)

        self.temporal_mamba = Mamba(MambaConfig(d_model=d_model, n_layers=2))
        self.ffn = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)


        self.fusion_gate = nn.Parameter(torch.zeros(num_links, 1))  # (L,1)


        if self.decoder_type == "direct":

            self.fc_direct = nn.Linear(d_model, pred_len)  # (B,L,pred_len)
        else:
            self.ar_cell = nn.GRUCell(d_model + 1, d_model)
            self.ar_proj = nn.Linear(d_model, 1)  

            self.step_embed = nn.Embedding(pred_len, d_model)

    def encode_context(self, x):
        """
        x: (B,T,L,F)
        return: x_fused (B,L,D) 
        """
        B, T, L, F = x.shape

        x = self.embedding(x)  # (B,T,L,D)
        pos = torch.arange(T, device=x.device)
        pos_enc = self.pos_embedding(pos).unsqueeze(0).unsqueeze(2)  # (1,T,1,D)
        x = x + pos_enc

        x = self.agg_layer(x)  # (B,T,L,D)

        x_attn = self.spatial_attn(x)              # (B,T,L,D)
        x_attn = self.norm1(x_attn)
        x_attn_last = x_attn[:, -1, :, :]          # (B,L,D)

        x_mamba = x.permute(0, 2, 1, 3).reshape(B * L, T, -1)  # (B*L,T,D)
        x_mamba_out = self.temporal_mamba(x_mamba)             # (B*L,T,D)
        x_mamba_out = self.ffn(x_mamba_out) + x_mamba_out
        x_mamba_out = self.norm2(x_mamba_out)
        x_mamba_last = x_mamba_out[:, -1, :].view(B, L, -1)    # (B,L,D)

        gate = torch.sigmoid(self.fusion_gate).unsqueeze(0)     # (1,L,1)
        x_fused = gate * x_attn_last + (1 - gate) * x_mamba_last  # (B,L,D)
        return x_fused

    def forward(self, x, y_true=None):
        """
        x: (B,T,L,F)
        y_true: (B,pred_len,L,1) 
        return: (B,pred_len,L,1)
        """
        B, T, L, F = x.shape
        ctx = self.encode_context(x)  # (B,L,D)

        if self.decoder_type == "direct":
            # ----- Direct multi-horizon head -----
            y = self.fc_direct(ctx)            # (B,L,pred_len)
            y = y.permute(0, 2, 1).unsqueeze(-1)  # (B,pred_len,L,1)
            return y

        else:
            last_y = torch.zeros(B, L, 1, device=x.device)  # (B,L,1)
            h = ctx  # (B,L,D) 

            outs = []
            for t in range(self.pred_len):
                step_e = self.step_embed.weight[t].unsqueeze(0).unsqueeze(1)  # (1,1,D)
                step_e = step_e.expand(B, L, -1)                              # (B,L,D)

                ar_in = torch.cat([h, last_y, step_e], dim=-1)  # (B,L,D+1+D) = (B,L,2D+1)

                if not hasattr(self, "ar_in_proj"):
                    self.ar_in_proj = nn.Linear(2 * self.d_model + 1, self.d_model)
                ar_in_red = torch.tanh(self.ar_in_proj(ar_in))   # (B,L,D)


                h = h.reshape(B * L, self.d_model)
                ar_in_red = ar_in_red.reshape(B * L, self.d_model)
                h = self.ar_cell(ar_in_red, h)                   # (B*L,D)
                h = h.view(B, L, self.d_model)

                y_t = self.ar_proj(h)                            # (B,L,1)
                outs.append(y_t)

                if self.teacher_forcing and (y_true is not None):
                    
                    last_y = y_true[:, t, :, :]                  # (B,L,1)
                else:
                    
                    last_y = y_t.detach()

            y = torch.stack(outs, dim=1)  # (B,pred_len,L,1)
            return y

