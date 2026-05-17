import torch
import torch.nn as nn
import math

FEAT_H = 56
FEAT_W = 56
NUM_PATCHES = FEAT_H * FEAT_W


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, h=56, w=56):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")

        pe = torch.zeros(dim, h, w)
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0., half_dim, 2) * -(math.log(10000.) / half_dim)
        )
        pos_w = torch.arange(0., w).unsqueeze(1)
        pos_h = torch.arange(0., h).unsqueeze(1)

        pe[0:half_dim:2, :, :] = (torch.sin(pos_w * div_term).T.unsqueeze(1).expand(-1, h, -1))
        pe[1:half_dim:2, :, :] = (torch.cos(pos_w * div_term).T.unsqueeze(1).expand(-1, h, -1))
        pe[half_dim::2, :, :] = (torch.sin(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, w))
        pe[half_dim + 1::2, :, :] = (torch.cos(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, w))

        pe = pe.permute(1, 2, 0).reshape(1, h * w, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class FeatureEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, h=56, w=56):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_enc = PositionalEncoding2D(hidden_dim, h, w)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='linear')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.pos_enc(self.norm(self.proj(x)))


class RefGuideBlock(nn.Module):
    def __init__(self, dim, h=56, w=56, local_kernel=3, ffn_ratio=2,drop=0.0,):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w

        self.feat_norm = nn.LayerNorm(dim)
        self.ref_norm = nn.LayerNorm(dim)

        self.rfi_q = nn.Linear(dim, dim)
        self.rfi_k = nn.Linear(dim, dim)
        self.rfi_v = nn.Linear(dim, dim)
        self.rfi_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=local_kernel,padding=local_kernel // 2, groups=dim),
            nn.BatchNorm2d(dim),
            )
        self.rfi_proj = nn.Linear(dim, dim)

        self.ram_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.ram_proj = nn.Linear(dim, dim)

        self.branch_alpha = nn.Parameter(torch.tensor(0.0))

        self.ffn_norm = nn.LayerNorm(dim)
        ffn_hidden = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(drop),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat, ref):
        bsz, num_tokens, dim = feat.shape

        feat_n = self.feat_norm(feat)
        ref_n = self.ref_norm(ref)

        q = self.rfi_q(feat_n)
        k = self.rfi_k(ref_n)
        v = self.rfi_v(ref_n)
        gate = torch.sigmoid(q * k * (dim ** -0.5))
        rfi_out = gate * v
        rfi_out = rfi_out.permute(0, 2, 1).reshape(bsz, dim, self.h, self.w)
        rfi_out = self.rfi_local(rfi_out)
        rfi_out = rfi_out.reshape(bsz, dim, num_tokens).permute(0, 2, 1)
        rfi_out = self.rfi_proj(rfi_out)

        ram_gate = self.ram_gate(torch.cat([feat_n, ref_n], dim=-1))
        ram_out = ram_gate * feat_n
        ram_out = self.ram_proj(ram_out)

        alpha = torch.sigmoid(self.branch_alpha)
        combined = alpha * rfi_out + (1.0 - alpha) * ram_out

        out = self.ffn_norm(combined)
        out = combined + self.ffn(out)
        return out


class UniRCM_Net(nn.Module):
    def __init__(self, in_features, out_features, num_patches=NUM_PATCHES,
                 hidden_dim=512, num_blocks=3, h=FEAT_H, w=FEAT_W):
        super().__init__()
        self.feat_embed = FeatureEmbedding(in_features, hidden_dim, h, w)
        self.ref_param = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.1)
        self.ref_pos = PositionalEncoding2D(hidden_dim, h, w)

        self.blocks = nn.ModuleList([
            RefGuideBlock(hidden_dim, h=h, w=w, drop=0.0)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        bsz = x.shape[0]
        feat = self.feat_embed(x)
        ref = self.ref_pos(self.ref_param.expand(bsz, -1, -1))
        for block in self.blocks:
            feat = block(feat, ref)
        return self.out_proj(self.final_norm(feat))