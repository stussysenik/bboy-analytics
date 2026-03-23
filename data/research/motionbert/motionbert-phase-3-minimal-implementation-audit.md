# motionbert — Minimal Implementation Audit

_Generated: 2026-03-23T02:10:52.304Z_

---

# MotionBERT Minimal Reimplementation Plan

## Architecture Overview

MotionBERT = **DSTformer backbone** + **task-specific heads**. The backbone is the paper's contribution — a dual-stream spatio-temporal transformer that alternates between joint-to-joint attention (spatial) and frame-to-frame attention (temporal), fused with learned weights.

```
Input (B,T,J,2) → Embedding (B,T,J,D) → [DSTformer Block × 5] → Head → Output (B,T,J,3)

Each DSTformer Block:
  x → S-MHSA(x) → s_out
  x → T-MHSA(x) → t_out
  fused = α·s_out + (1-α)·t_out    # α learned per-block
  out = FFN(fused) + x              # residual
```

---

## ESSENTIAL — Critical Path

### Module 1: `SpatialTemporalAttention` (~80 LOC)

**What it does:** Standard multi-head self-attention, applied either across joints (spatial, per-frame) or across frames (temporal, per-joint) depending on how the input is reshaped.

**Key functions:**
```python
def __init__(self, dim: int = 256, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0)
def forward(self, x: Tensor) -> Tensor  # x: (B*N, L, D) → (B*N, L, D)
```

**Data structures:**
- `qkv`: `nn.Linear(D, 3*D)` — projects to queries, keys, values
- `proj`: `nn.Linear(D, D)` — output projection

**Pseudocode:**
```python
class Attention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads           # 32
        self.scale = self.head_dim ** -0.5         # 1/sqrt(32)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (BN, L, D) where BN = B*T (spatial) or B*J (temporal), L = J or T
        BN, L, D = x.shape
        # qkv: (BN, L, 3, H, head_dim)
        qkv = self.qkv(x).reshape(BN, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each: (BN, H, L, head_dim)

        # attn: (BN, H, L, L) — attention matrix
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # out: (BN, L, D)
        out = (attn @ v).transpose(1, 2).reshape(BN, L, D)
        return self.proj_drop(self.proj(out))
```

**Estimated LOC:** 35

---

### Module 2: `DSTformerBlock` (~60 LOC)

**What it does:** One dual-stream block — runs spatial and temporal attention in parallel, fuses with a learned scalar, applies FFN with residual.

**Key functions:**
```python
def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, num_joints: int = 17, num_frames: int = 243)
def forward(self, x: Tensor) -> Tensor  # x: (B, T, J, D) → (B, T, J, D)
```

**Data structures:**
- `spatial_attn`: `Attention` — applied per-frame across joints
- `temporal_attn`: `Attention` — applied per-joint across frames
- `alpha`: `nn.Parameter(torch.zeros(1))` — fusion weight (sigmoid → [0,1])
- `norm1, norm2, norm3`: `nn.LayerNorm(D)`
- `mlp`: 2-layer FFN `Linear(D, D*4) → GELU → Linear(D*4, D)`

**Pseudocode:**
```python
class DSTformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0):
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, num_heads)
        self.temporal_attn = Attention(dim, num_heads)
        self.alpha = nn.Parameter(torch.zeros(1))     # init → sigmoid(0) = 0.5
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden, dim), nn.Dropout(0.0),
        )

    def forward(self, x):
        # x: (B, T, J, D)
        B, T, J, D = x.shape

        # Spatial stream: attention across J joints, independently per frame
        # Reshape: (B, T, J, D) → (B*T, J, D)
        xs = self.norm_s(x).reshape(B * T, J, D)
        xs = self.spatial_attn(xs).reshape(B, T, J, D)

        # Temporal stream: attention across T frames, independently per joint
        # Reshape: (B, T, J, D) → (B, J, T, D) → (B*J, T, D)
        xt = self.norm_t(x).permute(0, 2, 1, 3).reshape(B * J, T, D)
        xt = self.temporal_attn(xt).reshape(B, J, T, D).permute(0, 2, 1, 3)

        # Fusion: learned weighted sum
        w = torch.sigmoid(self.alpha)        # scalar in [0, 1]
        fused = w * xs + (1 - w) * xt        # (B, T, J, D)

        # Residual + FFN
        x = x + fused
        x = x + self.mlp(self.norm_ffn(x))   # (B, T, J, D)
        return x
```

**Estimated LOC:** 45

---

### Module 3: `DSTformer` (backbone) (~70 LOC)

**What it does:** Full backbone — embedding + positional encodings + 5 DSTformer blocks + output projection.

**Key functions:**
```python
def __init__(self, num_joints=17, in_channels=2, dim=256, depth=5, num_heads=8, num_frames=243)
def forward(self, x: Tensor) -> Tensor  # x: (B, T, J, C_in) → (B, T, J, D)
```

**Data structures:**
- `joint_embed`: `nn.Linear(C_in, D)` — input projection
- `spatial_pos`: `nn.Parameter(torch.zeros(1, 1, J, D))` — joint-index PE
- `temporal_pos`: `nn.Parameter(torch.zeros(1, T, 1, D))` — frame-index PE
- `blocks`: `nn.ModuleList` of 5 `DSTformerBlock`
- `norm`: `nn.LayerNorm(D)` — final normalization

**Pseudocode:**
```python
class DSTformer(nn.Module):
    def __init__(self, num_joints=17, in_channels=2, dim=256, depth=5,
                 num_heads=8, mlp_ratio=4.0, num_frames=243):
        self.joint_embed = nn.Linear(in_channels, dim)
        # Positional embeddings: both learnable, added to every block input
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.blocks = nn.ModuleList([
            DSTformerBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, J, C_in) e.g. (B, 243, 17, 2)
        B, T, J, C = x.shape

        # Embed: (B, T, J, C_in) → (B, T, J, D)
        x = self.joint_embed(x)

        # Add positional embeddings (broadcast over batch)
        x = x + self.spatial_pos[:, :, :J, :]    # (1, 1, J, D) broadcasts
        x = x + self.temporal_pos[:, :T, :, :]    # (1, T, 1, D) broadcasts

        for block in self.blocks:
            x = block(x)                           # (B, T, J, D)

        return self.norm(x)                        # (B, T, J, D)
```

**Estimated LOC:** 40

---

### Module 4: `MotionBERT3DLift` (task head) (~30 LOC)

**What it does:** Thin regression head on top of DSTformer — projects D→3 for 3D joint coordinates.

**Key functions:**
```python
def __init__(self, num_joints=17, in_channels=2, dim=256, ...)
def forward(self, x_2d: Tensor) -> Tensor  # (B, T, J, 2) → (B, T, J, 3)
```

**Pseudocode:**
```python
class MotionBERT3DLift(nn.Module):
    def __init__(self, num_joints=17, in_channels=2, dim=256, depth=5,
                 num_heads=8, num_frames=243):
        self.backbone = DSTformer(num_joints, in_channels, dim, depth,
                                   num_heads, num_frames=num_frames)
        self.head = nn.Linear(dim, 3)

    def forward(self, x_2d):
        # x_2d: (B, T, J, 2) — normalized 2D keypoints
        feat = self.backbone(x_2d)    # (B, T, J, D=256)
        x_3d = self.head(feat)        # (B, T, J, 3)
        return x_3d
```

**Estimated LOC:** 20

---

### Module 5: `LossFunction` (~25 LOC)

**What it does:** MPJPE loss + velocity loss for temporal consistency.

**Key functions:**
```python
def mpjpe(pred: Tensor, target: Tensor) -> Tensor
def velocity_loss(pred: Tensor, target: Tensor) -> Tensor
def total_loss(pred, target, lambda_vel=0.5) -> Tensor
```

**Pseudocode:**
```python
def mpjpe(pred, target):
    # pred, target: (B, T, J, 3)
    # Per-joint L2 error, averaged
    return torch.mean(torch.norm(pred - target, dim=-1))  # scalar

def velocity_loss(pred, target):
    # Temporal finite differences: (B, T-1, J, 3)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(pred_vel - target_vel, dim=-1))

def total_loss(pred, target, lambda_vel=0.5):
    return mpjpe(pred, target) + lambda_vel * velocity_loss(pred, target)
```

**Estimated LOC:** 15

---

### Module 6: `Dataset` (H36M loader) (~80 LOC)

**What it does:** Loads 2D keypoint sequences (from CPN detections) and paired 3D ground truth. Handles the 243-frame sliding window.

**Key functions:**
```python
def __init__(self, data_path: str, num_frames: int = 243, split: str = 'train')
def __getitem__(self, idx) -> Tuple[Tensor, Tensor]  # (T, J, 2), (T, J, 3)
```

**Data structures:**
- `motion_2d`: `np.ndarray` shape `(N, T, J, 2)` — detected 2D keypoints (CPN or GT)
- `motion_3d`: `np.ndarray` shape `(N, T, J, 3)` — ground truth 3D

**Pseudocode:**
```python
class H36MDataset(Dataset):
    def __init__(self, data_path, num_frames=243, split='train'):
        # Load pre-processed .npy files (standard MotionBERT format)
        data = np.load(data_path, allow_pickle=True)
        self.motion_2d = data['motion_2d']  # (N_clips, T, 17, 2) normalized
        self.motion_3d = data['motion_3d']  # (N_clips, T, 17, 3) root-relative mm
        self.num_frames = num_frames

    def __len__(self):
        return len(self.motion_2d)

    def __getitem__(self, idx):
        # Extract clip, pad/crop to num_frames
        m2d = self.motion_2d[idx]         # (T_clip, 17, 2)
        m3d = self.motion_3d[idx]         # (T_clip, 17, 3)

        T = m2d.shape[0]
        if T >= self.num_frames:
            # Random crop
            start = random.randint(0, T - self.num_frames)
            m2d = m2d[start:start + self.num_frames]
            m3d = m3d[start:start + self.num_frames]
        else:
            # Pad by repeating last frame
            pad = self.num_frames - T
            m2d = np.concatenate([m2d, np.tile(m2d[-1:], (pad, 1, 1))], axis=0)
            m3d = np.concatenate([m3d, np.tile(m3d[-1:], (pad, 1, 1))], axis=0)

        return torch.FloatTensor(m2d), torch.FloatTensor(m3d)
```

**Estimated LOC:** 55

---

### Module 7: `train.py` (training loop) (~80 LOC)

**What it does:** Standard PyTorch training loop — single GPU, AdamW, cosine LR schedule.

**Key functions:**
```python
def train_one_epoch(model, loader, optimizer, device) -> float
def evaluate(model, loader, device) -> float  # returns MPJPE in mm
def main()
```

**Pseudocode:**
```python
def main():
    model = MotionBERT3DLift(num_joints=17, in_channels=2, dim=256,
                              depth=5, num_heads=8, num_frames=243).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_set = H36MDataset('data/h36m_train.npz', num_frames=243, split='train')
    val_set   = H36MDataset('data/h36m_test.npz', num_frames=243, split='test')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for x_2d, x_3d in train_loader:
            # x_2d: (B, 243, 17, 2), x_3d: (B, 243, 17, 3)
            x_2d, x_3d = x_2d.cuda(), x_3d.cuda()
            pred_3d = model(x_2d)                         # (B, 243, 17, 3)
            loss = total_loss(pred_3d, x_3d, lambda_vel=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            errors = []
            for x_2d, x_3d in val_loader:
                x_2d, x_3d = x_2d.cuda(), x_3d.cuda()
                pred_3d = model(x_2d)
                err = torch.mean(torch.norm(pred_3d - x_3d, dim=-1))  # mm
                errors.append(err.item())
            print(f"Epoch {epoch}: loss={epoch_loss/len(train_loader):.4f}, "
                  f"MPJPE={np.mean(errors):.1f}mm")
```

**Estimated LOC:** 65

---

## NICE-TO-HAVE

### 1. Confidence-weighted input (`+~15 LOC`)

MotionBERT uses 2D detection confidence as a third input channel when using detected (not GT) 2D keypoints: `C_in = 3` (x, y, confidence).

- **What you lose:** ~2-4mm MPJPE on detected 2D inputs (model can't downweight noisy joints)
- **Quality impact:** Matters for CPN/ViTPose detections; irrelevant for GT-2D evaluation
- **LOC:** Change `in_channels=2` → `in_channels=3`, append confidence channel in dataset

### 2. Mesh recovery head (SMPL regression) (`+~60 LOC`)

Linear head that regresses SMPL parameters (pose θ ∈ ℝ^72, shape β ∈ ℝ^10) from backbone features. Needs SMPL body model for mesh generation.

- **What you lose:** No mesh output, only joint coordinates
- **Quality impact:** 0mm on 3D pose lifting — this is a separate task
- **LOC:** ~60 for head + SMPL forward kinematics wrapper

### 3. Action recognition head (`+~40 LOC`)

Global average pooling → MLP classifier over NTU RGB+D action classes.

- **What you lose:** No action classification capability
- **Quality impact:** 0mm on 3D pose lifting — separate task
- **LOC:** ~40

### 4. Multi-task training (`+~30 LOC`)

Joint training on pose lifting + mesh + action recognition with weighted losses. The paper shows ~1-2mm improvement from multi-task regularization.

- **What you lose:** ~1-2mm MPJPE improvement from regularization effect
- **Quality impact:** Minor — most of the benefit comes from pretraining, not joint training
- **LOC:** ~30

### 5. Procrustes-aligned MPJPE (PA-MPJPE) evaluation (`+~20 LOC`)

Standard metric — aligns prediction to GT via Procrustes before computing MPJPE.

- **What you lose:** Can't report PA-MPJPE (a standard benchmark metric)
- **Quality impact:** 0 on actual model quality; metric-only
- **LOC:** ~20 (scipy `orthogonal_procrustes`)

### 6. Flip augmentation at test time (`+~15 LOC`)

Run input and horizontally-flipped input, average predictions. Standard trick.

- **What you lose:** ~0.5-1mm MPJPE
- **LOC:** ~15

---

## SKIP

| Component | Why safe to skip |
|-----------|-----------------|
| **Distributed training (DDP)** | Single GPU trains in ~12h for 100 epochs on H36M. Paper used 8×V100 for speed, not necessity. |
| **Benchmark evaluation scripts** | H36M protocol-1/protocol-2, 3DPW eval — boilerplate that wraps the same model.forward() |
| **Visualization** | Skeleton rendering, video overlay — pure I/O, no model logic |
| **Ablation infrastructure** | Spatial-only, temporal-only, depth sweeps — set `depth=` and `alpha=` manually |
| **Data preprocessing pipelines** | CPN 2D detection, H36M raw→processed — use pre-processed .npz from MotionBERT repo directly |
| **Mixed precision / gradient checkpointing** | Model is 6.3M params — fits comfortably in FP32 on any modern GPU |
| **EMA (exponential moving average)** | ~0.3mm improvement, training-only complexity |

---

## Dependency Audit

### PyTorch
- **Minimum:** PyTorch ≥ 1.12 (for `nn.MultiheadAttention` improvements, but we write our own)
- **Recommended:** PyTorch ≥ 2.0 (torch.compile for ~30% speedup, optional)

### Required libraries

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `torch` | ≥ 1.12 | Core | Only dependency for model code |
| `numpy` | ≥ 1.21 | Data loading | Standard |
| `einops` | **NOT needed** | — | Paper repo uses it but we reshape manually — fewer deps |
| `timm` | **NOT needed** | — | Paper repo imports it for `trunc_normal_` — we use `nn.init.trunc_normal_` (PyTorch native since 1.8) |

### Pretrained weights

| Checkpoint | URL | Pretrained on | Relevance |
|------------|-----|--------------|-----------|
| **MotionBERT-Lite (3D pose lifting, H36M)** | `https://github.com/Walter0807/MotionBERT/releases` → `best_epoch.bin` | H36M (S1,5,6,7,8 train; S9,11 test) | **Direct use** — this is the 39.2mm checkpoint. Load and evaluate immediately. |
| **MotionBERT (full, multi-task pretrained)** | Same repo → `pretrain/` | H36M + AMASS + 3DPW + NTU | Better starting point for fine-tuning to new domains (breaking). ~6.3M params, ~25MB file. |
| **CPN-detected 2D keypoints for H36M** | Same repo → `data/` | Cascaded Pyramid Network detections | Required to reproduce the 39.2mm CPN-detected result. GT-2D gives 26.9mm. |

**Does pretrain data matter for our use case?** Yes, critically. Per the architecture survey, AMASS training distribution has <1% inverted poses, so pretrained weights carry upright bias. For breaking, fine-tuning on AIST++/BRACE or switching to 3D→3D temporal refinement (bypassing the 2D embedding entirely) is necessary.

---

## Total Estimate

| Category | LOC | Components |
|----------|-----|------------|
| **Essential** | **~315** | Attention (35) + Block (45) + Backbone (40) + Head (20) + Loss (15) + Dataset (55) + Train (65) |
| **+ Nice-to-have** | **~495** | + Confidence (15) + Mesh head (60) + Action head (40) + Multi-task (30) + PA-MPJPE (20) + Flip aug (15) |

### Time to implement
- Essential (315 LOC): One focused session. The architecture is straightforward — five identical blocks with two attention paths fused by a scalar.
- Validation against pretrained checkpoint: Additional time for data download/format wrangling — this is typically the bottleneck, not the model code.

### File structure
```
motionbert_mini/
├── model.py          # Modules 1-4: Attention, Block, Backbone, Head  (~140 LOC)
├── loss.py           # Module 5: MPJPE + velocity loss                (~15 LOC)
├── dataset.py        # Module 6: H36M loader                          (~55 LOC)
├── train.py          # Module 7: Training loop                        (~65 LOC)
└── eval.py           # Inference + MPJPE computation                  (~40 LOC)
```

### Key insight from the survey context

For the breakdancing use case, the **most valuable part to reimplement is the DSTformer backbone** (Modules 1-3, ~120 LOC) — understanding how spatial/temporal attention factorization works and where it breaks down for inverted poses. The task head is trivial. The survey's conclusion that a 3-layer 1D temporal conv captures ~70% of the benefit at 1/40th the training cost means the reimplementation's primary value is **educational and as a baseline**, not as the production path.
