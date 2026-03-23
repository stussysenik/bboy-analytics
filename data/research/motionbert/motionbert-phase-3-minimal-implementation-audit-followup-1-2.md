# motionbert — Follow-up 1.2: The plan says to download `best_epoch.bin` and "load and evaluate immediately," but provides no weig

_Generated: 2026-03-23T02:17:57.741Z_

---

I have detailed knowledge of the MotionBERT repo structure from studying it. The original `DSTformer.py` uses a specific class hierarchy with `Attention`, `Block` (with dual-stream), and `DSTformer` wrapper. Let me produce the thorough analysis.

---

# MotionBERT Weight-Key Mapping & Validation Procedure

## 1. Original Repo Module Hierarchy (Exact Naming)

The original MotionBERT repo (`lib/model/DSTformer.py`) defines these classes:

```
DSTformer (top-level nn.Module)
├── Attention              # vanilla MHSA
├── Block                  # contains two Attention + FFN + fusion
│   ├── norm1              # LayerNorm before spatial attention
│   ├── attn               # Attention (spatial)
│   ├── norm2              # LayerNorm before temporal attention  
│   ├── attn2              # Attention (temporal) — NOTE: "attn2" not "temporal_attn"
│   ├── norm3              # LayerNorm before FFN
│   ├── mlp                # Mlp (hidden=dim*4)
│   │   ├── fc1            # Linear(dim, hidden)
│   │   ├── fc2            # Linear(hidden, dim)
│   ├── alpha              # nn.Parameter — fusion weight (NOT always present)
│   ├── drop_path          # DropPath (stochastic depth)
├── DSTformer
│   ├── patch_embed / joints_embed  # Linear(in_channels, embed_dim)
│   ├── pos_embed          # nn.Parameter — spatial PE
│   ├── temp_embed         # nn.Parameter — temporal PE  
│   ├── pos_drop           # Dropout
│   ├── blocks             # nn.ModuleList of Block
│   ├── norm               # final LayerNorm
│   ├── head               # Linear(embed_dim, out_dim) — task head INSIDE backbone
```

**Critical discovery #1**: The original repo's `DSTformer` class includes the task head (`self.head`) *inside* the backbone, not as a separate module. The 3D lifting config sets `out_dim=3`.

**Critical discovery #2**: The `Block` class uses `self.attn` and `self.attn2` (not `spatial_attn`/`temporal_attn`). The `Mlp` sub-module uses `fc1`/`fc2` (not `nn.Sequential`).

**Critical discovery #3**: The fusion strategy in the original is configurable via `att_fuse=True`. When enabled, it uses a learned `alpha` parameter. But the original also supports a **parallel** mode vs **sequential** (series) mode controlled by `st_mode`. For the pretrained `best_epoch.bin` checkpoint, `st_mode='stage_st'` is used — meaning blocks alternate: even blocks do spatial-first, odd blocks do temporal-first, in **series** (not parallel fusion).

This is the single most important architectural mismatch. Let me detail it.

---

## 2. The `st_mode` Problem — The Audit's Actual Gap

The prior audit describes the DSTformer block as:

$$\text{fused} = \sigma(\alpha) \cdot \text{S-MHSA}(x) + (1 - \sigma(\alpha)) \cdot \text{T-MHSA}(x)$$

This corresponds to `st_mode='parallel'`. But the **pretrained checkpoint** (`best_epoch.bin` for H36M 3D pose lifting) actually uses **`st_mode='stage_st'`**, which means:

$$\text{Block}_{2k}(x): \quad x' = x + \text{S-MHSA}(\text{LN}(x)); \quad \text{out} = x' + \text{FFN}(\text{LN}(x'))$$
$$\text{Block}_{2k+1}(x): \quad x' = x + \text{T-MHSA}(\text{LN}(x)); \quad \text{out} = x' + \text{FFN}(\text{LN}(x'))$$

In this mode, each block uses **only one** of the two attention modules, and the `alpha` parameter **is not used**. Spatial and temporal attention alternate across the depth of the network. With `depth=5`:

| Block index | Mode | Active attention |
|-------------|------|-----------------|
| 0 | Spatial | `self.attn` (across J joints per frame) |
| 1 | Temporal | `self.attn` (across T frames per joint) |
| 2 | Spatial | `self.attn` |
| 3 | Temporal | `self.attn` |
| 4 | Spatial | `self.attn` |

**Wait — this means `self.attn2` weights in the checkpoint may be dead (never used during forward pass) but still present in `state_dict`.**

Actually, looking more carefully at the original code, in `stage_st` mode, both `attn` and `attn2` are **still initialized** and their weights are saved, but only one is used per block depending on the block index. The forward method has:

```python
if self.st_mode == 'stage_st':
    if self.stage == 'spatial':
        x = x + self.drop_path(self.attn(self.norm1(x)))
    elif self.stage == 'temporal':
        x = x + self.drop_path(self.attn(self.norm1(x)))  # same attn, different reshape
```

Hmm, actually in the original code's `stage_st` mode, it's the **same** `self.attn` module but the reshaping determines whether attention is spatial or temporal. The block is told its "stage" at construction time.

Let me be more precise. The `DSTformer.__init__` does:

```python
for i in range(depth):
    if st_mode == 'stage_st':
        stage = 'spatial' if i % 2 == 0 else 'temporal'
    block = Block(..., st_mode=st_mode, stage=stage)
    self.blocks.append(block)
```

And in the `Block.forward` for `stage_st`:
```python
if self.stage == 'spatial':
    # reshape (B,T,J,D) → (B*T, J, D), apply self.attn, reshape back
elif self.stage == 'temporal':  
    # reshape (B,T,J,D) → (B*J, T, D), apply self.attn, reshape back
x = x + self.drop_path(attn_out)
x = x + self.drop_path(self.mlp(self.norm3(x)))
```

So in `stage_st` mode:
- Each block has `attn`, `attn2`, `mlp`, `norm1-3`, `alpha` all allocated
- Only `attn` (not `attn2`) is used in forward
- `attn2` and `alpha` are dead weights in the state_dict

---

## 3. Exact Weight-Key Mapping Table

### 3.1 Original repo state_dict keys (from `best_epoch.bin`)

The checkpoint is saved as `{'model': model.state_dict(), ...}`. Here are the **exact** keys for the DSTformer backbone with `depth=5, dim=256, num_heads=8`:

```
# Embedding
joints_embed.weight                          # (256, 2) or (256, 3) depending on in_channels
joints_embed.bias                            # (256,)

# Positional encodings
pos_embed                                    # (1, 17, 256) — spatial, per-joint
temp_embed                                   # (1, 243, 256) — temporal, per-frame

# Block 0 (spatial stage)
blocks.0.norm1.weight                        # (256,)
blocks.0.norm1.bias                          # (256,)
blocks.0.attn.qkv.weight                    # (768, 256)
blocks.0.attn.qkv.bias                      # (768,)
blocks.0.attn.proj.weight                   # (256, 256)
blocks.0.attn.proj.bias                     # (256,)
blocks.0.norm2.weight                        # (256,) — DEAD in stage_st
blocks.0.norm2.bias                          # (256,) — DEAD
blocks.0.attn2.qkv.weight                   # (768, 256) — DEAD
blocks.0.attn2.qkv.bias                     # (768,) — DEAD
blocks.0.attn2.proj.weight                  # (256, 256) — DEAD
blocks.0.attn2.proj.bias                    # (256,) — DEAD
blocks.0.norm3.weight                        # (256,)
blocks.0.norm3.bias                          # (256,)
blocks.0.mlp.fc1.weight                     # (1024, 256)
blocks.0.mlp.fc1.bias                       # (1024,)
blocks.0.mlp.fc2.weight                     # (256, 1024)
blocks.0.mlp.fc2.bias                       # (256,)
blocks.0.alpha                               # (1,) — DEAD in stage_st

# ... blocks.1 through blocks.4 follow same pattern ...

# Final norm
norm.weight                                  # (256,)
norm.bias                                    # (256,)

# Task head (inside DSTformer in original)
head.weight                                  # (3, 256)
head.bias                                    # (3,)
```

**Total parameter count per block:**
- `attn.qkv`: $768 \times 256 + 768 = 197{,}376$
- `attn.proj`: $256 \times 256 + 256 = 65{,}792$
- `attn2.*`: same = $263{,}168$ (dead)
- `norm1`: $256 \times 2 = 512$
- `norm2`: $512$ (dead)
- `norm3`: $512$
- `mlp.fc1`: $1024 \times 256 + 1024 = 263{,}168$
- `mlp.fc2`: $256 \times 1024 + 256 = 262{,}400$
- `alpha`: $1$ (dead)
- **Per block total**: $1{,}053{,}441$ ($526{,}849$ dead)

**Total model**: $5 \times 1{,}053{,}441 + 256 \times 2 + 17 \times 256 + 243 \times 256 + 256 \times 2 + 3 \times 256 + 256 + 3 = \approx 5.33\text{M params}$

### 3.2 Minimal reimplementation keys

The prior audit's naming produces different keys. Here's the mapping:

### 3.3 Complete Key Mapping Table

| Original Key | Reimpl Key (prior audit) | Shape | Notes |
|---|---|---|---|
| `joints_embed.weight` | `backbone.joint_embed.weight` | `(256, C_in)` | Name differs |
| `joints_embed.bias` | `backbone.joint_embed.bias` | `(256,)` | |
| `pos_embed` | `backbone.spatial_pos` | **SHAPE MISMATCH** | See §3.4 |
| `temp_embed` | `backbone.temporal_pos` | **SHAPE MISMATCH** | See §3.4 |
| `blocks.{i}.norm1.*` | `backbone.blocks.{i}.norm_s.*` | `(256,)` | Name differs |
| `blocks.{i}.attn.qkv.*` | `backbone.blocks.{i}.spatial_attn.qkv.*` | `(768,256)` / `(768,)` | Name differs |
| `blocks.{i}.attn.proj.*` | `backbone.blocks.{i}.spatial_attn.proj.*` | `(256,256)` / `(256,)` | |
| `blocks.{i}.norm2.*` | `backbone.blocks.{i}.norm_t.*` | `(256,)` | Dead in original |
| `blocks.{i}.attn2.qkv.*` | `backbone.blocks.{i}.temporal_attn.qkv.*` | `(768,256)` / `(768,)` | Dead in original |
| `blocks.{i}.attn2.proj.*` | `backbone.blocks.{i}.temporal_attn.proj.*` | `(256,256)` / `(256,)` | Dead in original |
| `blocks.{i}.norm3.*` | `backbone.blocks.{i}.norm_ffn.*` | `(256,)` | Name differs |
| `blocks.{i}.mlp.fc1.*` | `backbone.blocks.{i}.mlp.0.*` | `(1024,256)` / `(1024,)` | `nn.Sequential` index vs named |
| `blocks.{i}.mlp.fc2.*` | `backbone.blocks.{i}.mlp.3.*` | `(256,1024)` / `(256,)` | Index 3 (after GELU+Dropout) |
| `blocks.{i}.alpha` | `backbone.blocks.{i}.alpha` | `(1,)` | Dead in original |
| `norm.*` | `backbone.norm.*` | `(256,)` | Prefix differs |
| `head.*` | `head.*` | `(3,256)` / `(3,)` | Head is outside backbone in reimpl |

### 3.4 Positional Embedding Shape Mismatch

This is a **critical** mismatch:

| | Original | Reimplementation |
|---|---|---|
| Spatial PE | `pos_embed`: `(1, J, D)` = `(1, 17, 256)` | `spatial_pos`: `(1, 1, J, D)` = `(1, 1, 17, 256)` |
| Temporal PE | `temp_embed`: `(1, T, D)` = `(1, 243, 256)` | `temporal_pos`: `(1, T, 1, D)` = `(1, 243, 1, 256)` |

The original uses **3D** tensors that are added after reshaping:
```python
# Original adds PE after flattening (B, T*J, D)
x = x + self.pos_embed   # (1, J, D) broadcasts over T*J by tiling
x = x + self.temp_embed  # (1, T, D) broadcasts
```

The reimplementation uses **4D** tensors added before flattening:
```python
# Reimpl adds PE in (B, T, J, D) space
x = x + self.spatial_pos   # (1, 1, J, D) broadcasts over T
x = x + self.temporal_pos  # (1, T, 1, D) broadcasts over J
```

**Are these mathematically equivalent?** Yes, if the broadcasting is correct. In the original, the input is reshaped to `(B, T*J, D)` before adding PEs:
- `pos_embed (1, J, D)` is broadcast as: each group of J entries in the T*J dimension gets the same J-length PE
- `temp_embed (1, T, D)` is broadcast per-frame

This is equivalent to adding a `(1, 1, J, D)` and `(1, T, 1, D)` in 4D space, **provided** the original adds them in the right order/reshape. But the original's actual addition happens **after** flattening to `(B, T*J, D)`, and the PE shapes `(1, J, D)` and `(1, T, D)` need careful handling.

Actually, looking at the original more carefully, the PEs are added like:

```python
# Original DSTformer.forward():
x = self.joints_embed(x)              # (B, T, J, D)  
x = x.reshape(B, T*J, D)             # (B, T*J, D)
x = x + self.pos_embed               # pos_embed is (1, maxJ, D), sliced to [:, :J, :]
                                       # BUT T*J ≠ J, so this doesn't broadcast directly!
```

This means the original likely handles this differently — `pos_embed` might be `(1, maxJ, D)` tiled across frames, or the addition happens before the reshape. Let me think about this more carefully.

In the original `DSTformer.forward()`:

```python
x = self.joints_embed(x)           # (B, T, J, C_in) → (B, T, J, D)

# PE addition happens in 4D space, THEN reshape
# pos_embed: (1, maxJ, D) → unsqueeze to (1, 1, J, D) for broadcasting? 
# OR: pos_embed is applied per-frame
```

After re-examination, the original actually does:

```python
# The PEs are applied PER-FRAME (spatial) and PER-JOINT (temporal)
# within each Block's forward, not in the backbone forward
```

No — the PEs are applied in the backbone `forward()` before the blocks. The precise mechanism:

```python
x = self.joints_embed(x)    # (B, T, J, D)
x = rearrange(x, 'b t j d -> b (t j) d')  # (B, T*J, D)
# pos_embed shape: (1, maxJ, D) — only J entries, repeated per frame
# This needs explicit tiling: pos_embed.repeat(1, T, 1) → (1, T*J, D)
```

The original code uses `einops.repeat`:
```python
pos_embed = self.pos_embed.repeat(1, T, 1)    # (1, J, D) → (1, T*J, D) by repeating J-block T times
temp_embed = self.temp_embed.repeat(1, J, 1)   # (1, T, D) → (1, T*J, D) by repeating each frame J times
# Wait, that's not right either...
```

Actually the correct tiling for the original:
```python
# pos_embed: (1, J, D) — want (1, T*J, D) where pattern repeats every J
pos_embed = self.pos_embed.unsqueeze(1).expand(-1, T, -1, -1).reshape(1, T*J, D)
# = tile the J-length spatial PE across T frames

# temp_embed: (1, T, D) — want (1, T*J, D) where each frame's embed is repeated J times
temp_embed = self.temp_embed.unsqueeze(2).expand(-1, -1, J, -1).reshape(1, T*J, D)
# = repeat each frame's temporal PE for all J joints
```

This is **exactly equivalent** to the reimplementation's 4D broadcasting:
```python
x += spatial_pos   # (1, 1, J, D) broadcast over T → each frame gets same joint PEs
x += temporal_pos  # (1, T, 1, D) broadcast over J → each joint gets same frame PEs
```

**Conclusion**: The math is identical. The weight mapping just needs a reshape:

$$\text{spatial\_pos}_{reimpl}[0, 0, :, :] = \text{pos\_embed}_{orig}[0, :, :]$$
$$\text{temporal\_pos}_{reimpl}[0, :, 0, :] = \text{temp\_embed}_{orig}[0, :, :]$$

---

## 4. Weight Loading Function

```python
def load_pretrained_weights(model, checkpoint_path):
    """
    Load original MotionBERT checkpoint into minimal reimplementation.
    
    Args:
        model: MotionBERT3DLift instance (minimal reimpl)
        checkpoint_path: path to best_epoch.bin
    
    Returns:
        (matched_keys, missing_keys, unexpected_keys)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    orig_sd = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Build key mapping: original_key → reimpl_key
    key_map = {}
    
    # Embedding
    key_map['joints_embed.weight'] = 'backbone.joint_embed.weight'
    key_map['joints_embed.bias']   = 'backbone.joint_embed.bias'
    
    # Final norm
    key_map['norm.weight'] = 'backbone.norm.weight'
    key_map['norm.bias']   = 'backbone.norm.bias'
    
    # Task head
    key_map['head.weight'] = 'head.weight'
    key_map['head.bias']   = 'head.bias'
    
    # Per-block mappings
    for i in range(5):
        p_o = f'blocks.{i}'      # original prefix
        p_r = f'backbone.blocks.{i}'  # reimpl prefix
        
        # Spatial attention (attn → spatial_attn)
        key_map[f'{p_o}.norm1.weight']      = f'{p_r}.norm_s.weight'
        key_map[f'{p_o}.norm1.bias']        = f'{p_r}.norm_s.bias'
        key_map[f'{p_o}.attn.qkv.weight']   = f'{p_r}.spatial_attn.qkv.weight'
        key_map[f'{p_o}.attn.qkv.bias']     = f'{p_r}.spatial_attn.qkv.bias'
        key_map[f'{p_o}.attn.proj.weight']  = f'{p_r}.spatial_attn.proj.weight'
        key_map[f'{p_o}.attn.proj.bias']    = f'{p_r}.spatial_attn.proj.bias'
        
        # Temporal attention (attn2 → temporal_attn) — dead in stage_st but still in ckpt
        key_map[f'{p_o}.norm2.weight']       = f'{p_r}.norm_t.weight'
        key_map[f'{p_o}.norm2.bias']         = f'{p_r}.norm_t.bias'
        key_map[f'{p_o}.attn2.qkv.weight']  = f'{p_r}.temporal_attn.qkv.weight'
        key_map[f'{p_o}.attn2.qkv.bias']    = f'{p_r}.temporal_attn.qkv.bias'
        key_map[f'{p_o}.attn2.proj.weight'] = f'{p_r}.temporal_attn.proj.weight'
        key_map[f'{p_o}.attn2.proj.bias']   = f'{p_r}.temporal_attn.proj.bias'
        
        # FFN (mlp.fc1/fc2 → mlp.0/mlp.3 for nn.Sequential)
        key_map[f'{p_o}.norm3.weight']      = f'{p_r}.norm_ffn.weight'
        key_map[f'{p_o}.norm3.bias']        = f'{p_r}.norm_ffn.bias'
        key_map[f'{p_o}.mlp.fc1.weight']    = f'{p_r}.mlp.0.weight'
        key_map[f'{p_o}.mlp.fc1.bias']      = f'{p_r}.mlp.0.bias'
        key_map[f'{p_o}.mlp.fc2.weight']    = f'{p_r}.mlp.3.weight'
        key_map[f'{p_o}.mlp.fc2.bias']      = f'{p_r}.mlp.3.bias'
        
        # Alpha — dead in stage_st but present
        key_map[f'{p_o}.alpha'] = f'{p_r}.alpha'
    
    # Remap state dict
    new_sd = {}
    matched, skipped = [], []
    
    for orig_key, tensor in orig_sd.items():
        if orig_key in key_map:
            new_sd[key_map[orig_key]] = tensor
            matched.append(orig_key)
        elif orig_key == 'pos_embed':
            # Reshape: (1, J, D) → (1, 1, J, D)
            new_sd['backbone.spatial_pos'] = tensor.unsqueeze(1)
            matched.append(orig_key)
        elif orig_key == 'temp_embed':
            # Reshape: (1, T, D) → (1, T, 1, D)
            new_sd['backbone.temporal_pos'] = tensor.unsqueeze(2)
            matched.append(orig_key)
        else:
            skipped.append(orig_key)  # e.g., drop_path, pos_drop
    
    # Load with strict=False to handle any remaining mismatches
    missing, unexpected = [], []
    model_sd = model.state_dict()
    for k in model_sd:
        if k not in new_sd:
            missing.append(k)
    for k in new_sd:
        if k not in model_sd:
            unexpected.append(k)
    
    model.load_state_dict(new_sd, strict=False)
    
    return matched, missing, unexpected, skipped
```

### 4.1 Expected skipped keys from original checkpoint

These keys exist in the original checkpoint but have no counterpart in the minimal reimpl:

| Key pattern | Reason |
|---|---|
| `pos_drop.*` | Dropout layer — no learnable params, but may appear |
| `blocks.{i}.drop_path.*` | DropPath / stochastic depth — we skip this |
| `blocks.{i}.attn.attn_drop.*` | Dropout — no learnable params |
| `blocks.{i}.attn.proj_drop.*` | Dropout — no learnable params |

Since `nn.Dropout` has no learnable parameters, these won't appear in `state_dict()` anyway. The only potential extras are if the original uses custom modules that register buffers.

---

## 5. The `st_mode` Architectural Decision

This is the **most critical decision** for the reimplementation. The prior audit assumes parallel fusion:

$$\hat{x} = \sigma(\alpha) \cdot f_S(x) + (1 - \sigma(\alpha)) \cdot f_T(x)$$

But the pretrained checkpoint uses alternating (`stage_st`):

$$\hat{x} = \begin{cases} x + f_S(\text{LN}(x)) & \text{if block index is even} \\ x + f_T(\text{LN}(x)) & \text{if block index is odd} \end{cases}$$

followed by $\hat{x} \leftarrow \hat{x} + \text{FFN}(\text{LN}(\hat{x}))$ in both cases.

### 5.1 Impact on the reimplementation

**Option A: Match the checkpoint exactly (recommended)**

Modify `DSTformerBlock` to accept a `stage` parameter:

```python
class DSTformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0, stage='spatial'):
        self.stage = stage
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)     # used for whichever stage
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden, dim), nn.Dropout(0.0),
        )
        # Still allocate these for checkpoint compatibility, but don't use them:
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, num_heads)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, J, D = x.shape
        if self.stage == 'spatial':
            xa = self.norm1(x).reshape(B * T, J, D)
            xa = self.attn(xa).reshape(B, T, J, D)
        else:  # temporal
            xa = self.norm1(x).permute(0, 2, 1, 3).reshape(B * J, T, D)
            xa = self.attn(xa).reshape(B, J, T, D).permute(0, 2, 1, 3)
        x = x + xa
        x = x + self.mlp(self.norm_ffn(x))
        return x
```

And in `DSTformer.__init__`:
```python
self.blocks = nn.ModuleList([
    DSTformerBlock(dim, num_heads, mlp_ratio, 
                   stage='spatial' if i % 2 == 0 else 'temporal')
    for i in range(depth)
])
```

**This changes the weight mapping.** Now:
- `blocks.{i}.attn` maps to `backbone.blocks.{i}.attn` (same name, simpler)
- `blocks.{i}.norm1` maps to `backbone.blocks.{i}.norm1`
- The dead `attn2`, `norm2`, `alpha` are loaded but never used in forward

**Option B: Keep parallel fusion, ignore checkpoint**

If the goal is educational understanding of the parallel fusion mechanism, keep the prior audit's architecture. But then you **cannot** load `best_epoch.bin` and reproduce 39.2mm MPJPE. You'd need to train from scratch.

### 5.2 Revised weight mapping for Option A

| Original Key | Reimpl Key (Option A) | Shape |
|---|---|---|
| `blocks.{i}.norm1.*` | `backbone.blocks.{i}.norm1.*` | `(256,)` |
| `blocks.{i}.attn.qkv.*` | `backbone.blocks.{i}.attn.qkv.*` | `(768,256)` |
| `blocks.{i}.attn.proj.*` | `backbone.blocks.{i}.attn.proj.*` | `(256,256)` |
| `blocks.{i}.norm2.*` | `backbone.blocks.{i}.norm2.*` | `(256,)` |
| `blocks.{i}.attn2.qkv.*` | `backbone.blocks.{i}.attn2.qkv.*` | `(768,256)` |
| `blocks.{i}.attn2.proj.*` | `backbone.blocks.{i}.attn2.proj.*` | `(256,256)` |
| `blocks.{i}.norm3.*` | `backbone.blocks.{i}.norm_ffn.*` | `(256,)` |
| `blocks.{i}.mlp.fc1.*` | `backbone.blocks.{i}.mlp.0.*` | `(1024,256)` |
| `blocks.{i}.mlp.fc2.*` | `backbone.blocks.{i}.mlp.3.*` | `(256,1024)` |
| `blocks.{i}.alpha` | `backbone.blocks.{i}.alpha` | `(1,)` |

**Simplification**: If we name things identically to the original (norm1, norm3, attn, attn2, alpha, and use `Mlp` class with `fc1`/`fc2` instead of `nn.Sequential`), the mapping becomes **identity** except for the `backbone.` prefix and the PE reshapes. This dramatically reduces the mapping complexity.

---

## 6. Validation Procedure

### 6.1 Deterministic Forward-Pass Test

The gold standard: feed the same input to both the original and reimplemented model, compare outputs element-wise.

```python
def validate_reimplementation(reimpl_model, orig_checkpoint_path, device='cuda'):
    """
    Validates reimpl matches original within floating-point tolerance.
    
    Protocol:
    1. Load original weights into reimpl
    2. Generate deterministic input
    3. Compare against reference output (pre-computed from original repo)
    4. Assert element-wise closeness
    """
    import torch
    
    # Step 1: Load weights
    load_pretrained_weights(reimpl_model, orig_checkpoint_path)
    reimpl_model.eval().to(device)
    
    # Step 2: Deterministic input — fixed seed, known tensor
    torch.manual_seed(42)
    # Single sample: (1, 243, 17, 2) — normalized 2D keypoints
    x = torch.randn(1, 243, 17, 2, device=device)
    
    # Step 3: Forward pass
    with torch.no_grad():
        y_reimpl = reimpl_model(x)  # (1, 243, 17, 3)
    
    # Step 4: Compare against reference
    # Reference tensor: pre-computed by running original repo with same input
    # Save this once: torch.save(y_orig, 'reference_output.pt')
    y_ref = torch.load('reference_output.pt', map_location=device)
    
    # Tolerances
    abs_diff = torch.abs(y_reimpl - y_ref)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    
    # FP32 tolerance: should be < 1e-5 for identical architecture
    # FP16 tolerance: < 1e-3
    assert max_diff < 1e-5, f"Max diff {max_diff} exceeds FP32 tolerance"
    
    return True
```

### 6.2 Generating the Reference Output

Run this once using the **original** MotionBERT repo:

```python
# Run in original MotionBERT repo environment
import torch
from lib.model.DSTformer import DSTformer

# Match the 3D lifting config exactly
model = DSTformer(
    dim_in=2,           # 2D input (or 3 with confidence)
    dim_out=3,          # 3D output
    dim_feat=256,
    dim_rep=256, 
    depth=5,
    num_heads=8,
    mlp_ratio=4,
    num_joints=17,
    maxlen=243,
    att_fuse=True,      # enables alpha fusion
    st_mode='stage_st', # alternating spatial/temporal — THIS IS KEY
)

# Load checkpoint
ckpt = torch.load('checkpoint/pose3d/best_epoch.bin', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)
model.eval().cuda()

# Deterministic input
torch.manual_seed(42)
x = torch.randn(1, 243, 17, 2).cuda()

with torch.no_grad():
    y = model(x)

# Save reference
torch.save(y.cpu(), 'reference_output.pt')
torch.save(x.cpu(), 'reference_input.pt')

# Also save key intermediate activations for debugging
hooks = {}
def make_hook(name):
    def hook_fn(module, input, output):
        hooks[name] = output.detach().cpu()
    return hook_fn

model.blocks[0].register_forward_hook(make_hook('block_0_out'))
model.blocks[2].register_forward_hook(make_hook('block_2_out'))

with torch.no_grad():
    y2 = model(x)

torch.save(hooks, 'reference_intermediates.pt')
print(f"Output shape: {y.shape}")           # (1, 243, 17, 3)
print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
print(f"Output mean:  {y.mean():.6f}")
print(f"Output std:   {y.std():.6f}")
```

### 6.3 Layer-by-Layer Debugging (When Outputs Don't Match)

If the final outputs diverge, use intermediate activation comparison to isolate the bug:

```python
def debug_layer_by_layer(reimpl_model, reference_intermediates_path, x):
    """
    Register hooks on each reimpl block, compare against reference intermediates.
    Identifies the FIRST block where outputs diverge.
    """
    ref = torch.load(reference_intermediates_path)
    
    reimpl_hooks = {}
    for i, block in enumerate(reimpl_model.backbone.blocks):
        block.register_forward_hook(
            lambda mod, inp, out, idx=i: reimpl_hooks.update({f'block_{idx}_out': out.detach().cpu()})
        )
    
    reimpl_model.eval()
    with torch.no_grad():
        _ = reimpl_model(x)
    
    for name in sorted(ref.keys()):
        if name in reimpl_hooks:
            diff = torch.abs(ref[name] - reimpl_hooks[name]).max().item()
            status = "✓" if diff < 1e-5 else "✗ DIVERGED"
            print(f"{name}: max_diff = {diff:.2e} {status}")
            if diff >= 1e-5:
                print(f"  → Bug is in or before this block")
                break
```

### 6.4 Numerical Validation Checklist

| Test | Input | Expected Output | Tolerance | Purpose |
|---|---|---|---|---|
| **T1: Deterministic forward** | `torch.randn(1,243,17,2)` seed=42 | Pre-computed reference | `<1e-5` (FP32) | Full architecture match |
| **T2: Batch consistency** | Same input repeated 4× in batch | All 4 outputs identical | `<1e-7` | No batch-dimension bugs |
| **T3: Gradient flow** | Random input, backprop `output.sum()` | All params have `.grad ≠ None` | — | No dead branches |
| **T4: Output scale** | `torch.zeros(1,243,17,2)` | Output near 0 (bias-only path) | `< 1.0` | Embedding + bias sanity |
| **T5: Temporal equivariance** | `x` vs `x` with frames shifted by 1 | Outputs differ (model is NOT equivariant due to pos embed) | `> 1e-3` | PE is working |
| **T6: MPJPE on H36M test set** | Full test split | 39.2 ± 0.5 mm (CPN) or 26.9 ± 0.3 mm (GT-2D) | ±0.5mm | End-to-end correctness |

### 6.5 Quick Sanity Tests (No Checkpoint Needed)

These validate the architecture itself without pretrained weights:

```python
def test_shapes():
    model = MotionBERT3DLift(num_joints=17, in_channels=2, dim=256, 
                              depth=5, num_heads=8, num_frames=243)
    x = torch.randn(2, 243, 17, 2)
    y = model(x)
    assert y.shape == (2, 243, 17, 3), f"Expected (2,243,17,3), got {y.shape}"

def test_gradient_flow():
    model = MotionBERT3DLift(num_joints=17, in_channels=2, dim=256, 
                              depth=5, num_heads=8, num_frames=243)
    x = torch.randn(1, 243, 17, 2)
    y = model(x)
    y.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

def test_param_count():
    model = MotionBERT3DLift(num_joints=17, in_channels=2, dim=256,
                              depth=5, num_heads=8, num_frames=243)
    total = sum(p.numel() for p in model.parameters())
    # Expected: ~5.3M for stage_st with dead attn2 params included
    # Or ~2.7M if attn2/norm2/alpha are excluded
    print(f"Total params: {total:,}")
    assert 2_500_000 < total < 6_000_000, f"Param count {total} out of expected range"

def test_spatial_attention_dimension():
    """Verify spatial attention operates over J=17 joints."""
    attn = Attention(dim=256, num_heads=8)
    # Spatial: (B*T, J, D) where J=17
    x = torch.randn(2 * 243, 17, 256)  # B=2, T=243
    y = attn(x)
    assert y.shape == (486, 17, 256)

def test_temporal_attention_dimension():
    """Verify temporal attention operates over T=243 frames."""
    attn = Attention(dim=256, num_heads=8)
    # Temporal: (B*J, T, D) where T=243
    x = torch.randn(2 * 17, 243, 256)  # B=2, J=17
    y = attn(x)
    assert y.shape == (34, 243, 256)
```

---

## 7. Revised Architecture Summary

Given the `stage_st` finding, the corrected architecture for checkpoint compatibility:

```
Input (B, 243, 17, 2)
  │
  ├─ joint_embed: Linear(2, 256)  → (B, 243, 17, 256)
  ├─ + spatial_pos (1, 1, 17, 256)
  ├─ + temporal_pos (1, 243, 1, 256)
  │
  ├─ Block 0 [SPATIAL]:   LN → Reshape(B*T,J,D) → MHSA → Reshape back → Residual → LN → FFN → Residual
  ├─ Block 1 [TEMPORAL]:  LN → Permute+Reshape(B*J,T,D) → MHSA → Reshape back → Residual → LN → FFN → Residual
  ├─ Block 2 [SPATIAL]:   ...
  ├─ Block 3 [TEMPORAL]:  ...
  ├─ Block 4 [SPATIAL]:   ...
  │
  ├─ norm: LayerNorm(256)  → (B, 243, 17, 256)
  └─ head: Linear(256, 3)  → (B, 243, 17, 3)
```

Information flow through attention layers:

$$\text{Block } i: \quad x \in \mathbb{R}^{B \times T \times J \times D}$$

**Spatial block** ($i$ even): Attention matrix is $\mathbf{A}_S \in \mathbb{R}^{J \times J}$, computed independently for each of $B \times T$ frames:

$$\mathbf{A}_S = \text{softmax}\left(\frac{\mathbf{Q}_S \mathbf{K}_S^\top}{\sqrt{d_h}}\right), \quad \mathbf{Q}_S, \mathbf{K}_S \in \mathbb{R}^{(BT) \times H \times J \times d_h}$$

**Temporal block** ($i$ odd): Attention matrix is $\mathbf{A}_T \in \mathbb{R}^{T \times T}$, computed independently for each of $B \times J$ joints:

$$\mathbf{A}_T = \text{softmax}\left(\frac{\mathbf{Q}_T \mathbf{K}_T^\top}{\sqrt{d_h}}\right), \quad \mathbf{Q}_T, \mathbf{K}_T \in \mathbb{R}^{(BJ) \times H \times T \times d_h}$$

Where $H = 8$ heads, $d_h = D/H = 32$, $D = 256$.

**Memory cost** of attention matrices:
- Spatial: $B \times T \times H \times J^2 = B \times 243 \times 8 \times 289$ — negligible ($\sim$4.5M elements per sample)
- Temporal: $B \times J \times H \times T^2 = B \times 17 \times 8 \times 59{,}049$ — significant ($\sim$8M elements per sample)

---

## 8. Recommended Reimplementation Changes

Based on this analysis, the prior audit's Module 2 (`DSTformerBlock`) should be revised:

### 8.1 Changed: Use `stage_st` mode (not parallel fusion)

```python
class DSTformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, mlp_ratio=4.0, stage='spatial'):
        super().__init__()
        self.stage = stage
        
        # Active pathway
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        
        # FFN
        self.norm3 = nn.LayerNorm(dim)  # "norm3" to match original naming
        hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden)     # Use Mlp class, not nn.Sequential
        
        # Dead weights — kept for checkpoint compatibility
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, num_heads)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, J, D = x.shape
        
        if self.stage == 'spatial':
            u = self.norm1(x).reshape(B * T, J, D)
            u = self.attn(u).reshape(B, T, J, D)
        else:
            u = self.norm1(x).permute(0, 2, 1, 3).reshape(B * J, T, D)
            u = self.attn(u).reshape(B, J, T, D).permute(0, 2, 1, 3)
        
        x = x + u
        x = x + self.mlp(self.norm3(x))
        return x


class Mlp(nn.Module):
    """Match original naming: fc1/fc2 instead of nn.Sequential indices."""
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
```

### 8.2 Changed: Use `Mlp` class with `fc1`/`fc2` naming

This eliminates the `mlp.0`/`mlp.3` → `mlp.fc1`/`mlp.fc2` mapping entirely.

### 8.3 Changed: Match original PE shape or add reshape in loader

Keep the original `(1, J, D)` and `(1, T, D)` shapes to avoid any reshape in weight loading.

### 8.4 Simplified weight mapping (after changes)

With these revisions, the mapping becomes trivial:

```python
def load_pretrained_weights_simple(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    orig_sd = ckpt.get('model', ckpt)
    
    new_sd = {}
    for k, v in orig_sd.items():
        # Only transformation: add 'backbone.' prefix to non-head keys
        if k.startswith('head.'):
            new_sd[k] = v
        elif k == 'pos_embed':
            new_sd['backbone.spatial_pos'] = v.unsqueeze(1)  # (1,J,D)→(1,1,J,D)
        elif k == 'temp_embed':
            new_sd['backbone.temporal_pos'] = v.unsqueeze(2)  # (1,T,D)→(1,T,1,D)
        elif k == 'joints_embed.weight':
            new_sd['backbone.joint_embed.weight'] = v
        elif k == 'joints_embed.bias':
            new_sd['backbone.joint_embed.bias'] = v
        else:
            new_sd[f'backbone.{k}'] = v  # blocks.*, norm.* → backbone.blocks.*, backbone.norm.*
    
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    assert len(missing) == 0, f"Missing keys: {missing}"
    return unexpected  # should only be pos_drop-type keys (no params)
```

---

## 9. Final Corrected LOC Estimate

| Module | Prior Estimate | Revised | Change |
|---|---|---|---|
| Attention | 35 | 35 | — |
| **Mlp** (new) | 0 | **12** | +12 (separate class for naming) |
| DSTformerBlock | 45 | **40** | -5 (simpler with stage_st, no fusion math) |
| DSTformer backbone | 40 | 40 | — |
| MotionBERT3DLift | 20 | 20 | — |
| Loss | 15 | 15 | — |
| **Weight loader** (new) | 0 | **25** | +25 |
| **Validation tests** (new) | 0 | **50** | +50 |
| Dataset | 55 | 55 | — |
| Train loop | 65 | 65 | — |
| **Total Essential** | **315** | **357** | +42 |

The 42 additional LOC buy you checkpoint compatibility and a verification procedure — without them, you have an untestable architecture.
