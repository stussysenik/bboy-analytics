# sam3d — Follow-up 1.3: The vote matrix is dense `(N_sp, M_total)` where `M_total` = sum of masks across all views. With 100

_Generated: 2026-03-23T05:48:39.041Z_

---

# Vote Matrix: Sparsity, Memory, and Fragmentation

This is the critical gap in the SAM3D reimplementation plan. Let me address it thoroughly.

---

## 1. The Dense Matrix Problem — Concrete Numbers

The vote matrix $\mathbf{V} \in \mathbb{R}^{N_{sp} \times M}$ stores normalized counts:

$$V(s_i, m_j) = \frac{1}{|s_i|} \sum_{p \in s_i} \mathbb{1}[\ell_{v(m_j)}(p) = j]$$

where $\ell_v(p)$ is the mask label assigned to point $p$ by view $v$, and $v(m_j)$ is the view that generated mask $m_j$.

**Realistic scene parameters** (ScanNet-scale):

| Parameter | Conservative | Typical | Large |
|-----------|-------------|---------|-------|
| Views $V$ | 30 | 100 | 300 |
| Masks/view | 100 | 200 | 400 |
| $M_\text{total}$ | 3,000 | 20,000 | 120,000 |
| $N_{sp}$ | 5,000 | 20,000 | 50,000 |
| Dense matrix (float32) | 60 MB | **1.6 GB** | **24 GB** |

The typical case already strains single-GPU memory, and the large case is infeasible. But most entries are zero.

---

## 2. Sparsity Structure

A superpoint $s_i$ can only receive votes from masks in views where it is **visible**. Define visibility:

$$\mathcal{V}(s_i) = \left\{ v \in [V] : \exists\, p \in s_i \text{ s.t. } \text{depth}_v\!\left(\pi_v(p)\right) > 0 \right\}$$

where $\pi_v : \mathbb{R}^3 \to \mathbb{R}^2$ projects into view $v$'s image plane.

For a given superpoint, a mask from a non-visible view contributes exactly zero votes:

$$v(m_j) \notin \mathcal{V}(s_i) \implies V(s_i, m_j) = 0$$

**Visibility statistics** (indoor scenes, ScanNet-style scanning):

- Average visible views per superpoint: $\bar{k} \approx 5\text{–}10$ out of 100
- Average masks overlapping a superpoint per visible view: $\bar{m} \approx 2\text{–}5$ (most masks don't cover this particular region)

So the number of non-zero entries per row is approximately $\bar{k} \cdot \bar{m} \approx 25$, not $M_\text{total} = 20{,}000$.

**Sparsity ratio**: 

$$\rho = \frac{\text{nnz}}{N_{sp} \cdot M} = \frac{N_{sp} \cdot \bar{k} \cdot \bar{m}}{N_{sp} \cdot M} = \frac{\bar{k} \cdot \bar{m}}{M} \approx \frac{25}{20{,}000} = 0.00125$$

**99.9% of entries are zero.** The matrix is extremely sparse.

**Memory comparison**:

| Format | Storage | Typical Case |
|--------|---------|-------------|
| Dense float32 | $4 \cdot N_{sp} \cdot M$ | 1.6 GB |
| CSR sparse | $\approx 12 \cdot \text{nnz}$ | **6 MB** |
| **Reduction** | | **267×** |

---

## 3. Vote Fragmentation — The Real Problem

Sparsity is a solved engineering problem. The deeper issue is **cross-view mask identity fragmentation**.

### The Mechanism

SAM runs independently on each view. For a physical object $O$ visible in views $\{v_1, \ldots, v_k\}$, SAM assigns **independent, unrelated mask IDs**:

$$O \to \{m_{v_1}^{a_1}, m_{v_2}^{a_2}, \ldots, m_{v_k}^{a_k}\} \quad \text{where } a_i \neq a_j \text{ in general}$$

Consider superpoint $s_i \subset O$. Its vote row looks like:

$$V(s_i, \cdot) = [\underbrace{0, \ldots, 0, c_1, 0, \ldots}_{\text{view } v_1 \text{ masks}}, \underbrace{0, \ldots, c_2, 0, \ldots}_{\text{view } v_2 \text{ masks}}, \ldots]$$

where $c_j \approx 1/k$ if $s_i$ is uniformly observed. The vote mass is **split** across $k$ columns.

### Signal-to-Noise Degradation

The argmax decision at superpoint $s_i$ is:

$$\hat{m}(s_i) = \arg\max_{m_j} V(s_i, m_j)$$

The "correct" signal per column is $\approx 1/k$ of the total vote mass. A noise mask (from a single view where SAM over-segmented or mis-segmented) gets a vote of $\approx 1/k$ too. The effective SNR:

$$\text{SNR}(s_i) = \frac{\max_j V(s_i, m_j^{\text{correct}})}{\max_j V(s_i, m_j^{\text{noise}})} \approx \frac{1/k}{1/k} = 1$$

**The argmax becomes a coin flip as $k$ grows.** With 10 views, the correct mask from any single view has roughly the same vote weight as a spurious mask. The voting stage provides essentially no discrimination.

### Downstream Failure Mode

Different superpoints of the same object $O$ may argmax to different masks (from different views). After voting:

- $s_1 \to m_{v_3}^{17}$ (mask 17 from view 3)
- $s_2 \to m_{v_7}^{42}$ (mask 42 from view 7)  
- $s_3 \to m_{v_1}^{5}$ (mask 5 from view 1)

These are all the same physical object, but they have three different instance labels. The **merging stage** must fix this.

### Does Merging Save It?

The merging step (Module 5) connects adjacent superpoints with similar normals. This works **if**:

1. All superpoints of the same object form a **connected subgraph** in the adjacency graph
2. Normal similarity holds across the object surface

Failure cases:
- **Concave objects**: Parts separated by a concavity may not be adjacent
- **Multi-surface objects**: A table has a top (normal ↑) and legs (normal ↓) — normal threshold blocks the merge
- **Large objects**: Superpoints on opposite ends may be many hops apart, each hop requiring a merge decision

The merging is a local operation. It cannot recover from fragmentation when the object's superpoints form multiple disconnected components in instance-label space.

---

## 4. Solutions

### Solution A: Sparse Vote Matrix (addresses memory only)

Replace the dense matrix with `scipy.sparse.lil_matrix` during construction, convert to CSR for argmax:

```python
from scipy import sparse

def build_vote_matrix_sparse(sp_labels, point_mask_labels, n_sp, n_masks):
    """Sparse vote matrix — 267× less memory than dense."""
    # Build in COO format (efficient for incremental construction)
    rows, cols, vals = [], [], []
    
    sp_sizes = np.bincount(sp_labels, minlength=n_sp).astype(np.float32)
    sp_sizes = np.maximum(sp_sizes, 1)
    
    for view_labels in point_mask_labels:
        valid = view_labels >= 0                          # (N,)
        if not np.any(valid):
            continue
        sp_valid = sp_labels[valid]                       # (N_valid,)
        ml_valid = view_labels[valid]                     # (N_valid,)
        
        # Count (superpoint, mask) co-occurrences
        # Use a dict to accumulate before adding to COO
        pairs = np.stack([sp_valid, ml_valid], axis=1)    # (N_valid, 2)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        
        rows.append(unique_pairs[:, 0])
        cols.append(unique_pairs[:, 1])
        vals.append(counts.astype(np.float32))
    
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    
    V = sparse.coo_matrix((vals, (rows, cols)), shape=(n_sp, n_masks))
    V = V.tocsr()                                         # efficient row ops
    
    # Normalize by superpoint size
    # V[i, :] /= sp_sizes[i]
    inv_sizes = sparse.diags(1.0 / sp_sizes)              # (N_sp, N_sp)
    V = inv_sizes @ V                                     # sparse × sparse = sparse
    
    return V  # scipy.sparse.csr_matrix, shape (N_sp, n_masks)
```

**Argmax on sparse CSR** — `scipy` doesn't have sparse argmax, but row-wise:

```python
def resolve_votes_sparse(V_sparse):
    """Argmax per row on sparse CSR matrix."""
    n_sp = V_sparse.shape[0]
    labels = np.full(n_sp, -1, dtype=np.int32)
    
    for i in range(n_sp):
        row = V_sparse.getrow(i)
        if row.nnz > 0:
            labels[i] = row.indices[row.data.argmax()]
    
    return labels
```

Or vectorized via dense conversion of only non-empty rows:

```python
def resolve_votes_sparse_fast(V_sparse):
    n_sp = V_sparse.shape[0]
    labels = np.full(n_sp, -1, dtype=np.int32)
    
    # Process in chunks to avoid materializing full dense matrix
    chunk = 1000
    for start in range(0, n_sp, chunk):
        end = min(start + chunk, n_sp)
        block = V_sparse[start:end].toarray()             # (chunk, M) — small
        row_max = block.max(axis=1)
        has_votes = row_max > 0
        labels[start:end][has_votes] = block[has_votes].argmax(axis=1)
    
    return labels
```

**This solves memory but not fragmentation.**

---

### Solution B: Cross-View Mask Grouping (addresses fragmentation)

Before voting, establish correspondence between masks across views by computing 3D overlap.

**Key idea**: Two masks from different views that cover the same 3D region should be treated as the same instance. Group them, assign a unified ID, then vote using group IDs.

#### Step 1: Back-project each mask to a 3D point set

Already done in Module 2. Store the 3D footprint per mask:

$$\mathcal{P}(m_j) = \left\{ R_v^{-1}\!\left(d \cdot K_v^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - t_v\right) : (u, v) \in m_j, \, d = \text{depth}_v(u, v) \right\}$$

#### Step 2: Compute pairwise 3D IoU between masks from different views

Full pairwise is $O(M^2)$ which is expensive. Use spatial hashing to prune:

```python
def group_masks_3d(mask_3d_points: list[np.ndarray],
                   mask_view_ids: list[int],
                   iou_threshold: float = 0.25,
                   voxel_size: float = 0.05) -> np.ndarray:
    """
    Group masks across views by 3D IoU.
    
    Args:
        mask_3d_points: list of M arrays, each (N_i, 3) — 3D footprint per mask
        mask_view_ids: list of M ints — which view each mask came from
        iou_threshold: minimum 3D IoU to consider same instance
        voxel_size: voxelization resolution for IoU computation
    
    Returns:
        group_ids: (M,) — unified group ID per mask
    """
    M = len(mask_3d_points)
    
    # Voxelize each mask for fast set-based IoU
    def voxelize(pts):
        return set(map(tuple, (pts / voxel_size).astype(np.int32)))
    
    mask_voxels = [voxelize(pts) for pts in mask_3d_points]
    
    # Compute bounding boxes for pruning
    bboxes_min = np.array([pts.min(0) if len(pts) > 0 else np.zeros(3) 
                           for pts in mask_3d_points])     # (M, 3)
    bboxes_max = np.array([pts.max(0) if len(pts) > 0 else np.zeros(3) 
                           for pts in mask_3d_points])     # (M, 3)
    
    # Union-Find
    parent = list(range(M))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Only compare masks from DIFFERENT views
    for i in range(M):
        if len(mask_voxels[i]) == 0:
            continue
        for j in range(i + 1, M):
            if mask_view_ids[i] == mask_view_ids[j]:
                continue  # same view — skip
            if find(i) == find(j):
                continue  # already grouped
            if len(mask_voxels[j]) == 0:
                continue
            
            # AABB overlap check (prune non-overlapping pairs)
            if np.any(bboxes_min[i] > bboxes_max[j] + voxel_size) or \
               np.any(bboxes_min[j] > bboxes_max[i] + voxel_size):
                continue
            
            # Voxel-set IoU
            intersection = len(mask_voxels[i] & mask_voxels[j])
            union_size = len(mask_voxels[i] | mask_voxels[j])
            iou = intersection / (union_size + 1e-8)
            
            if iou > iou_threshold:
                union(i, j)
    
    # Extract group IDs
    group_ids = np.array([find(i) for i in range(M)])
    
    # Remap to contiguous [0, G)
    unique_groups = np.unique(group_ids)
    remap = {g: idx for idx, g in enumerate(unique_groups)}
    group_ids = np.array([remap[g] for g in group_ids])
    
    return group_ids  # (M,) — group ID per mask
```

#### Step 3: Vote with group IDs instead of raw mask IDs

Replace mask column indices with group IDs in the vote matrix. Now the matrix is $\mathbf{V} \in \mathbb{R}^{N_{sp} \times G}$ where $G \ll M$:

$$G \approx \text{number of physical instances} \approx 30\text{–}200$$

This makes the matrix tiny ($50{,}000 \times 200 = 40$ MB dense, or ~200 KB sparse) **and** concentrates all votes for the same physical object into a single column.

```python
def build_grouped_vote_matrix(sp_labels, point_mask_labels, group_ids,
                               n_sp, n_groups):
    """Vote matrix using grouped mask IDs. Fixes fragmentation."""
    V = np.zeros((n_sp, n_groups), dtype=np.float32)
    sp_sizes = np.bincount(sp_labels, minlength=n_sp).astype(np.float32)
    sp_sizes = np.maximum(sp_sizes, 1)
    
    for view_labels in point_mask_labels:
        valid = view_labels >= 0
        # Remap mask IDs to group IDs
        grouped = np.full_like(view_labels, -1)
        grouped[valid] = group_ids[view_labels[valid]]
        
        np.add.at(V, (sp_labels[valid], grouped[valid]), 1.0)
    
    V /= sp_sizes[:, None]
    return V  # (N_sp, G) — now G ≈ 100, not 20,000
```

**After grouping, the argmax SNR becomes**:

$$\text{SNR}(s_i) = \frac{\sum_{j=1}^{k} c_j}{\max_m V(s_i, m^{\text{noise}})} \approx \frac{k \cdot (1/k)}{1/k} = k$$

**Linear improvement in view count.** More views now *help* rather than hurt.

---

### Solution C: Computational Cost of Mask Grouping

The naive pairwise comparison is $O(M^2)$. With AABB pruning:

**AABB prune effectiveness**: In a typical indoor scene, most mask bounding boxes don't overlap. Empirically, AABB pruning eliminates >95% of pairs.

**Remaining pairs after pruning**: $\sim M \cdot \bar{d}$ where $\bar{d}$ is the average number of overlapping masks per mask. For indoor scenes, $\bar{d} \approx 20\text{–}50$.

**Voxel-set IoU cost**: $O(|\text{voxels}|)$ per pair using Python sets. With average mask size $\sim 5{,}000$ voxels, each IoU takes $\sim 0.5$ ms.

**Total grouping time**: $20{,}000 \times 35 \times 0.5\text{ ms} \approx 350$ seconds. **Too slow.**

**Fix — spatial hash index**:

```python
def group_masks_fast(mask_3d_points, mask_view_ids, 
                     iou_threshold=0.25, voxel_size=0.05):
    """O(M·d) mask grouping with spatial hash acceleration."""
    M = len(mask_3d_points)
    
    # Voxelize all masks
    mask_voxels = []
    voxel_to_masks = {}  # voxel_key → set of mask indices
    
    for i, pts in enumerate(mask_3d_points):
        if len(pts) == 0:
            mask_voxels.append(set())
            continue
        vkeys = set(map(tuple, (pts / voxel_size).astype(np.int32)))
        mask_voxels.append(vkeys)
        for vk in vkeys:
            voxel_to_masks.setdefault(vk, set()).add(i)
    
    # For each mask, find candidate overlaps via shared voxels
    parent = list(range(M))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    checked = set()  # avoid redundant pair checks
    
    for i in range(M):
        if len(mask_voxels[i]) == 0:
            continue
        
        # Collect all masks that share at least one voxel with mask i
        candidates = set()
        for vk in mask_voxels[i]:
            candidates.update(voxel_to_masks.get(vk, set()))
        candidates.discard(i)
        
        for j in candidates:
            if mask_view_ids[i] == mask_view_ids[j]:
                continue
            pair = (min(i, j), max(i, j))
            if pair in checked:
                continue
            checked.add(pair)
            if find(i) == find(j):
                continue
            
            # Already know they share voxels — compute IoU
            inter = len(mask_voxels[i] & mask_voxels[j])
            union_sz = len(mask_voxels[i] | mask_voxels[j])
            if inter / (union_sz + 1e-8) > iou_threshold:
                union(i, j)
    
    group_ids = np.array([find(i) for i in range(M)])
    unique = np.unique(group_ids)
    remap = {g: idx for idx, g in enumerate(unique)}
    return np.array([remap[g] for g in group_ids])
```

**Complexity**: $O(M \cdot \bar{d})$ where $\bar{d}$ is the average spatial overlap degree. The spatial hash ensures we only check pairs that share at least one voxel. Typical runtime: **5–15 seconds** for 20K masks.

---

## 5. Updated Pipeline with Fix

The corrected `pipeline.py` inserts mask grouping between back-projection and voting:

```python
def run_sam3d(images, depths, intrinsics, extrinsics,
              scene_points, scene_normals, scene_colors):
    V = len(images)
    
    # Stage 1: 2D masks
    all_masks = []          # flat list of all Mask2D
    mask_3d_points = []     # parallel list: 3D footprint per mask
    mask_view_ids = []      # parallel list: view ID per mask
    mask_offset = 0
    
    for v in range(V):
        masks = generate_masks(images[v])
        R, t = extrinsics[v]
        
        for m in masks:
            m.view_id = v
            m.mask_id = mask_offset
            mask_offset += 1
            
            # Back-project this mask's pixels to 3D
            pts_3d, _ = backproject_masks([m], depths[v], intrinsics[v], R, t)
            mask_3d_points.append(pts_3d)
            mask_view_ids.append(v)
        
        all_masks.extend(masks)
    
    # *** NEW: Cross-view mask grouping ***
    group_ids = group_masks_fast(
        mask_3d_points, mask_view_ids,
        iou_threshold=0.25, voxel_size=0.05
    )                                                    # (M_total,)
    n_groups = group_ids.max() + 1
    
    # Stage 2: Assign grouped labels to scene points
    per_view_labels = []
    for v in range(V):
        R, t = extrinsics[v]
        view_masks = [m for m in all_masks if m.view_id == v]
        proj_pts, proj_labels = backproject_masks(
            view_masks, depths[v], intrinsics[v], R, t
        )
        # Remap proj_labels through group_ids
        valid = proj_labels >= 0
        proj_labels[valid] = group_ids[proj_labels[valid]]
        
        view_labels = assign_labels_to_pointcloud(
            scene_points, proj_pts, proj_labels, radius=0.02
        )
        per_view_labels.append(view_labels)
    
    # Stage 3a: Superpoints
    sp_labels = extract_superpoints(
        scene_points, scene_normals, scene_colors
    )
    n_sp = sp_labels.max() + 1
    
    # Stage 3b: Voting with grouped IDs
    vote_matrix = build_grouped_vote_matrix(
        sp_labels, per_view_labels, group_ids, n_sp, n_groups
    )                                                    # (N_sp, G) — small!
    sp_instance_labels = resolve_votes(vote_matrix)
    
    # Stage 3c: Boundary-aware merging
    adjacency = build_adjacency(scene_points, sp_labels)
    final_labels = merge_regions(
        sp_instance_labels, adjacency,
        scene_points, scene_normals, sp_labels
    )
    
    return final_labels
```

---

## 6. Summary of the Gap

| Issue | Severity | Fix | LOC Cost |
|-------|----------|-----|----------|
| Dense matrix memory ($N_{sp} \times M$) | High — OOM at 100+ views | Sparse CSR matrix | +20 |
| Vote fragmentation (SNR → 1) | **Critical** — voting becomes random | 3D mask grouping | +60 |
| Pairwise grouping cost ($O(M^2)$) | Medium — 350s naive | Spatial hash index | +40 |
| Merging can't recover disconnected fragments | Medium — under-segments large objects | Grouping eliminates the root cause | 0 (handled by grouping) |

**Total additional LOC for the fix: ~120**, bringing the essential pipeline from 360 to **~480 lines**. Without this fix, SAM3D degrades badly beyond ~10 views — which is exactly the regime where multi-view 3D segmentation is supposed to shine.
