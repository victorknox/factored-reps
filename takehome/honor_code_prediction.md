# Pre-Registered Prediction: Geometry of Residual Stream Activations

**Date:** 2026-03-15 (written BEFORE running post-training geometry analysis)
**Status:** Pre-registered. This document will not be retroactively edited.

---

## 1. Exact Dataset Design

- **K = 3** Mess3 ergodic components with parameters:
  - C0_slow: x=0.08, a=0.75 (sticky, slow mixing)
  - C1_mid: x=0.15, a=0.55 (moderate dynamics)
  - C2_fast: x=0.25, a=0.40 (fast mixing, diffuse)
- **Mixture weights:** uniform (1/3 each)
- **Sequence length:** 16 tokens
- **Each sequence** is generated entirely by one component (non-ergodic at sequence level)
- **Vocabulary:** {0, 1, 2} (3 tokens, shared across all components)
- **Training set:** 50,000 sequences; **Validation:** 5,000

## 2. Model Architecture

- Attention-only transformer (no MLP blocks)
- d_model=128, n_heads=4, n_layers=4
- Context window: 15 (input length for next-token prediction)
- Pre-norm with LayerNorm
- Trained with Adam, lr=3e-4, cosine schedule, 200 epochs

---

## 3. Mathematical Prediction

### Sufficient statistics for optimal prediction

The optimal next-token predictor needs the predictive distribution:

$$p(x_{t+1} | x_{1:t}) = \sum_{c=1}^{K} p(c | x_{1:t}) \sum_{s=1}^{3} p(s_t = s | c, x_{1:t}) \cdot p(x_{t+1} | s_t = s, c)$$

This requires the **joint posterior** $p(c, s_t | x_{1:t})$, which is a distribution over $K \times 3 = 9$ joint states. Since it sums to 1, it lives in an **8-dimensional simplex** $\Delta^8$.

However, the sufficient statistic decomposes as:
- $p(c | x_{1:t})$ — a vector in $\Delta^{K-1} = \Delta^2$
- $p(s_t | c, x_{1:t})$ — three vectors, each in $\Delta^2$, one per component

The joint is:
$$p(c, s_t | x_{1:t}) = p(c | x_{1:t}) \cdot p(s_t | c, x_{1:t})$$

### Predicted geometry

**Primary prediction:** The residual stream at later layers and later context positions will represent an approximation of the joint posterior $p(c, s_t | x_{1:t})$, embedded approximately linearly in a subspace of dimension ≤ 8 (the simplex dimension of the 9-state joint space).

**Key structural predictions:**

1. **Cluster structure by component:** At later positions (where the posterior over component identity concentrates), activations will form K=3 roughly separated clusters, one per component. Within each cluster, the geometry will reflect the component-specific belief simplex.

2. **Hierarchical encoding across layers:**
   - **Early layers** (embedding, block 0): Primarily encode token identity and simple positional information. Limited separation by component.
   - **Middle layers** (blocks 1-2): Component identity begins to separate. The representation starts reflecting $p(c | x_{1:t})$.
   - **Late layers** (blocks 3, ln_final): Both component identity and within-component belief are well-represented. The geometry approximates the full joint posterior.

3. **Context position dependence:**
   - **Position 0 (first token):** Little to no component discrimination possible. All components share vocab. Geometry is collapsed/uniform.
   - **Positions 1-5:** Gradual component discrimination emerges as token statistics differ. The belief simplex structure within each component starts to appear.
   - **Positions 6-15:** Component identity is largely resolved. Activations concentrate onto component-specific belief sub-manifolds. The representation effectively becomes a "union of simplices" rather than a full 8-simplex.

4. **Effective dimensionality:**
   - Early positions: low dimensionality (≤ 3-4 principal components for 95% variance)
   - Late positions: higher dimensionality (up to 5-8) reflecting the full joint space
   - But probably not full 8D because the posterior typically concentrates on one component

### Quantitative predictions

- **Linear probe for component identity:** R² should increase with both layer depth and context position. At the final layer and late positions, classification accuracy should exceed 90%.
- **Linear probe for within-component belief:** R² should be lower at early positions and layers, but reach moderate-to-high values (0.6-0.9) at the final layer for late positions.
- **PCA dimensionality:** For the final layer at late positions, 95% variance should be captured in 4-7 dimensions, reflecting the effective dimension of the joint posterior.

## 4. Intuitive Prediction

I expect the transformer to solve this task hierarchically:
1. First figure out "which world am I in" (coarse component identity)
2. Then track "where am I within this world" (fine-grained belief)

This is analogous to how language models might first identify genre/register, then track topic-specific context. The geometry should look like 3 "blobs" in PCA space, with internal structure visible when zooming into each blob.

The sticky component (C0) should be most easily discriminated because its token sequences are most distinctive. The fast-mixing component (C2) should produce the most "spread out" belief geometry because its beliefs change rapidly.

## 5. Plausible Alternative Geometries

### Alternative A: Full 8-simplex without clustering
The model might represent the full joint posterior as a single connected manifold in ~8 dimensions, without sharp clustering by component. This would happen if the model doesn't learn a hierarchical decomposition but instead directly maps to the full joint space.

### Alternative B: Collapsed low-dimensional representation
If the model finds that component identity alone (without fine-grained belief tracking) is nearly sufficient for good prediction, it might collapse to a ~2-3 dimensional representation that primarily encodes $p(c | x_{1:t})$. Within-component beliefs might be poorly encoded if they don't contribute much to predictive accuracy.

### Alternative C: Token-statistics encoding rather than belief geometry
The model might encode raw token count statistics (e.g., frequency of each token so far) rather than the Bayesian posterior. This would produce a different geometry that correlates with beliefs but doesn't directly match the simplex structure.

### Alternative D: Non-linear manifold structure
The representation might be highly non-linear, with the simplex structure only recoverable through non-linear methods. Linear probes would show low R², while non-linear methods would show the structure exists but is encoded non-linearly.

---

## 6. What I will check

1. PCA and cumulative explained variance by layer and position
2. PCA scatter plots colored by component identity and by belief coordinates
3. Linear probes for component posterior, within-component belief, joint belief, and next-token distribution
4. Probe R² as a function of layer depth and context position
5. Per-component PCA to check for within-component simplex-like structure
6. Heatmaps of emergence timing across layers × positions

**I will report results honestly regardless of whether they match these predictions.**
