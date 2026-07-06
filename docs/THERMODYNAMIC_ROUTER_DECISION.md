# Thermodynamic Router v5 Decision

## Verdict

Do not merge the Kim-style specific-heat router into ExFusion v5 as a production
compute-saving router.

The softmax-as-Gibbs identity is legitimate, but the proposed routing
implementation computes:

```text
energy_scores = q @ k.T
shape = (B, H, L, L)
```

That is the full quadratic attention score matrix. If a token is then routed to
FNet or SSM, the system has already paid the expensive part of attention just to
decide not to use attention.

## Why It Fails The ExFusion Objective

ExFusion routing must reduce compute. A router whose decision requires full
attention scores has this cost profile:

```text
total_cost = qk_attention_scores + router_statistics + selected_path
```

For any token routed away from attention:

```text
saved_attention_cost ~= 0
extra_cost > 0
```

So the claimed saving inverts into overhead.

## What Is Still Useful

The Cv statistic can still be tested as an offline diagnostic:

- Does Cv correlate with token-level loss?
- Does Cv correlate with entropy, long-range dependency resolution, or merge
  error?
- Does Cv predict where ExFusion should retain attention-heavy capacity?
- Does it identify calibration examples where bilinear cross-terms dominate
  merge error?

That makes it an ablation feature, not a production gate.

## Required Gates Before Any v5 Integration

1. **FLOPs accounting**
   - Compare against entropy router, hidden-variance router, and a learned MLP
     difficulty head.
   - Report prefill and decode separately.

2. **Correlation study**
   - Cv vs token loss.
   - Cv vs attention entropy.
   - Cv vs merge residual error.
   - Cv vs bilinear cross-term error.

3. **Ablation**
   - `hidden_variance`
   - `attention_entropy`
   - `full_cv`
   - `local_window_cv`
   - `learned_router`

4. **Strict acceptance bar**
   - Do not accept unless quality improves at equal latency or latency improves
     at equal quality.
   - Do not accept if it requires full `(L, L)` attention scores before routing
     in the path that is supposed to avoid attention.

## Replacement Direction

For production v5, prioritize cheap pre-attention signals:

- hidden-state variance,
- token embedding novelty,
- residual norm,
- cheap local-window attention entropy,
- routing history,
- learned difficulty head trained against token loss and merge residuals.

If thermodynamic language is kept in the docs, describe it as an analogy or
diagnostic reparameterization, not as proof of new physical necessity.
