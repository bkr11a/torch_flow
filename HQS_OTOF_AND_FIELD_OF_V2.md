# HQS-OTOF and HQS-Field-OFv2

## Status and evidence boundary

This package adds two directly comparable optical-flow estimators to the
`torch_flow` `pgma` implementation line.

- `HQS-OTOF` is the controlled transplant model requested for the FlowIt-style
  measurement front end.
- `HQS-Field-OFv2` is the publication-oriented research model. It embeds
  optimal transport as a repeated latent measurement operator rather than only
  an initializer.

Both models are implemented research hypotheses. Static and algebraic checks
were executed in the packaging environment; tensor smoke and gradient tests
are supplied for the PyTorch workstation. These checks do not establish
benchmark accuracy or publication readiness. No architecture can ensure
acceptance at a tier-one venue.

FlowIt is a May 2026 arXiv preprint. The implementation in this package is
clean-room and based on its published mathematical description. It does not
copy FlowIt source code or reproduce its learned GRU refinement.

## Controlled model definitions

| Property | HQS-OTOF | HQS-Field-OFv2 |
|---|---|---|
| Hierarchical transformer features | Shared CNN pyramid, per-scale self/cross-transformers and content-gated cross-scale fusion | Identical |
| Initial global matching | All-pairs \(1/4\) correlation | Identical all-pairs \(1/4\) correlation |
| Initial assignment | Dustbin entropy-regularised Sinkhorn OT | Identical |
| Initial flow | Local expectation around OT peak | Identical |
| Confidence/observability | Local transported mass / total real transported mass | Identical |
| Initial-state bias | None | None |
| Coarse inverse update | Existing multi-hypothesis HQS-Field cells, seeded and globally conditioned by the static OT plan | Transport-moment analytic solve and symmetric semantic graph proximal |
| Re-acquisition | No repeated OT | Repeated full global OT at \(1/8\), with state weight \([0,.20,.45,.70]\) |
| Fine update | Existing local analytic mixture cells at \(1/4\) and \(1/2\) | Identical fine cells |
| Flow-residual bypass | None | None |
| Prediction schedule | \((2,4,2,2)\), ten predictions | Identical |

This comparison isolates the value of making transport part of the inverse
iteration:

\[
\text{HQS-OTOF}
=
\text{static OT measurement}
\rightarrow
\text{existing HQS field inference},
\]

\[
\text{HQS-Field-OFv2}
=
\text{OT measurement}
\leftrightarrow
\text{analytic data solve}
\leftrightarrow
\text{source-only field proximal}.
\]

## 1. HQS-OTOF

### Measurement front end

Transformer-enhanced features from scales \(1/4,1/8,1/16\) are projected,
resized and fused on the \(1/4\) grid:

\[
\mathbf g_i
=
\sum_{s\in\{4,8,16\}}
\omega_i^s
\mathcal P_s(\mathbf f_i^s),
\qquad
\sum_s\omega_i^s=1.
\]

The all-pairs score is

\[
C_{pq}
=
\frac{
\langle \bar{\mathbf g}_{1,p},\bar{\mathbf g}_{2,q}\rangle
}{T}.
\]

With a learned dustbin score \(a_\partial\), Sinkhorn solves the
dustbin-augmented entropy-regularised assignment:

\[
P^\star
=
\arg\min_{P\in\mathcal U_\partial}
-\langle P,C\rangle
+\varepsilon
\sum_{pq}P_{pq}(\log P_{pq}-1).
\]

The implementation computes the complete real-real score volume once, caches
it in FP16 on CUDA, and performs the Sinkhorn reductions plus plan decoding in
FP32 blocks. It therefore avoids recomputing dot products at every Sinkhorn
iteration and never stores both a full-precision score volume and full
transport plan simultaneously.

For the peak \(q_p^\star\) and local window \(\mathcal W(q_p^\star)\),

\[
\widehat q_p
=
\frac{
\sum_{q\in\mathcal W}P^\star_{pq}q
}{
\sum_{q\in\mathcal W}P^\star_{pq}+\epsilon
},
\qquad
\mathbf u_p^0=\widehat q_p-p.
\]

The transported-mass diagnostics are

\[
\Gamma_p
=
\sum_{q\in\mathcal W}P^\star_{pq},
\qquad
o_p
=
\sum_{q\in\Omega_2}P^\star_{pq},
\qquad
P^\star_{p\partial}=1-o_p.
\]

The local covariance supplies a positive-semidefinite \(2\times2\) precision.
The OT modes and observability also replace the original raw global top-\(K\)
mixture on the first \(1/16\) and \(1/8\) HQS iterations. The transport output
therefore affects both initialization and global measurement trust.

### Inverse updates

After initialization, HQS-OTOF retains the HQS-Field-OF analytic
mixture-majorisation update and source-conditioned graph proximal. FlowIt's
unrestricted GRU residual updater is not used.

HQS-OTOF is the correct experimental answer to:

> What happens if the first five FlowIt stages replace the weak global matcher,
> while the established inverse-problem iterations remain otherwise intact?

It is not, by itself, the intended paper contribution.

## 2. HQS-Field-OFv2

### Joint inverse formulation

HQS-Field-OFv2 represents correspondence, the data-consistent flow and the
regularised field as separate variables:

\[
\begin{aligned}
\min_{P,\mathbf w,\mathbf z}\quad&
-\langle P,C_\theta\rangle
+\varepsilon H(P)
+\Psi_\partial(P)\\
&+
\frac12\sum_{pq}P_{pq}
\left\|
\mathbf w_p-(q-p)
\right\|_{\Lambda_{pq}}^2\\
&+
\frac{\beta}{2}\|\mathbf w-\mathbf z\|^2
+\frac{\lambda}{2}\mathbf z^\mathsf T
L_\phi(I_1)\mathbf z .
\end{aligned}
\]

The initial \(1/4\) plan is state-independent. At the \(k\)-th \(1/8\)
retransport step the score becomes

\[
\widetilde C_{pq}^{\,k}
=
C_{pq}
-
\alpha_k
\frac{
\|q-(p+\mathbf z_p^k)\|^2
}{2\sigma^2},
\]

with

\[
(\alpha_0,\alpha_1,\alpha_2,\alpha_3)
=(0,.20,.45,.70).
\]

The hard invariant \(\alpha_0=0\) prevents the first plan from simply
confirming an incorrect initialization.

### Analytic transport data step

Local transport moments give a proposal \(\boldsymbol\mu_p\), confidence,
observability and full precision \(\Lambda_p\succeq0\). A learned calibrator may
alter only scalar support and log precision; it cannot predict a flow vector.
The update is

\[
\begin{bmatrix}
\Lambda_p
+(\beta_k+\tau_k)I
\end{bmatrix}
\Delta\mathbf w_p
=
-
\left[
\Lambda_p(\mathbf w_p^k-\boldsymbol\mu_p)
+\beta_k(\mathbf w_p^k-\mathbf z_p^k)
\right].
\]

When dustbin mass dominates, support and therefore measurement precision
approach zero. The data operator then reduces to damped HQS consensus rather
than inventing a correspondence.

### Source-semantic field proximal

At \(1/16\) and \(1/8\), learned source features define non-negative symmetric
semantic affinities \(A_{pq}=A_{qp}\ge0\). Therefore

\[
L_\phi=D-A\succeq0.
\]

Let \(A_k\) below denote the diagonal, support-dependent data anchor, not the
semantic adjacency \(A\).

The field subproblem is

\[
\left[
\beta_k A_k+\eta_k I+\lambda_kL_\phi(I_1)
\right]\mathbf z^{k+1}
=
\beta_k A_k\mathbf w^{k+1}
+\eta_k\mathbf z^k.
\]

Jacobi iterations solve this fixed-affinity positive-definite system. Target
features, correlations and target-derived motion vectors are not accepted by
the proximal interface. Fine scales use the established source-only local graph
to limit cost.

## Files

- `models/hqs_ot_components.py`: cached all-pairs scores, blockwise Sinkhorn
  and transport decoding,
  transport data solve, semantic graph and transport-field cell.
- `hqs_pytorch/customML/customModels/HQSOTOpticalFlow.py`: Model 1.
- `hqs_pytorch/customML/customModels/HQSFieldOpticalFlowV2.py`: Model 2.
- `configs/dropins/11_hqs_otof.yaml`: Model 1 configuration.
- `configs/dropins/12_hqs_field_of_v2.yaml`: Model 2 configuration.
- `tests/test_hqs_ot_models.py`: operator, configuration and forward tests.
- `smoketest_hqs_transport.py`: reduced and full-configuration smoke tests.
- `losses/flow_loss.py`: OT mixture, matchability and observability
  supervision.
- `engine/trainer.py`: model-declared matcher warm-up and operator-gain
  diagnostics.

## Apply

The incremental patch expects the previously supplied HQS-LM-OF and
HQS-Field-OF drop-ins to be present.

```bash
cd /home/bradrice/repos/torch_flow

git apply --check /path/to/HQS-OTOF-Field-OFv2-incremental.patch
git apply /path/to/HQS-OTOF-Field-OFv2-incremental.patch
```

The overlay archive can instead be extracted over the current `pgma`
workspace. It does not remove or overwrite HQSCore, HQS-LM-OF,
HQS-Field-OF or scene-flow paths.

## Verification

```bash
python -m pytest -q \
  tests/test_hqs_ot_models.py \
  tests/test_hqs_field.py \
  tests/test_hqs_lm.py \
  tests/test_import_and_logging.py

python smoketest_hqs_transport.py \
  --model hqs_otof \
  --device cuda

python smoketest_hqs_transport.py \
  --model hqs_field_of_v2 \
  --device cuda
```

Then exercise the complete channel widths and ten-stage schedules:

```bash
python smoketest_hqs_transport.py \
  --model hqs_otof \
  --device cuda \
  --full-config

python smoketest_hqs_transport.py \
  --model hqs_field_of_v2 \
  --device cuda \
  --full-config
```

The full smoke input is only \(32\times48\), so it validates construction and
the complete schedule, not realistic memory consumption.

## Initial stability gates

Use separate checkpoints. Neither model is checkpoint-compatible with
HQSCore, HQS-LM-OF or HQS-Field-OF.

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/11_hqs_otof.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_otof_stability \
  --stop-after-stage u01_chairs_baseline \
  training.num_steps=2000 \
  training.val_every=1000 \
  training.save_every=2000 \
  training.log_every=20 \
  data.batch_size=1 \
  val_data.batch_size=1 \
  mlflow.enabled=false
```

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/12_hqs_field_of_v2.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_field_of_v2_stability \
  --stop-after-stage u01_chairs_baseline \
  training.num_steps=2000 \
  training.val_every=1000 \
  training.save_every=2000 \
  training.log_every=20 \
  data.batch_size=1 \
  val_data.batch_size=1 \
  mlflow.enabled=false
```

The first 5,000 steps of a normal run are matcher warm-up. The training engine
now respects model-declared measurement prefixes; it no longer assumes the
global matcher is named `pgma`.

## Memory and runtime controls

At a \(320\times768\) crop, the \(1/4\) grid contains \(15{,}360\) tokens and
the conceptual real-real matrix contains \(235{,}929{,}600\) entries per
sample. The configured CUDA FP16 score cache is therefore approximately
472 MB per sample, before duals, features, gradients and solver state.
Consequently:

- keep both model batch sizes at one on the RTX 5090 initially;
- retain gradient checkpointing;
- do not increase the \(1/4\) crop before measuring peak CUDA memory;
- use gradient accumulation only after the single-sample path is stable;
- treat the \(1/4\) OT latency as an ablation variable, not a free component.

If memory is exceeded, reduce crop size. Do not silently change
`HQS-OTOF.ot_scale`; that would invalidate its model definition.

## Decisive experiment matrix

| ID | Measurement | Inverse update | Purpose |
|---|---|---|---|
| B0 | Original HQS-Field top-\(K\) | Existing HQS-Field | Current structured reference |
| O1 | Static \(1/4\) OT | Existing HQS-Field | `HQS-OTOF`; isolates the FlowIt-style front end |
| F2 | \(1/4\) OT + repeated \(1/8\) OT | Analytic transport data + semantic field proximal | `HQS-Field-OFv2`; proposed method |
| G2 | Same OT features/plans | Capacity-matched learned residual updater | Required control for the structure claim |
| R0 | Published FlowIt | Published refinement | External reference |

The package implements B0, O1 and F2. G2 remains a required experimental
control; it should reuse the same encoder and OT measurements and replace only
the structured updates.

For every scale and iteration, report:

\[
\Delta_{\mathrm{data}}
=
\operatorname{EPE}(\mathbf z^k)
-
\operatorname{EPE}(\mathbf w^{k+1}),
\]

\[
\Delta_{\mathrm{prox}}
=
\operatorname{EPE}(\mathbf w^{k+1})
-
\operatorname{EPE}(\mathbf z^{k+1}).
\]

The trainer records:

- `operator/data_epe_gain`;
- `operator/proximal_epe_gain`;
- `operator/data_epe_gain_s40_plus`;
- `operator/proximal_epe_gain_unmatched`;
- mean OT confidence, observability, dustbin mass and entropy.

Also compute transport recall within 1, 2 and 4 pixels, stratified by
`s0_10`, `s10_40`, `s40_plus`, visible/unmatched and boundary/interior regions.

## Decision rules

1. Stop if either model is non-finite during the 2,000-step gate.
2. At 10,000 steps, require materially better \(s40+\) initialization EPE and
   top-\(K\) recall than HQS-Field-OF.
3. Require positive mean data-step gain in visible regions.
4. Require positive proximal gain in unmatched/low-support regions without a
   boundary-EPE regression.
5. Compare F2 against G2 with identical features, OT settings, data, steps and
   parameter budget.
6. Do not claim an inverse-formulation benefit if F2 only improves because it
   has more transport solves, parameters or compute.

## Publication assessment

`HQS-OTOF` is an attributed component transplant and a necessary baseline. A
paper whose only contribution is “FlowIt initialization followed by HQS” is
unlikely to satisfy a tier-one novelty threshold.

`HQS-Field-OFv2` has a defensible methodological distinction:

- correspondence is a repeated latent variable;
- unmatched mass explicitly switches off the data operator;
- motion enters only through closed-form data updates;
- the field prior is source-only and positive semidefinite;
- contribution of each operator is measurable.

Tier-one potential depends on evidence that this structure improves at least
one consequential axis relative to a capacity-matched learned updater:
large-motion accuracy, unmatched-region completion, calibration,
cross-dataset generalisation, data efficiency or robustness. Interpretability
alone will not compensate for a large benchmark accuracy gap.

## Primary external references

- Safadoust et al., *FlowIt: Global Matching via Hierarchical Transformers and
  Optimal Transport for Optical Flow*, arXiv:2603.28759v2, 2026:
  https://arxiv.org/abs/2603.28759
- Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*,
  NeurIPS 2013:
  https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport
- Xu et al., *GMFlow: Learning Optical Flow via Global Matching*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.html
- Teed and Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical
  Flow*, ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf
