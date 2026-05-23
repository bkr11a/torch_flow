# E0 - Sanity Check

## Experiment Metadata

#### Experiment ID

`HQS-EXP-001`

#### Experiment title:

#### Date and Time

DD/MM/YYYY - HH:MM:SS

#### MLFlow Link

#### MLFlow Experiment Name

#### MLFlow Run Name

#### MLFlow Artefact Store

#### **Repository**

#### **Branch**

#### **Commit Hash**

#### **Configuration file:**

`configs/...yaml`

#### **Hardware:**\nGPU model, number of GPUs, VRAM, CPU, RAM.

#### **Runtime:**\nTraining time, evaluation time, number of epochs/steps.

#### **Random seeds:**\nList all seeds used.

#### **Status:**\n`planned / running / completed / failed / superseded`

## Research Question

State the experiment as a precise question.

Example:

> Does replacing the full-flow residual
>
> $$
> I_x^k u^k + I_y^k v^k + I_t^k
> $$
>
> with the delta-flow residual
>
> $$
> I_t^k + I_x^k \Delta u^k + I_y^k \Delta v^k
> $$
>
> improve recurrent stability and endpoint error?

This section should be narrow. Avoid broad claims such as "Does physics improve optical flow?" The project documents explicitly warn that the paper should focus on one optical-flow energy, one update rule, one architecture, and one central claim.

## Hypothesis

Write a falsifiable hypothesis.

Example:

> **H1:** The delta-flow residual will reduce per-iteration oscillation and improve validation EPE relative to the full-flow residual because the data-consistency update is locally linearised around the current warped estimate.

Also include the null hypothesis:

> **H0:** There is no measurable difference between the full-flow and delta-flow residual formulations under matched training conditions.

## Claim Being Tested

Identify which paper-level claim this experiment supports.

Choose one or more:

| Claim | Evidence required |
|-------|-------------------|
| Accuracy | EPE, angular error, benchmark comparison |
| Stability | Perturbation tests, residual decrease, update magnitude behaviour |
| Cross-domain generalisation | Train on one dataset, test on another without fine-tuning |
| Data efficiency | Reduced training fractions and degradation curves |
| Interpretability | Intermediate flow, residual maps, auxiliary variables, update magnitudes |
| Efficiency | Runtime, memory, parameter count, recurrent iterations |

The work plan explicitly states that evaluation should go beyond endpoint error and should match the claimed benefits of operator structure.

## Mathematical Object Under Test

This section is essential for this project.

Write the relevant energy, operator, or update equation.

Example:

$$
\begin{align*}
\mathcal{E}(u,v) &= \mathcal{D}(I1​,I2​,u,v)+\lambda \mathcal{R}(u,v) \\
z^{k+1} &= \mathcal{T}_\theta^k​(z^k;I_1​,I_2​)
\end{align*}
$$

Then define the specific component tested.

Example:

$$
\begin{align*}
r_0^k​(x)&=I_2​(x+w^k(x))−I_1​(x) \\
r_{\text{lin}}^k​(x;\Delta w^k) &= r_0^k​(x)+\nabla I_2^k​(x)^{\mathsf{T}}\Delta w^k(x)
\end{align*}
$$

The report should state whether the experiment concerns the data-consistency operator $\mathcal{C}^{k}$, warping operator $\mathcal{W}^k$, proximal/regularisation operator $\mathcal{P}_{\theta, k}$, auxilarly state $q^k$, or learned correction block. The chosen architecture is meant to decompose the update as

$$
\mathcal{T}_{\theta}^k = \mathcal{P}_{\theta, k} \circ \mathcal{C}^k \circ \mathcal{W}^k,
$$

where warping, data-consistency correction, and learned proximal/regularisation have identifiable roles.

## Experimental Variables

#### Independent variable

What has changed?

Example:

| Variant | Description |
|---------|-------------|
| A       | Full-flow residual in HQS data step |
| B       | Delta-flow residual in HQS data step |

#### Dependent variables

What is measured?

Example:

| Metric | Meaning |
|--------|---------|
| Validation EPE | Accuracy |
| AE     | Angular error |
| Photometric residual per iteration | Data-consistency behaviour |
| $\|\Delta w^k\|$ per iteration | Stability of recurrent updates |
| Residual variance | Oscillation or instability |
| Runtime / memory | Efficiency |

#### Controlled variables

What is fixed?

Example:

* Same dataset split.
* Same model capacity.
* Same number of HQS iterations.
* Same training schedule.
* Same random seeds.
* Same augmentations.
* Same loss weights.
* Same evaluation code.

This is important because the required ablations include equal-capacity baselines; otherwise, improvements may be attributed to parameter count rather than operator structure.

## Model Variants

Describe each model precisely.

| Model ID | Description | Parameters | Structural difference |
|----------|-------------|------------|-----------------------|
| `M0`     | Baseline HQSFlow with full-flow residual | X M        | Original data step    |
| `M1`     | HQSFlow with delta-flow residual | X M        | Correct warped linearisation |
| `M2`     | No auxiliary variable qqq | X M        | Tests HQS state contribution |
| `M3`     | Generic CNN prox block | X M        | Tests proximal/operator structure |

For this project, the core ablations should include: equal-capacity unconstrained baseline, loss-only physics-regularised baseline, removing the data-consistency operator, replacing the proximal module with a generic CNN block, removing auxiliary variable $q$, varying recurrent iterations $K$, shared versus unshared weights, and removing photometric/regularisation losses during training.

## Dataset and Splits

**Training dataset:**\nExample: FlyingChairs / FlyingThings3D / Sintel clean / KITTI.

**Validation dataset:**\nExample: Sintel clean validation.

**Test dataset:**\nExample: KITTI or cross-domain dataset.

**Split details:**\nNumber of image pairs, resolution, filtering rules, excluded samples.

**Preprocessing:**\nResizing, cropping, normalisation, colour augmentation, geometric augmentation.

**Cross-domain setting:**\nExample:

> Train on FlyingChairs, evaluate on Sintel without fine-tuning.

**Reduced-data setting:**\nExample:

> Train with 100%, 50%, 25%, 10%, and 5% of the training set.

## Training Protocol

Report enough detail that the experiment is reproducible.

| Item | Value |
|------|-------|
| Optimiser | AdamW / Adam / SGD |
| Initial learning rate |       |
| Scheduler |       |
| Batch size |       |
| Number of epochs / steps |       |
| Loss terms |       |
| Loss weights |       |
| Sequence loss weighting |       |
| Number of recurrent iterations $K$ |       |
| HQS beta schedule |       |
| Lambda schedule |       |
| Correction gate schedule |       |
| Gradient clipping |       |
| Mixed precision | yes/no |
| Checkpoint selection | best validation EPE / final / averaged |

Also record any failure conditions:

* divergence,
* NaNs,
* exploding gradients,
* unstable update magnitudes,
* photometric residual increasing systematically,
* validation collapse after a certain epoch.

## Evaluation Protocol

Specify exactly how the model is evaluated.

**Primary metrics:**

$$
\text{EPE} = \frac{1}{\lvert \Omega \rvert}​ \sum_{x \in \Omega} \lvert\lvert w_{\text{pred}}​(x)−w_{\text{gt}}​(x)\rvert\rvert_2​
$$

**Secondary metrics:**

* EPE by motion magnitude.
* EPE near motion boundaries.
* EPE in occluded/non-occluded regions.
* Photometric residual.
* Smoothness energy.
* Runtime.
* Memory.
* Parameter count.

**Iterative metrics:**

For $k = 1, \dots, K$ report; $\text{EPE}(w^k)$, $\lvert \lvert w^{k+1} - w^{k} \rvert \rvert$, $\lvert\lvert r^k_0 \rvert \rvert$, $\lvert \lvert w^{k+1} - q^{k} \rvert \rvert$. The work plan specifically requires logging intermediate flow estimates, residual maps, and update magnitudes, because these support the interpretability and stability claims.

## Perturbation / Robustness Tests

Use this section when testing stability.

| Perturbation | Levels | Expected relevance |
|--------------|--------|--------------------|
| Gaussian noise | $\sigma$=0.01,0.03,0.05 | Sensor noise       |
| Brightness shift | ±10%, ±20% | Brightness constancy violation |
| Contrast change | 0.8×, 1.2× | Appearance variation |
| Blur         | kernel 3, 5, 7 | Motion blur / defocus |
| JPEG compression | quality 90, 70, 50 | Compression artefacts |
| Small geometric perturbation | minor scale/translation | Registration instability |

Report both raw performance and degradation relative to the clean case:

$$
\Delta \text{EPE} = \text{EPE}_{\text{perturbed}​} − \text{EPE}_{\text{clean}}​.
$$

## Results

#### Quantitative results

| Model | Dataset | EPE all | EPE matched | EPE unmatched | s0-10 | s10-40 | s40+ | d0-10 | d10-60 | d60-140 | f1  | Params | Runtime | Memory |
|-------|---------|---------|-------------|---------------|-------|--------|------|-------|--------|---------|-----|--------|---------|--------|
| Baseline |         |         |             |               |       |        |      |       |        |         |     |        |         |        |
| Proposed |         |         |             |               |       |        |      |       |        |         |     |        |         |        |

#### Per-iteration results

| Iteration $k$ | EPE | Photometric residual | Mean update norm | Mean $\lvert\lvert w^k−q^k \rvert\rvert$ |
|-------------|-----|----------------------|------------------|----------------------------------------|
| 0           |     |                      |                  |                                        |
| 1           |     |                      |                  |                                        |
| 2           |     |                      |                  |                                        |
| $K$         |     |                      |                  |                                        |

#### Ablation results

| Ablation | EPE | Robustness $\Delta$EPE | Interpretation |
|----------|-----|----------------------|----------------|
| Full model |     |                      |                |
| No auxiliary variable $q$ |     |                      | Tests HQS state |
| No data-consistency operator |     |                      | Tests explicit image consistency |
| Generic CNN prox |     |                      | Tests operator-structured prox |
| Loss-only physics |     |                      | Tests architecture vs loss regularisation |
| Unconstrained equal-capacity |     |                      | Tests structural restriction |

Each ablation table should include a short interpretation of what the ablation proves or fails to prove. This is explicitly required in the work plan.

## Qualitative Results

Include figures, not just tables.

Recommended panels:


 1. Image 1.
 2. Image 2.
 3. Ground-truth flow.
 4. Predicted flow.
 5. Error map.
 6. Warped image.
 7. Photometric residual map.
 8. Intermediate flows $w^1, \dots, w^{K}$
 9. Delta updates $\Delta w^1, \dots, \Delta w^K$
10. Auxiliary variable $q^k$ or coupling residual $w^k-q^k$.

For the HQSFlow paper, qualitative results should show whether intermediate states have interpretable roles such as residual reduction, flow refinement, or regularisation of spatial structure.

## Interpretation

This is the most important part of the report.

Answer:


1. Did the experiment support the hypothesis?
2. Was the improvement due to operator structure, or could it be due to capacity/training effects?
3. Which metric supports the claim most directly?
4. Which metric contradicts or weakens the claim?
5. Did the model fail in specific regimes?
6. Does the result justify a paper-level claim?

Example:

> The delta-flow residual reduced update oscillation across recurrent iterations and improved validation EPE by X%. The strongest evidence is the monotonic reduction in $\|r_0^k\|$ and lower $\|\Delta w^k\|$ variance. However, the improvement is not yet sufficient to support a general robustness claim because perturbation results have not been run.

## Failure Modes

Document failures explicitly.

| Failure mode | Evidence | Possible cause | Next diagnostic |
|--------------|----------|----------------|-----------------|
| EPE improves early then worsens | Per-iteration curve | recurrent drift | vary $K$, gate schedule |
| Residual decreases but EPE worsens | residual/EPE mismatch | brightness constancy violation | add occlusion mask or robust loss |
| Large errors near boundaries | error map | over-smoothing prox | edge-aware regularisation |
| Training unstable | gradient logs | data-step conditioning | clamp beta, inspect determinant |
| No ablation difference | table    | structure not active | inspect $w-q$, delta norms |

This section prevents overclaiming. The project notes warn against claiming theoretical convergence unless the learned operators satisfy explicit assumptions and against claiming physical correctness merely because photometric or smoothness losses are used.

## Threats to Validity

Include at least the following.

**Internal validity:**\nAre model variants actually equal capacity? Are seeds sufficient? Are training schedules identical?

**External validity:**\nDoes the result generalise beyond the chosen dataset?

**Construct validity:**\nDoes the chosen metric really measure the claim? For example, EPE alone does not establish interpretability.

**Implementation validity:**\nWas the residual implemented correctly? Was the flow channel convention `[dy, dx]` versus `[dx, dy]` checked? Was warping performed at the correct scale?

## Reproducibility Checklist

Include:

* Repository commit hash.
* Config file.
* Dataset version and path.
* Environment file.
* CUDA/PyTorch versions.
* Random seeds.
* Training command.
* Evaluation command.
* Checkpoint path.
* Log path.
* WandB/TensorBoard run ID.
* Exact code diff if testing a small change.

## Conclusion

Use a constrained conclusion.

Template:

> This experiment \[supports / does not support / partially supports\] the hypothesis that \[specific structural component\] improves \[specific behaviour\] under \[specific condition\]. The result justifies the following limited claim: \[claim\]. It does not yet justify claims about \[unsupported broader claim\].

Example:

> This experiment supports the claim that the delta-flow residual improves recurrent stability under the current training setting. It does not yet justify a cross-domain generalisation claim because no out-of-domain evaluation was performed.

## Next Experiments

List only experiments that follow logically.

Example:


1. Repeat with three random seeds.
2. Add equal-capacity unconstrained baseline.
3. Run perturbation test under brightness shift and blur.
4. Visualise residual maps and update magnitudes.
5. Test $K \in \{3,5,8,12\}$.

## Minimal One-Page Version

For quick reporting after each run:

```bash
Experiment ID:
Title:
Date:
Commit:
Config:
Dataset:
Seed(s):
Hardware:

Research question:
Hypothesis:
Model variants:
Controlled variables:

Mathematical component tested:
Metrics:
Training protocol:
Evaluation protocol:

Main quantitative result:
Main qualitative observation:
Failure modes:
Interpretation:
Conclusion:
Next experiment:
```

## Best-Practice Rule for This Project

Every experiment report should contain these five elements:


1. **The mathematical component being tested**\nExample: data-consistency residual, HQS auxiliary variable, proximal module, recurrent state.
2. **The architectural consequence**\nExample: whether the update uses $w^k$, $\Delta w^k$, $q^k$ or learned correction.
3. **The controlled comparison**\nExample: equal-capacity baseline, loss-only baseline, ablation.
4. **Evidence beyond EPE**\nExample: residual maps, update magnitudes, perturbation behaviour, cross-domain degradation.
5. **A narrow justified conclusion**\nExample: "This supports stability under brightness perturbation," not "physics-inspired learning works."