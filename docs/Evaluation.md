# Evaluation Metrics

This document defines the no-ground-truth evaluation metrics used in the full
image-set scripts.

The goal is to measure whether an attribution map is faithful to the model's
own behavior, without requiring human segmentation masks or bounding-box
ground truth.

## Notation

Let:

- `x` be the input image.
- `b` be the background/replacement image.
- `c` be the explained class.
- `f_c(x)` be the model score for class `c`.
- `a_i` be the attribution value for pixel or region `i`.
- `d` be the number of pixels/regions.
- `m in {0, 1}^d` be a binary mask, where `1` means keep the original image and
  `0` means use the background.

The masked model score is:

```math
v(m) = f_c(x \odot m + b \odot (1 - m)).
```

The full-image and background-only scores are:

```math
f_S = v(\mathbf{1}), \qquad f_0 = v(\mathbf{0}).
```

For a fraction `alpha in [0, 1]`, define `T_alpha` as the binary mask that keeps
the top `ceil(alpha d)` pixels/regions according to the attribution map `a`.
In the implementation, "top" means largest attribution value.

The normalization denominator used by several metrics is:

```math
Z = \max(|f_S - f_0|, \epsilon),
```

where `epsilon` prevents division by zero.

## AUC-: Deletion AUC

Deletion removes the most attributed pixels first and measures how quickly the
model confidence drops.

For fraction `alpha`, the deletion mask is:

```math
m_{\mathrm{del}}(\alpha) = \mathbf{1} - T_\alpha.
```

The deletion score curve is:

```math
y_{\mathrm{del}}(\alpha) = v(m_{\mathrm{del}}(\alpha)).
```

The reported `auc_del` is the area under the clipped and normalized deletion
curve used by `saliency_to_auc`.

Interpretation:

- Lower `auc_del` is better.
- A good explanation removes truly important evidence early, so confidence
  should fall quickly.

Reference: Petsiuk et al. introduced deletion as an automatic causal metric for
saliency maps in RISE.

## AUC+: Insertion AUC

Insertion starts from the background image and inserts the most attributed
pixels first.

For fraction `alpha`, the insertion mask is:

```math
m_{\mathrm{ins}}(\alpha) = T_\alpha.
```

The insertion score curve is:

```math
y_{\mathrm{ins}}(\alpha) = v(m_{\mathrm{ins}}(\alpha)).
```

The reported `auc_ins` is the area under the clipped and normalized insertion
curve used by `saliency_to_auc`.

Interpretation:

- Higher `auc_ins` is better.
- A good explanation should recover the model confidence using only a small
  fraction of the most attributed pixels.

Reference: Petsiuk et al. introduced insertion together with deletion in RISE.

## Drop@Fraction

Columns:

```text
drop_at_10
drop_at_20
drop_at_30
```

Drop@fraction is the normalized confidence drop after removing the top
attributed fraction.

For `alpha in {0.10, 0.20, 0.30}`:

```math
\mathrm{Drop@}\alpha =
\frac{f_S - v(\mathbf{1} - T_\alpha)}{Z}.
```

Interpretation:

- Higher is better.
- It is a checkpoint version of the deletion idea.
- Unlike `auc_del`, this is not an area; it is a single value at a fixed
  fraction.

## Insert@Fraction

Columns:

```text
insert_at_10
insert_at_20
insert_at_30
```

Insert@fraction measures how much confidence is recovered when only the top
attributed fraction is kept.

For `alpha in {0.10, 0.20, 0.30}`:

```math
\mathrm{Insert@}\alpha =
\frac{v(T_\alpha) - f_0}{Z}.
```

Interpretation:

- Higher is better.
- It is a checkpoint version of the insertion idea.

## Sufficiency

Columns:

```text
sufficiency_at_10
sufficiency_at_20
sufficiency_gap_at_10
sufficiency_gap_at_20
```

Sufficiency asks whether the selected top-attribution pixels are enough for the
model to preserve its prediction score.

The raw sufficiency score saved in this project is:

```math
\mathrm{Sufficiency@}\alpha = v(T_\alpha).
```

The sufficiency gap is:

```math
\mathrm{SufficiencyGap@}\alpha = f_S - v(T_\alpha).
```

Interpretation:

- Higher `sufficiency_at_*` is better.
- Lower `sufficiency_gap_at_*` is better.
- A small sufficiency gap means the selected pixels contain enough evidence to
  nearly reproduce the full-image score.

Reference: The sufficiency gap follows the rationale-evaluation framing used in
ERASER, where sufficiency compares the full-input score with the score using
only the selected rationale.

## Comprehensiveness

Columns:

```text
comprehensiveness_at_10
comprehensiveness_at_20
```

Comprehensiveness asks whether removing the selected top-attribution pixels
removes the evidence used by the model.

For `alpha in {0.10, 0.20}`:

```math
\mathrm{Comprehensiveness@}\alpha =
f_S - v(\mathbf{1} - T_\alpha).
```

Interpretation:

- Higher is better.
- If the top-attribution pixels are truly important, removing them should reduce
  the model score.

Reference: This follows the comprehensiveness metric used in ERASER.

## Sensitivity-n Correlation

Column:

```text
sensitivity_n_corr
```

Sensitivity-n measures whether attribution mass over a removed subset predicts
the actual model score change caused by removing that subset.

For random removed subsets `R_j`, this project computes:

```math
A_j = \sum_{i \in R_j} \max(a_i, 0),
```

and:

```math
D_j = f_S - v(\mathbf{1} - R_j).
```

The metric is the Pearson correlation:

```math
\mathrm{SensitivityN} =
\mathrm{corr}\left(\{A_j\}_j, \{D_j\}_j\right).
```

Implementation details:

- Random subsets remove between 5% and 50% of pixels/regions.
- The current default uses 20 random subsets.
- Only positive attribution mass is used in `A_j`.

Interpretation:

- Higher is better.
- A high value means attribution mass predicts real model confidence drop.

Reference: Sensitivity-n was introduced by Ancona et al. for evaluating
feature-attribution methods. Later image-attribution evaluation surveys describe
it as correlating attribution sums with output differences after removing
features.

## Stability: Top-10 Jaccard

Column:

```text
stability_top10_jaccard
```

This is a lightweight attribution-rank stability proxy. It does not recompute
the explanation on a perturbed image. Instead, it adds small noise to the
attribution map and checks whether the top 10% attribution set remains stable.

Let:

```math
T_{0.10}(a)
```

be the top-10% mask for attribution map `a`. For perturbed attribution maps:

```math
a^{(r)} = a + \eta^{(r)},
```

the score is:

```math
\mathrm{StabilityTop10} =
\frac{1}{R}
\sum_{r=1}^R
\frac{|T_{0.10}(a) \cap T_{0.10}(a^{(r)})|}
{|T_{0.10}(a) \cup T_{0.10}(a^{(r)})|}.
```

Interpretation:

- Higher is better.
- A value near `1` means the most important pixels/regions are stable under
  small attribution perturbations.

Relation to literature:

- This is inspired by the broader use of perturbation and randomization checks
  for saliency maps, especially the warning that visual inspection alone is not
  enough.
- It is not the full model-randomization sanity check from Adebayo et al.; it is
  a cheap local rank-stability diagnostic.

## Per-Image Curves

The scripts also save:

```text
faithfulness_curves_<image_no>.png
```

This figure contains:

- Drop@fraction curve:

```math
\alpha \mapsto \frac{f_S - v(\mathbf{1} - T_\alpha)}{Z}
```

- Insert@fraction curve:

```math
\alpha \mapsto \frac{v(T_\alpha) - f_0}{Z}
```

- Sufficiency curve:

```math
\alpha \mapsto v(T_\alpha)
```

- Comprehensiveness curve:

```math
\alpha \mapsto f_S - v(\mathbf{1} - T_\alpha)
```

- Sensitivity-n scatter:

```math
x = \sum_{i \in R_j} \max(a_i, 0),
\qquad
y = f_S - v(\mathbf{1} - R_j).
```

All curve legends are sorted best-first. For the four curves, sorting uses the
final value at the largest plotted fraction. For the Sensitivity-n scatter,
sorting uses Pearson correlation.

## Metric Summary

| Metric | CSV columns | Better |
| :----- | :---------- | :----- |
| Insertion AUC | `auc_ins` | Higher |
| Deletion AUC | `auc_del` | Lower |
| Drop@fraction | `drop_at_10/20/30` | Higher |
| Insert@fraction | `insert_at_10/20/30` | Higher |
| Sufficiency | `sufficiency_at_10/20` | Higher |
| Sufficiency gap | `sufficiency_gap_at_10/20` | Lower |
| Comprehensiveness | `comprehensiveness_at_10/20` | Higher |
| Sensitivity-n | `sensitivity_n_corr` | Higher |
| Stability | `stability_top10_jaccard` | Higher |

## References

1. Vitali Petsiuk, Abir Das, Kate Saenko. "RISE: Randomized Input Sampling for
   Explanation of Black-box Models." BMVC 2018.
   https://github.com/eclique/RISE

2. Marco Ancona, Enea Ceolini, Cengiz Oztireli, Markus Gross. "Towards better
   understanding of gradient-based attribution methods for Deep Neural
   Networks." ICLR 2018.
   https://openreview.net/forum?id=Sy21R9JAW

3. Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong,
   Richard Socher, Byron C. Wallace. "ERASER: A Benchmark to Evaluate
   Rationalized NLP Models." ACL 2020.
   https://aclanthology.org/2020.acl-main.408/

4. Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt,
   Been Kim. "Sanity Checks for Saliency Maps." NeurIPS 2018.
   https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html

5. Richard Tomsett, Dan Harborne, Supriyo Chakraborty, Prudhvi Gurram, Alun
   Preece. "Sanity Checks for Saliency Metrics." AAAI 2020.
   https://doi.org/10.1609/aaai.v34i04.6064

