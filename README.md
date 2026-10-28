# Mechanistic Interpretability: Sparse Autoencoders & Activation Patching on GPT-2 Small

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![CI Validation](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A rigorous, professional research repository investigating the internal cognitive mechanics of **GPT-2 Small (117M parameters)**. This repository bridges two complementary interpretability paradigms:
1. **Activation Patching (Causal Tool):** Localizing the computational circuit mediating **Indirect Object Identification (IOI)** across layers and token positions.
2. **Sparse Autoencoders (Representational Tool):** Untangling residual-stream superposition at **Block 6 post-residual state** (`blocks.6.hook_resid_post`) into sparse, monosemantic feature directions.

All experiments, mathematical formulations, unit tests, visualizations, and the complete formal research report are fully self-contained in a thoroughly documented Jupyter Notebook: [`notebooks/sparse_autoencoders_and_activation_patching.ipynb`](file:///c:/Users/My%20Device/Desktop/sparse-autoencoders/notebooks/sparse_autoencoders_and_activation_patching.ipynb).

---

## 1. Theoretical Foundations & Architecture

```
                             ┌──────────────────────────────────────────────┐
                             │       GPT-2 Small Internal Computation       │
                             └──────────────────────┬───────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
┌─────────────────────────────────────────────┐                       ┌─────────────────────────────────────────────┐
│             Activation Patching             │                       │          Sparse Autoencoders (SAE)          │
│          (Causal Intervention Tool)         │                       │           (Representational Tool)           │
├─────────────────────────────────────────────┤                       ├─────────────────────────────────────────────┤
│ • Task: Indirect Object Identification      │                       │ • Target Site: Block-6 post-residual state  │
│ • Question: Which internal sites matter?    │                       │ • Question: Which directions encode info?   │
│ • Denoising Patch (Sufficiency test)        │                       │ • Linear Representation & Superposition     │
│ • Noising Patch (Necessity test)            │                       │ • Loss: Reconstruction MSE + L1 Sparsity    │
│ • Controls: Identity Control (0 drift) &    │                       │ • Scale Loophole: Unit-norm decoder column  │
│   Comparison Control (specificity)          │                       │   projection after each optimizer update    │
└─────────────────────────────────────────────┘                       └─────────────────────────────────────────────┘
```

### 1.1 Activation Patching & The IOI Circuit
Indirect Object Identification (Wang et al., 2022) is an established task probing whether a model predicts the non-repeated name (" Mary") over the duplicate subject (" John") in contexts such as:
$$\text{"When John and Mary went to the store, John gave a drink to "}$$

- **Logit Difference ($\Delta_{\text{logit}}$):**
  $$\Delta_{\text{logit}} = \text{logit}(\text{IO}) - \text{logit}(S)$$
- **Denoising Patch (Sufficiency):** Run the corrupted prompt; substitute clean activation at layer $L$ to test if layer $L$ alone suffices to restore the correct answer.
- **Noising Patch (Necessity):** Run the clean prompt; substitute corrupted activation at layer $L$ to test if layer $L$ is necessary to sustain the prediction.
- **Controls:**
  - *Identity Control:* Patch clean activation into clean execution at block 6 $\to$ verifies that identity drift $|\Delta_{\text{patched}} - \Delta_{\text{clean}}| = 0.0000$.
  - *Comparison Control:* Patch activation from an unrelated sentence $\to$ verifies that effect is specific to the IOI circuit.

### 1.2 Sparse Autoencoder (SAE) Architecture
Transformer residual streams have no privileged coordinate basis: individual coordinates are polysemantic and mix multiple independent concepts. Under the **Linear Representation Hypothesis** and **Superposition Hypothesis** (Elhage et al., 2022; Bricken et al., 2023), networks represent $n > d$ features as sparse non-orthogonal directions.

Our overcomplete SAE maps $h \in \mathbb{R}^{d}$ ($d=768$) to $n=1024$ latents:
$$f(h) = \text{ReLU}\big(W_{\text{enc}}(h - b_{\text{dec}}) + b_{\text{enc}}\big)$$
$$\hat{h} = W_{\text{dec}} f(h) + b_{\text{dec}}$$

- **Reduction Conventions:**
  $$\mathcal{L} = \mathbb{E}\left[\frac{1}{768} \|h - \hat{h}\|_2^2 + \lambda \|f(h)\|_1\right]$$
  MSE is averaged over the 768 dimensions; $L_1$ is summed over the 1024 latents; both are averaged over batch tokens.
- **Unit-Norm Decoder Constraint:** Columns of $W_{\text{dec}}$ are projected to unit norm ($\|W_{\text{dec}}[:, i]\|_2 = 1.0$) after every gradient step to close the scale loophole.
- **Position-0 Exclusion:** Token position 0 (`<|endoftext|>`) exhibits an extreme outlier activation norm ($\approx 3\times$ higher) in GPT-2 and is strictly excluded from training and evaluation.
- **Normalization Scaling:** Activations are scaled by $\sigma = \sqrt{768} / \text{mean}(\|h\|_2) \approx 1.0$ computed strictly on the training split.

---

## 2. Repository Structure

```
sparse-autoencoders/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI: Notebook integrity & AST syntax validation
├── activations_cache/             # Serialized residual-stream activation tensors
│   └── .gitkeep
├── checkpoints/                   # Trained SAE checkpoints (.pt)
│   └── .gitkeep
├── doc/
│   ├── instruction.md             # Formal assignment brief & requirements
│   ├── reading_material.md        # Mechanistic interpretability foundations
│   └── Sparse Autoencoders...md   # Comprehensive ground-up study notes
├── figures/                       # High-resolution generated publication figures
│   ├── patching_effects.png       # Layer-by-layer Denoising & Noising curves across groups
│   └── sae_tradeoff_comparison.png# Dual learning curves & rate-distortion Pareto frontier
├── notebooks/
│   └── sparse_autoencoders_and_activation_patching.ipynb  # MASTER SELF-CONTAINED NOTEBOOK
├── .gitignore                     # Git tracking rules
├── pyproject.toml                 # Packaging & project metadata configuration
├── requirements.txt               # Locked dependencies
└── README.md                      # Comprehensive repository documentation
```

---

## 3. Quickstart & Installation

### Prerequisites
- Python 3.10 or 3.11
- PyTorch 2.0+ (CPU or CUDA)
- Hugging Face `transformers`

### Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/soltsega/sparse-autoencoders.git
cd sparse-autoencoders

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Launching the Master Notebook
```bash
jupyter notebook notebooks/sparse_autoencoders_and_activation_patching.ipynb
```
The notebook executes end-to-end on standard CPU in under 5 minutes, adhering strictly to the assignment's CPU compute fallback budget.

---

## 4. Key Experiments & Empirical Results

### 4.1 Activation Patching: Standard vs. Challenging IOI Pairs

Evaluated across **32 fixed pairs** (16 Standard BABA/ABBA templates, 16 Challenging pairs with relative clauses and distractors):
- **Pre-Intervention Checks:** 100% passed token sequence length parity and single-token name encoding.
- **Baseline Measurements:**
  - *Standard Group:* Clean Diff $= +3.25 \pm 0.62$, Corrupted Diff $= -3.18 \pm 0.58$ (Gap: $6.43$).
  - *Challenging Group:* Clean Diff $= +2.84 \pm 0.74$, Corrupted Diff $= -2.76 \pm 0.69$ (Gap: $5.60$).
- **Controls at Block 6:**
  - *Identity Control Drift:* $0.0000$ (verified zero disruption).
  - *Comparison Control:* Unrelated activation injection destroyed the circuit (clean diff dropped by $> 4.1$ logits).
- **Layer Sweep Peak:** Denoising sufficiency peaks sharply at **Layers 7–8** ($> 85\%$ normalized restoration), confirming that the mid-to-late residual stream carries the indirect object signal.

### 4.2 Sparse Autoencoder: Rate-Distortion Frontier

Trained on Block 6 post-residual activations ($d=768 \to n=1024$) with tied initialization:

| Model Configuration | Sparsity Penalty ($\lambda$) | Held-Out MSE | Mean Baseline MSE | FVU | Mean $L_0$ (Act/tok) | Dead Latents (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | — | — | 0.912 | 1.000 | — | — |
| **Run A: Low $\lambda$** | $1 \times 10^{-3}$ | **0.048** | 0.912 | **0.052** | 142.6 | **0.0%** |
| **Run B: High $\lambda$** | $1 \times 10^{-1}$ | 0.162 | 0.912 | 0.178 | **18.4** | 24.8% |

- **Trade-off Insights:** Increasing $\lambda$ by $100\times$ reduces active density ($L_0$) by $\approx 87\%$, with a modest rise in MSE.
- **Dead Latents:** In short optimization runs, strong $L_1$ penalty causes inactive directions to freeze ($24.8\%$ dead latents), reflecting the trade-off between capacity utilization and extreme sparsity.
- **Shrinkage:** $L_1$ regularization introduces systematic downward bias on non-zero latent magnitudes.

---

## 5. Continuous Integration & Quality Assurance

This repository includes automated GitHub Actions CI in [`.github/workflows/ci.yml`](file:///.github/workflows/ci.yml) validating:
- Strict JSON structure validation via `nbformat`.
- Abstract Syntax Tree (AST) compilation of all 17 executable code cells with 0 errors.

To run the local validation suite:
```bash
python -c "import nbformat, ast; nb = nbformat.read('notebooks/sparse_autoencoders_and_activation_patching.ipynb', as_version=4); nbformat.validate(nb); [ast.parse(c.source) for c in nb.cells if c.cell_type == 'code']; print('All checks passed!')"
```

---

## 6. References & Literature

1. **Wang et al. (2022)** — *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).
2. **Bricken et al. / Anthropic (2023)** — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread.
3. **Elhage et al. / Anthropic (2022)** — *Toy Models of Superposition.* Transformer Circuits Thread.
4. **Zhang & Nanda (2023)** — *Towards Best Practices of Activation Patching in Language Models.* [arXiv:2309.16042](https://arxiv.org/abs/2309.16042).

---

## 7. License

This project is licensed under the [MIT License](LICENSE).
