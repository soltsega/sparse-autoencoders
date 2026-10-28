# Research Assignment — Activation Patching & Sparse Autoencoders

## Train and evaluate your own SAE

### The representation question

A GPT-2 residual-stream vector is 768 numbers, but that does not imply 768 individually meaningful features. Train an SAE to describe activations at **one fixed site**, preferably **the block-6 post-residual state**, using ordinary varied text that is **separate from the IOI patching prompts**. GPT-2 stays frozen throughout.

For each activation `h`, train an encoder and decoder:

$$
f(h)=\mathrm{ReLU}\!\big(W_{\mathrm{enc}}(h-b_{\mathrm{dec}})+b_{\mathrm{enc}}\big),
$$

$$
\hat h=W_{\mathrm{dec}}f(h)+b_{\mathrm{dec}}.
$$

The encoder maps a 768-dimensional state to a latent vector; the decoder reconstructs 768 dimensions. Use **1,024 latent units** by default. This dictionary is overcomplete because it has more candidate feature directions than input dimensions, even though only a subset should be active on a given token.

Minimize reconstruction error plus a sparsity penalty:

$$
\mathcal L=\mathbb E\left[\frac1{768}\lVert h-\hat h\rVert_2^2+\lambda\,\lVert f(h)\rVert_1\right].
$$

State your reduction conventions. Keep decoder column norms constrained (for example, renormalize to unit norm after each optimizer step) to avoid a scale loophole in the sparsity cost. Use consistent activation scaling between runs.

### Required experiments

1. **Collect activations.** Use varied short passages rather than the 32 supplied IOI patching prompts alone. Target about **10,000–20,000 token-position activation vectors**, removing padding states and splitting source passages into training and held-out sets **before extraction**. Save the activation-site name, data provenance, seeds, and tensor shapes. Cached activations are recommended so GPT-2 need not run during every SAE update.
2. **Train two small SAEs.** Same architecture, data split, number of steps, and initialization seed; two nonzero sparsity penalties with a clear low/high contrast. Around **300–600 optimizer steps per run** with batches of 128 is a starting budget, not a requirement to reach a particular quality threshold. Record learning curves and checkpoint weights. Do not change the held-out set to find a preferred result.
3. **Evaluate on unseen activations.** Report held-out reconstruction MSE (under a stated normalization), mean active latents per token (`L_0`, using an explicit threshold), and percentage of latents that never activate on the evaluation sample. Compare a simple mean-activation reconstruction baseline if possible.
4. **Inspect a few learned features.** For **two or three latents**, show their strongest activating token positions with enough surrounding text to interpret them. Write a tentative description, then test it against at least **two new counterexamples or contrastive contexts**. If the units do not have clear meanings, report that limitation rather than inventing names.
5. **Explain the trade-off.** Does the stronger penalty reduce activation density? What happens to reconstruction? Can a model reconstruct well without discovering semantically clean features? Include one figure comparing the two runs.

The goal is to **train and critically evaluate a small SAE**, not to recover a research-quality GPT-2 dictionary. On a limited corpus, many latents may be difficult to interpret and some may be inactive. These are scientifically meaningful observations.

### Compute fallback

Free Colab GPU availability varies. Start on CPU if needed, and do not wait for a GPU to begin. A limited-run fallback is **5,000–10,000 cached activations, 1,024 latents, and approximately 150–300 updates per run**, with the same required comparative analyses. Lower the batch size to 64 or 32 if necessary. If the GPU is unavailable, cache a small activation tensor yourself on CPU and report exactly what was feasible.

**There is no penalty for the fallback configuration.** Grade the validity of comparisons and interpretation, not elapsed GPU time or convergence to a target value. Checkpoint and download intermediate outputs before a Colab runtime resets.

Relevant reading: Sparse Autoencoders.

## Deliverables

Submit **one notebook or two closely linked notebooks**, plus saved output files and a **700–1,000-word report**. Report from measured output, not tutorial screenshots.

The report must include:

- A one-paragraph causal hypothesis, the 32 supplied pair IDs, pre-intervention tokenization checks and baseline results, the exact patch sites, and a stated choice of full or fixed fallback evaluation sample.
- A baseline table separated into standard/challenging groups; a figure comparing patch effects across those groups; the identity and comparison controls; and a limitation of the causal conclusion.
- The SAE architecture, site, corpus and split, training settings, two-run comparison, held-out reconstruction and activity statistics, and example contexts for inspected features.
- At least one observation that challenged an initial expectation.
- A reproducibility record: model ID, library versions, seeds, device, data source or saved cache, number of examples/tokens, SAE checkpoint, and any runtime or memory limitations.

A well-reported null patch, weak feature semantics, or incomplete convergence can still receive strong marks. **Do not optimize for striking heatmaps or attractive feature labels.** Source-code assistance is allowed if disclosed briefly and the intern can explain and independently check the resulting experiment.

## Important implementation boundaries

- You will not receive a starter notebook, model checkpoint file, activation cache, or prepared SAE. Obtain pretrained GPT-2 Small from a public model repository; the model remains frozen. Use the two study chapters and official library documentation to implement the experiments yourself.
- Keep GPT-2 in evaluation mode, disable gradients when only extracting activations, and separate language-model forward passes from SAE optimization.
- Do not mix the patching corpus with the SAE held-out evaluation split. The SAE need not discover the same feature implicated by the patch; proving that link would require a more selective experiment.
- No marks are tied to GPU access, paid APIs, large training runs, reproducing a famous circuit, or finding named features.

## Research and operating references

- TransformerLens — setup from scratch
- TransformerLens — current model structure and hook names
- Public text corpus — WikiText-2
- TransformerLens — GPT-2 IOI patching demo
- SAE Lens — training and activation caching
- Google Colab — resource availability and limits
- Wang et al., Interpretability in the Wild: a Circuit for Indirect Object Identification
- Zhang & Nanda, Towards Best Practices of Activation Patching