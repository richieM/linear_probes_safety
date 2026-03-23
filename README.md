# Linear Probes for Safety Monitoring

Train simple linear probes on a language model's internal activations to detect harmful/jailbreak queries — and compare them to the model's own behavioral safety judgement.

## The idea

When a model processes a harmful query, does it "know" it's harmful internally, even if it complies? We test this by:

1. Running ~400 labeled examples (harmful / benign) through Gemma-2-2B
2. Capturing the residual stream at every transformer layer (last token position)
3. Training a logistic regression probe at each layer — just a linear classifier on the hidden states
4. Comparing AUROC by layer to a behavioral baseline (prompting the model to self-classify)

If the probe outperforms the model's own safety judgement, it means harmful intent is encoded in the representations even when the output doesn't reflect it.

## Structure

```
make_dataset.py          # builds data/dataset.json (400 examples, 8 categories)
extract_activations.py   # loads Gemma-2-2B, caches activations to results/activations.npz
train_probes.py          # trains probes per layer, generates plots and findings
run_experiment.ipynb     # Colab/Jupyter notebook that runs everything end-to-end
```

## Dataset

400 examples, balanced 50/50 harmful/benign, 80/20 train/test split.

| Harmful | Benign counterpart |
|---|---|
| Jailbreak attempts (DAN, roleplay, etc.) | AI safety & red-teaming research questions |
| Weapons / dangerous synthesis requests | Chemistry safety & education questions |
| Cybercrime requests | Cybersecurity education questions |
| Manipulation / scam scripts | Psychology & persuasion research questions |

The topics deliberately overlap so the probe has to detect *intent*, not just keywords.

## Running it

### In Google Colab (recommended — free GPU)

1. Upload the files or clone this repo
2. Runtime → Change runtime type → **T4 GPU**
3. Open `run_experiment.ipynb` and run cells top to bottom
4. You'll need a HuggingFace token (free) and to accept the [Gemma-2 license](https://huggingface.co/google/gemma-2-2b)

### Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyterlab
huggingface-cli login
jupyter lab run_experiment.ipynb
```

In the extraction cell, set `device="mps"` (Apple Silicon) or `device="cpu"`.

## What to look for

- **AUROC near 0.5 at early layers** → the model hasn't built the concept of harm yet
- **Sharp jump in middle layers** → where harmful intent becomes linearly separable
- **Probe AUROC > baseline AUROC** → the model's internal representations flag harm that its output behavior misses

## References

- [Cheap monitors can catch sleeper agents (Anthropic, 2025)](https://alignment.anthropic.com/2025/cheap-monitors/)
- [Simple probes can catch sleeper agents (Hubinger et al., 2024)](https://arxiv.org/abs/2401.05566)
