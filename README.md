# Test-Time Training Layers for ETFs

Implements Test-Time Training (TTT) layers that continue to update via self-supervised loss during inference. The model adapts to distribution shift in real‑time without explicit continual learning. The score combines TTT prediction with momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- LSTM backbone with test‑time trainable layers
- Self‑supervised adaptation on new samples
- Score = TTT prediction × (1 + last_return)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-test-time-training-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow with adaptation to current market conditions.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
