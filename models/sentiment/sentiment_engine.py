"""
FinBERT Sentiment Engine
=========================
Loads ProsusAI/finbert from Hugging Face.

For each headline / text:
  1. Tokenise and run through FinBERT
  2. Softmax → (P_neg, P_neu, P_pos)
  3. Sentiment score  S = P_pos − P_neg  ∈ (−1, +1)

GBM parameter adjustment with sentiment:
  μ_new   = μ  +  α × S          (drift shift)
  σ_new   = σ  × (1 + β × |S|)  (vol amplification)

Output: plain text report + optional integration hook for MC engine.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"
ALPHA       = 0.05    # sentiment → drift adjustment  (annualised units)
BETA        = 0.10    # sentiment → vol amplification factor

# Sentiment score bounds for clipping (prevents nonsensical adjustments).
SCORE_MIN, SCORE_MAX = -1.0, 1.0

# FinBERT label order  (as per ProsusAI model card).
# Index 0 = positive, 1 = negative, 2 = neutral
label_map  = {0: "positive", 1: "negative", 2: "neutral"}
IDX_POS    = 0
IDX_NEG    = 1

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load FinBERT
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading FinBERT ({MODEL_NAME}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()                         # inference mode — no gradient tracking
print("  Model loaded ✓\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sentiment scoring function
# ─────────────────────────────────────────────────────────────────────────────
def get_sentiment_score(text: str) -> dict:
    """
    Run FinBERT on a single string.

    Returns:
        {
          'text'     : original text,
          'P_pos'    : P(positive),
          'P_neg'    : P(negative),
          'P_neu'    : P(neutral),
          'score'    : P_pos - P_neg  ∈ (-1, +1),
          'label'    : dominant class label,
        }
    """
    inputs     = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
    input_ids  = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)

    # Softmax over logits → class probabilities
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    P_pos  = predictions[0][IDX_POS].item()
    P_neg  = predictions[0][IDX_NEG].item()
    P_neu  = predictions[0][2].item()
    score  = float(np.clip(P_pos - P_neg, SCORE_MIN, SCORE_MAX))
    label  = label_map[predictions[0].argmax().item()]

    return {
        "text"  : text,
        "P_pos" : P_pos,
        "P_neg" : P_neg,
        "P_neu" : P_neu,
        "score" : score,
        "label" : label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. GBM parameter adjustment
# ─────────────────────────────────────────────────────────────────────────────
def adjust_gbm_params(mu: float, sigma: float,
                      sentiment_score: float,
                      alpha: float = ALPHA,
                      beta:  float = BETA) -> tuple[float, float]:
    """
    Adjust annualised GBM drift (μ) and volatility (σ) using sentiment.

    Formulas:
        μ_new  = μ  +  α × S
        σ_new  = σ  × (1 + β × |S|)

    Args:
        mu              : baseline annualised drift
        sigma           : baseline annualised volatility
        sentiment_score : S ∈ (-1, +1)  from get_sentiment_score()
        alpha           : drift sensitivity  (default 0.05)
        beta            : vol sensitivity    (default 0.10)

    Returns:
        (mu_new, sigma_new)
    """
    S        = float(np.clip(sentiment_score, SCORE_MIN, SCORE_MAX))
    mu_new   = mu    + alpha * S
    sigma_new = sigma * (1.0 + beta * abs(S))
    return mu_new, sigma_new


# ─────────────────────────────────────────────────────────────────────────────
# 4. Demo — run on a set of representative financial headlines
# ─────────────────────────────────────────────────────────────────────────────
DEMO_HEADLINES = [
    # Clearly positive
    "JPMorgan beats earnings expectations with record quarterly profit.",
    "Fed signals rate cuts ahead as inflation cools to 2-year low.",
    "S&P 500 surges 3% on strong jobs data and cooling inflation.",
    # Neutral / mixed
    "Federal Reserve holds rates steady, awaits further data.",
    "Goldman Sachs quarterly earnings in line with analyst estimates.",
    # Clearly negative
    "Bank of America warns of rising credit losses amid economic slowdown.",
    "Markets tumble as GDP contracts sharply in Q3, recession fears mount.",
    "Silicon Valley Bank collapses; regulators step in to guarantee deposits.",
]

# Baseline GBM params (illustrative: SPY-like asset)
MU_BASE    = 0.10    # 10% ann. drift
SIGMA_BASE = 0.18    # 18% ann. volatility

SEP = "=" * 72
print(SEP)
print("  FINBERT SENTIMENT ENGINE")
print(SEP)
print(f"\n  Baseline params:  μ={MU_BASE:.2%}  σ={SIGMA_BASE:.2%}")
print(f"  Adjustment:  μ_new = μ + α×S   "
      f"[α={ALPHA}]    σ_new = σ×(1+β×|S|)  [β={BETA}]\n")
print(f"  {'Headline':<56} {'Score':>7}  {'Label':<10}  {'μ_new':>8}  {'σ_new':>8}")
print(f"  {'-'*100}")

score_list = []
for headline in DEMO_HEADLINES:
    result = get_sentiment_score(headline)
    mu_new, sig_new = adjust_gbm_params(MU_BASE, SIGMA_BASE, result["score"])
    score_list.append(result["score"])
    short = (headline[:53] + "..") if len(headline) > 55 else headline
    print(f"  {short:<56} {result['score']:>+7.4f}  {result['label']:<10}  "
          f"{mu_new:>+8.2%}  {sig_new:>8.2%}")

# Aggregate sentiment signal
agg_score = float(np.mean(score_list))
mu_agg, sig_agg = adjust_gbm_params(MU_BASE, SIGMA_BASE, agg_score)

print(f"\n  {'─'*100}")
print(f"  {'Aggregate (mean of all headlines)':<56} {agg_score:>+7.4f}  {'—':<10}  "
      f"{mu_agg:>+8.2%}  {sig_agg:>8.2%}")

print(f"\n{SEP}")
print(f"  INTERPRETATION")
print(SEP)
print(f"\n  Aggregate sentiment score : {agg_score:>+.4f}")
if agg_score > 0.1:
    tone = "BULLISH — positive news flow"
elif agg_score < -0.1:
    tone = "BEARISH — negative news flow"
else:
    tone = "NEUTRAL / MIXED"
print(f"  Market tone               : {tone}")
print(f"\n  Adjusted GBM params (aggregate):")
print(f"    μ  :  {MU_BASE:.2%}  →  {mu_agg:.2%}  "
      f"(Δ = {mu_agg - MU_BASE:>+.4f}  annualised drift shift)")
print(f"    σ  :  {SIGMA_BASE:.2%}  →  {sig_agg:.2%}  "
      f"(×{sig_agg/SIGMA_BASE:.3f}  vol amplification)")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration hook — ready to use in MC engine
# ─────────────────────────────────────────────────────────────────────────────
def sentiment_adjusted_params(headlines: list[str],
                               mu_baseline: np.ndarray,
                               sigma_baseline: np.ndarray,
                               alpha: float = ALPHA,
                               beta:  float = BETA
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised wrapper for MC engine integration.

    Takes a list of headlines and baseline per-asset μ/σ arrays.
    Returns adjusted μ and σ arrays using the aggregate sentiment score.

    Usage (in contagion.py or base_mc.py):
        mu_adj, sig_adj = sentiment_adjusted_params(
            headlines, MU_ANNUAL / 252, SIG_ANNUAL / np.sqrt(252)
        )
    """
    scores = [get_sentiment_score(h)["score"] for h in headlines]
    S_agg  = float(np.clip(np.mean(scores), SCORE_MIN, SCORE_MAX))
    mu_adj    = mu_baseline    + alpha * S_agg
    sigma_adj = sigma_baseline * (1.0 + beta * abs(S_agg))
    return mu_adj, sigma_adj
