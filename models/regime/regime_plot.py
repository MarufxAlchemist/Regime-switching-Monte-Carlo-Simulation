"""
Regime Plot Over Real Price Data
=================================
Uses S&P 500 daily returns from yfinance to fit a 3-state HMM.
Plots regime bands (Bull / Bear / Crisis) overlaid on the price series.

Auto-tuning: if COVID crash (Feb–May 2020) is not classified as Crisis,
the Crisis emission mean is manually biased lower and the model is refit.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
from hmmlearn import hmm

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download real data
# ─────────────────────────────────────────────────────────────────────────────
TICKER     = "^GSPC"    # S&P 500
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

print(f"Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)
price  = raw["Close"].squeeze()
ret    = np.log(price / price.shift(1)).dropna()   # log returns
obs    = ret.values.reshape(-1, 1)                 # (T, 1)  — univariate
dates  = ret.index
T      = len(obs)
print(f"  {T} trading days loaded.")

# COVID crash window (inclusive)
COVID_START = "2020-02-20"
COVID_END   = "2020-05-01"
covid_mask  = (dates >= COVID_START) & (dates <= COVID_END)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fit 3-state HMM helper
# ─────────────────────────────────────────────────────────────────────────────
NUM_STATES   = 3
REGIME_NAMES = ["Bull", "Bear", "Crisis"]
REGIME_COLS  = ["#2ecc71", "#e67e22", "#e74c3c"]   # green / amber / red

def fit_hmm(obs: np.ndarray,
            crisis_mu_bias: float = 0.0,
            n_iter: int = 300) -> hmm.GaussianHMM:
    """
    Fit Gaussian HMM.
    crisis_mu_bias: extra downward shift added to the crisis seed mean
                    (used for tuning if COVID is misclassified).
    """
    scalar = obs[:, 0]
    q20    = np.percentile(scalar, 20)
    q80    = np.percentile(scalar, 80)

    mu_bull   = obs[scalar >= q80].mean(axis=0)
    mu_crisis = obs[scalar <= q20].mean(axis=0) - crisis_mu_bias
    mu_bear   = obs[(scalar > q20) & (scalar < q80)].mean(axis=0)

    model = hmm.GaussianHMM(
        n_components=NUM_STATES,
        covariance_type="full",
        n_iter=n_iter,
        tol=1e-6,
        random_state=42,
        init_params="c",
        params="stmc",
    )
    model.means_     = np.array([mu_bull, mu_bear, mu_crisis])
    model.transmat_  = np.array([
        [0.92, 0.06, 0.02],
        [0.10, 0.85, 0.05],
        [0.05, 0.20, 0.75],
    ])
    model.startprob_ = np.array([0.60, 0.30, 0.10])
    model.fit(obs)
    return model


def decode_and_align(model: hmm.GaussianHMM, obs: np.ndarray):
    """Decode Viterbi path and sort states: highest mean→Bull, lowest→Crisis."""
    _, raw_states = model.decode(obs, algorithm="viterbi")
    mu            = model.means_[:, 0]
    A             = model.transmat_
    persistence   = np.diag(A)
    score         = (mu / (mu.std() + 1e-12) +
                     0.1 * persistence / (persistence.std() + 1e-12))
    perm          = np.argsort(-score)              # perm[0]=Bull, perm[2]=Crisis

    # remap raw state indices → semantic labels {0=Bull,1=Bear,2=Crisis}
    inv_perm      = np.empty(NUM_STATES, dtype=int)
    for k in range(NUM_STATES):
        inv_perm[perm[k]] = k
    states = inv_perm[raw_states]
    return states, perm, model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Initial fit
# ─────────────────────────────────────────────────────────────────────────────
print("Fitting 3-state Gaussian HMM ...")
model = fit_hmm(obs, crisis_mu_bias=0.0)
states, perm, model = decode_and_align(model, obs)

covid_crisis_frac = states[covid_mask].tolist().count(2) / covid_mask.sum()
print(f"  COVID window classified as Crisis: {covid_crisis_frac:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Auto-tune if COVID is not predominantly Crisis
# ─────────────────────────────────────────────────────────────────────────────
TUNE_THRESHOLD = 0.60    # at least 60% of COVID days must be Crisis
bias_step      = 0.002
max_rounds     = 8

if covid_crisis_frac < TUNE_THRESHOLD:
    print(f"  COVID crisis coverage {covid_crisis_frac:.1%} < {TUNE_THRESHOLD:.0%} "
          f"— tuning crisis mean bias ...")
    for rnd in range(1, max_rounds + 1):
        bias = bias_step * rnd
        m    = fit_hmm(obs, crisis_mu_bias=bias)
        s, p, m = decode_and_align(m, obs)
        frac = s[covid_mask].tolist().count(2) / covid_mask.sum()
        print(f"    Round {rnd}: bias={bias:.4f}  COVID-crisis={frac:.1%}")
        if frac >= TUNE_THRESHOLD:
            states, perm, model = s, p, m
            covid_crisis_frac   = frac
            print(f"  ✓ Threshold met at round {rnd} (bias={bias:.4f})")
            break
    else:
        # Use last result even if threshold not fully met
        states, perm, model = s, p, m
        covid_crisis_frac   = frac
        print(f"  ⚠ Max rounds reached; best COVID-crisis={covid_crisis_frac:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Extract fitted parameters
# ─────────────────────────────────────────────────────────────────────────────
mu_fitted    = model.means_[perm, 0] * 252          # annualised mean per regime
vol_fitted   = np.sqrt(model.covars_[perm, 0, 0]) * np.sqrt(252)
A_aligned    = model.transmat_[np.ix_(perm, perm)]
hold_times   = 1.0 / (1.0 - np.diag(model.transmat_) + 1e-12)
hold_aligned = hold_times[perm]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plot ...")
fig, axes = plt.subplots(
    3, 1, figsize=(16, 10),
    gridspec_kw={"height_ratios": [3, 1, 1]},
    sharex=True
)
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

price_vals = price.loc[dates].values
ax_price, ax_regime, ax_ret = axes

# ── Panel 1: Price with regime background ────────────────────────────────────
ax_price.plot(dates, price_vals, color="#ffffff", linewidth=1.0, zorder=5)
ax_price.set_ylabel("S&P 500 Price", color="white", fontsize=11)
ax_price.set_title("S&P 500 — HMM Regime Classification (3 States)",
                   color="white", fontsize=13, pad=10)

# shade regime bands
for k, (col, name) in enumerate(zip(REGIME_COLS, REGIME_NAMES)):
    mask_k = (states == k)
    # find contiguous runs
    in_run = False
    start  = None
    for i, flag in enumerate(mask_k):
        if flag and not in_run:
            start  = i
            in_run = True
        elif not flag and in_run:
            ax_price.axvspan(dates[start], dates[i - 1],
                             alpha=0.25, color=col, zorder=0)
            in_run = False
    if in_run:
        ax_price.axvspan(dates[start], dates[-1], alpha=0.25, color=col, zorder=0)

# annotate COVID
ax_price.axvspan(COVID_START, COVID_END, alpha=0.10, color="white", zorder=1)
ax_price.text(
    np.datetime64("2020-03-20"), price_vals.min() * 1.02,
    "COVID\nCrash", color="white", fontsize=8, ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="#e74c3c", alpha=0.7)
)

patches = [mpatches.Patch(color=REGIME_COLS[k], label=REGIME_NAMES[k],
                           alpha=0.7) for k in range(NUM_STATES)]
ax_price.legend(handles=patches, loc="upper left",
                facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# ── Panel 2: Discrete regime index ───────────────────────────────────────────
regime_colors = np.array(REGIME_COLS)[states]
for k in range(NUM_STATES):
    m = states == k
    ax_regime.fill_between(dates, 0, 1,
                           where=m, color=REGIME_COLS[k], alpha=0.8)
ax_regime.set_yticks([0.15, 0.5, 0.85])
ax_regime.set_yticklabels(["Bull", "Bear", "Crisis"], color="white", fontsize=8)
ax_regime.set_ylabel("Regime", color="white", fontsize=9)
ax_regime.set_ylim(0, 1)

# ── Panel 3: Daily log returns ────────────────────────────────────────────────
ret_vals = obs[:, 0]
pos = ret_vals >= 0
ax_ret.bar(dates[pos],  ret_vals[pos],  color="#2ecc71", width=1, alpha=0.7)
ax_ret.bar(dates[~pos], ret_vals[~pos], color="#e74c3c", width=1, alpha=0.7)
ax_ret.axhline(0, color="white", linewidth=0.5)
ax_ret.set_ylabel("Log Return", color="white", fontsize=9)
ax_ret.set_xlabel("Date", color="white", fontsize=10)

plt.tight_layout()
out_path = "models/regime/regime_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"  Saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Print extracted parameters
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 60
print(f"\n{SEP}")
print("  REGIME-DEPENDENT PARAMETERS  (fitted on real S&P 500)")
print(SEP)
occ = [(states == k).sum() / T for k in range(NUM_STATES)]
print(f"\n  {'Regime':<8} {'Ann.μ':>8} {'Ann.σ':>8}  {'Hold(d)':>8}  {'Occ.':>7}")
print(f"  {'-'*50}")
for k in range(NUM_STATES):
    print(f"  {REGIME_NAMES[k]:<8} {mu_fitted[k]:>7.2%} {vol_fitted[k]:>8.2%}  "
          f"{hold_aligned[k]:>8.1f}  {occ[k]:>6.1%}")

print(f"\n  Transition Matrix (aligned):")
hdr = "         " + "".join(f"  {n:<8}" for n in REGIME_NAMES)
print(hdr)
for i in range(NUM_STATES):
    row = "".join(f"  {A_aligned[i,j]:.4f}  " for j in range(NUM_STATES))
    print(f"  {REGIME_NAMES[i]:<8}{row}")

print(f"\n  COVID crisis coverage: {covid_crisis_frac:.1%}")
print(SEP)
