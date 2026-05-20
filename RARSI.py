from __future__ import annotations


import os
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.special import logsumexp          # NEW — for log-space HMM
from scipy.stats import norm as _scipy_norm  # NEW — for z-score thresholds
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1)  UNIVERSE PRESETS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

NIFTY50_TICKERS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "BAJAJFINSV.NS", "ONGC.NS",
    "TECHM.NS", "COALINDIA.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATAMOTORS.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "BPCL.NS", "BRITANNIA.NS",
    "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS",
    "TATASTEEL.NS", "TATACONSUM.NS", "UPL.NS", "VEDL.NS", "SHREECEM.NS", "SBILIFE.NS", "HDFCBANK.NS"]

NIFTY100_EXTRA_TICKERS: List[str] = [
    "BOSCHLTD.NS", "DABUR.NS", "DLF.NS", "GODREJCP.NS", "HAVELLS.NS",
    "JINDALSTEL.NS", "LUPIN.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "PAGEIND.NS", "PIDILITIND.NS", "PNB.NS", "SBICARD.NS", "SIEMENS.NS",
    "TORNTPHARM.NS", "TRENT.NS", "VOLTAS.NS", "ZYDUSLIFE.NS", "AUBANK.NS",
    "BANDHANBNK.NS", "BERGEPAINT.NS", "CANBK.NS", "CHOLAFIN.NS", "CONCOR.NS",
    "CUMMINSIND.NS", "FEDERALBNK.NS", "GLAND.NS", "IRCTC.NS", "JUBLFOOD.NS",
    "KPITTECH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LTIM.NS", "LTTS.NS",
    "MARICO.NS", "MFSL.NS", "OFSS.NS", "PERSISTENT.NS", "PETRONET.NS",
    "SAIL.NS", "IDFCFIRSTB.NS", "IGL.NS", "MGL.NS", "TATAPOWER.NS",
    "AARTIIND.NS", "AMBUJACEM.NS", "BALKRISIND.NS", "ESCORTS.NS",
]

SECTOR_TICKERS: Dict[str, List[str]] = {
    "IT":      ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
                "LTIM.NS","LTTS.NS","PERSISTENT.NS","KPITTECH.NS","OFSS.NS"],
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
                "INDUSINDBK.NS","PNB.NS","CANBK.NS","FEDERALBNK.NS","BANDHANBNK.NS"],
    "Pharma":  ["SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","DIVISLAB.NS","LUPIN.NS",
                "TORNTPHARM.NS","ZYDUSLIFE.NS","GLAND.NS","LAURUSLABS.NS","APOLLOHOSP.NS"],
    "Energy":  ["RELIANCE.NS","ONGC.NS","BPCL.NS","COALINDIA.NS","NTPC.NS",
                "POWERGRID.NS","IGL.NS","MGL.NS","PETRONET.NS","TATAPOWER.NS"],
    "Auto":    ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
                "EICHERMOT.NS","BOSCHLTD.NS","MOTHERSON.NS","CUMMINSIND.NS","ASHOKLEY.NS"],
    "Metals":  ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS",
                "JINDALSTEL.NS","NALCO.NS","NMDC.NS","COALINDIA.NS","HINDZINC.NS"],
    "FMCG":    ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
                "GODREJCP.NS","MARICO.NS","TATACONSUM.NS","EMAMILTD.NS","COLPAL.NS"],
    "Telecom": ["BHARTIARTL.NS","IDEA.NS","RAILTEL.NS","TATACOMM.NS","MTNL.NS"],
}

LARGE_CAP_TICKERS: List[str] = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","LT.NS","KOTAKBANK.NS",
    "HINDUNILVR.NS","AXISBANK.NS","BAJFINANCE.NS","SUNPHARMA.NS","ASIANPAINT.NS",
    "MARUTI.NS","HCLTECH.NS","TITAN.NS","WIPRO.NS","POWERGRID.NS",
]
MID_CAP_TICKERS: List[str] = [
    "TRENT.NS","DABUR.NS","PERSISTENT.NS","MUTHOOTFIN.NS","CUMMINSIND.NS",
    "TORNTPHARM.NS","LAURUSLABS.NS","LTTS.NS","KPITTECH.NS","AUBANK.NS",
    "BANDHANBNK.NS","PIDILITIND.NS","SIEMENS.NS","CHOLAFIN.NS","TATAPOWER.NS",
    "COFORGE.NS","ESCORTS.NS","HAVELLS.NS","GLAND.NS","MFSL.NS",
]
SMALL_CAP_TICKERS: List[str] = [
    "IRCTC.NS","JUBLFOOD.NS","OFSS.NS","BALKRISIND.NS","AARTIIND.NS",
    "RAILTEL.NS","EMAMILTD.NS","COLPAL.NS","MGL.NS","IGL.NS",
    "NAUKRI.NS","LUPIN.NS","BOSCHLTD.NS","BERGEPAINT.NS","MOTHERSON.NS",
    "VOLTAS.NS","DLF.NS","CONCOR.NS","AMBUJACEM.NS","LTIM.NS",
]

DATASET_REGISTRY: Dict[str, List[str]] = {
    "nifty50":   NIFTY50_TICKERS,
    "nifty100":  list(dict.fromkeys(NIFTY50_TICKERS + NIFTY100_EXTRA_TICKERS)),
    "large_cap": LARGE_CAP_TICKERS,
    "mid_cap":   MID_CAP_TICKERS,
    "small_cap": SMALL_CAP_TICKERS,
    **{f"sector_{k.lower()}": v for k, v in SECTOR_TICKERS.items()},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2)  DATA LOADING  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_yf_close(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        close = (data["Close"] if "Close" in lvl0
                 else data.xs("Close", axis=1, level=0, drop_level=True)).copy()
    else:
        close = data[["Close"]].copy() if "Close" in data.columns else data.copy()
    return close.to_frame() if isinstance(close, pd.Series) else close


def fetch_price_data(
    tickers: Sequence[str],
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    min_coverage: float = 0.7,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.strip() for t in tickers if t and str(t).strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")
    print(f"  Fetching {len(tickers)} tickers [{start} → {end}] …")
    raw = yf.download(
        tickers=tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True, repair=True,
    )
    close = _flatten_yf_close(raw)
    if close.empty:
        raise RuntimeError("Yahoo Finance returned no usable price data.")
    min_obs = int(np.ceil(min_coverage * len(close)))
    close = close.dropna(axis=1, thresh=min_obs).ffill().dropna()
    if close.empty:
        raise RuntimeError("All tickers dropped after cleaning.")
    print(f"  Retained {close.shape[1]} tickers over {close.shape[0]} trading days.")
    return close


def load_dataset(name_or_tickers, start="2015-01-01", end="2025-01-01"):
    tickers = (DATASET_REGISTRY[name_or_tickers]
               if isinstance(name_or_tickers, str) else list(name_or_tickers))
    prices = fetch_price_data(tickers, start=start, end=end)
    returns = prices.pct_change().dropna()
    spins = np.sign(returns.values)
    return prices.iloc[1:].reset_index(drop=True), returns, spins, list(prices.columns)

# ─────────────────────────────────────────────────────────────────────────────
# 3)  ADAPTIVE PCA  +  3-STATE SPIN ENCODER
#     FIX 2.1: Added spin_encoding="zscore" to mitigate in-sample quantile
#     bias.  Z-score thresholds remain meaningful across regime changes
#     because they measure deviation in units of training volatility.
# ─────────────────────────────────────────────────────────────────────────────

class PCAPreprocessor:

    def __init__(
        self,
        n_pca: Union[int, str] = "auto",
        pca_variance_target: float = 0.75,
        min_pca: int = 3,
        spin_quantile: float = 0.33,
        spin_encoding: str = "quantile",   # FIX 2.1 — new parameter
    ):
        self.n_pca_spec = n_pca
        self.pca_variance_target = pca_variance_target
        self.min_pca = min_pca
        self.spin_quantile = float(np.clip(spin_quantile, 0.05, 0.45))
        if spin_encoding not in ("quantile", "zscore"):
            raise ValueError("spin_encoding must be 'quantile' or 'zscore'.")
        self.spin_encoding = spin_encoding

        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_chosen_: int = 0

        # Thresholds in PC space — semantics differ by encoding mode:
        #   quantile:  empirical percentiles of training scores
        #   zscore:    ±z* in standardised units (per-component std stored)
        self._spin_lo: Optional[np.ndarray] = None   # (n_pca,)
        self._spin_hi: Optional[np.ndarray] = None   # (n_pca,)
        self._pc_std:  Optional[np.ndarray] = None   # (n_pca,) — zscore only

    def fit(self, returns: np.ndarray) -> "PCAPreprocessor":
        T, N = returns.shape
        self.mean_ = returns.mean(axis=0)
        centered = returns - self.mean_
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)

        total_var = (s ** 2).sum() + 1e-12
        self.explained_variance_ratio_ = s ** 2 / total_var
        cum_ev = np.cumsum(self.explained_variance_ratio_)
        hard_cap = int(np.clip(max(self.min_pca, T // 5), self.min_pca, N - 1))

        if self.n_pca_spec == "auto":
            n_var = int(np.searchsorted(cum_ev, self.pca_variance_target) + 1)
        else:
            n_var = int(self.n_pca_spec)

        n_chosen = int(np.clip(min(n_var, hard_cap), self.min_pca, N - 1))
        self.n_components_chosen_ = n_chosen
        self.components_ = Vt[:n_chosen]          # (n_pca, N)

        pc_scores = centered @ self.components_.T  # (T, n_pca)

        if self.spin_encoding == "quantile":
            # Original: fixed empirical quantiles from training window.
            q_lo = self.spin_quantile * 100
            q_hi = (1.0 - self.spin_quantile) * 100
            self._spin_lo = np.percentile(pc_scores, q_lo, axis=0)
            self._spin_hi = np.percentile(pc_scores, q_hi, axis=0)
            self._pc_std  = None

        else:  # "zscore"  — FIX 2.1
            # Compute z* such that P(|Z| > z*) = spin_quantile under N(0,1).
            # E.g. spin_quantile=0.33 → z* ≈ 0.44 (equal thirds by normal approx).
            z_star = float(_scipy_norm.ppf(1.0 - self.spin_quantile))
            self._pc_std  = pc_scores.std(axis=0) + 1e-8   # (n_pca,)
            # Store thresholds in raw score units (mean=0 since centred PCA).
            self._spin_lo = -z_star * self._pc_std
            self._spin_hi = +z_star * self._pc_std

        return self

    @property
    def explained_variance_total(self) -> float:
        if self.explained_variance_ratio_ is None:
            return 0.0
        return float(self.explained_variance_ratio_[:self.n_components_chosen_].sum())

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """Project returns → PC scores.  Shape: (..., n_pca)."""
        return (returns - self.mean_) @ self.components_.T

    def transform_spins(self, returns: np.ndarray) -> np.ndarray:
        scores = self.transform(returns)      # (..., n_pca)
        spins  = np.zeros_like(scores)
        spins[scores < self._spin_lo] = -1.0
        spins[scores > self._spin_hi] = +1.0
        return spins

    def inverse_signal(self, pc_signal: np.ndarray) -> np.ndarray:
        """Map a signal in PC space back to asset space.  (n_pca,) → (N,)."""
        return self.components_.T @ pc_signal   # (N,)

# ─────────────────────────────────────────────────────────────────────────────
# 4)  3-STATE BLUME–CAPEL MEAN-FIELD MODEL
#     FIX 1.1: Renamed from "MeanFieldPotts3" to "MeanFieldBlumeCapel".
#              Old name kept as alias for backward compatibility.
#     FIX 1.2: Added d-centering after each update for numerical conditioning.
#              (See header note — full gauge symmetry claim is incorrect.)
# ─────────────────────────────────────────────────────────────────────────────

def _sym_zero_diag(M: np.ndarray) -> np.ndarray:
    """Symmetrise a matrix and zero its diagonal in-place."""
    S = (M + M.T) / 2.0
    np.fill_diagonal(S, 0.0)
    return S


class MeanFieldBlumeCapel:
    def __init__(self, N: int, max_iter: int = 25, tol: float = 1e-4):
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.h = np.zeros(N)
        self.d = np.zeros(N)
        self.J = np.zeros((N, N))

    def _marginals(
        self, h_eff: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """3-state marginals for all nodes given effective field h_eff."""
        log_pos = self.d + h_eff
        log_neg = self.d - h_eff
        log_zer = np.zeros(self.N)

        mx    = np.maximum(np.maximum(log_pos, log_neg), log_zer)
        e_pos = np.exp(log_pos - mx)
        e_neg = np.exp(log_neg - mx)
        e_zer = np.exp(log_zer - mx)
        Z     = e_pos + e_neg + e_zer

        p_pos = e_pos / Z
        p_neg = e_neg / Z
        p_zer = e_zer / Z
        m     = p_pos - p_neg
        q2    = p_pos + p_neg
        return p_neg, p_zer, p_pos, m, q2

    def magnetisations(self) -> np.ndarray:
        """Fixed-point mean-field iteration; returns ⟨s⟩ ∈ (−1,+1)^N."""
        m = np.zeros(self.N)
        for _ in range(self.max_iter):
            h_eff = self.h + self.J @ m
            _, _, _, m_new, _ = self._marginals(h_eff)
            if np.max(np.abs(m_new - m)) < self.tol:
                return m_new
            m = m_new
        return m_new

    def log_partition_approx(self, m: np.ndarray) -> float:
        h_eff  = self.h + self.J @ m
        log_pos = self.d + h_eff
        log_neg = self.d - h_eff
        mx     = np.maximum(np.maximum(log_pos, log_neg), 0.0)
        log_Zi = mx + np.log(
            np.exp(log_pos - mx) + np.exp(-mx) + np.exp(log_neg - mx)
        )
        i_idx, j_idx = np.triu_indices(self.N, k=1)
        correction = np.sum(self.J[i_idx, j_idx] * m[i_idx] * m[j_idx])
        return float(log_Zi.sum() - correction)

    def pseudo_log_likelihood_gradient(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_eff  = self.h + self.J @ x
        _, _, _, m, q2 = self._marginals(h_eff)

        res_m  = x      - m
        res_q2 = x ** 2 - q2

        gh = res_m
        gd = res_q2
        gJ = np.outer(res_m, x) + np.outer(x, res_m)
        np.fill_diagonal(gJ, 0.0)
        return gh, gd, gJ

    def emission_log_prob(self, x: np.ndarray) -> float:
        m    = self.magnetisations()
        logZ = self.log_partition_approx(m)

        linear    = float(np.dot(self.h, x))
        quadratic = float(np.dot(self.d, x ** 2))
        i_idx, j_idx = np.triu_indices(self.N, k=1)
        pairwise  = float(np.sum(self.J[i_idx, j_idx] * x[i_idx] * x[j_idx]))
        return linear + quadratic + pairwise - logZ

    def emission_prob(self, x: np.ndarray) -> float:
        """exp(log p(x)); clipped to avoid underflow."""
        return float(np.exp(np.clip(self.emission_log_prob(x), -500, 0)))

    def expected_spins(self) -> np.ndarray:
        """Return mean-field magnetisations ⟨s⟩ ∈ (−1,+1)^N."""
        return self.magnetisations()


# Backward-compatibility alias (old code using MeanFieldPotts3 still works).
MeanFieldPotts3 = MeanFieldBlumeCapel

# ─────────────────────────────────────────────────────────────────────────────
# 5)  HMM FILTER — LOG-SPACE FORWARD ALGORITHM
#     FIX 2.3: Replaced linear-space forward step with a numerically stable
#     log-space implementation.  This eliminates the underflow→uniform-belief→
#     parameter-drift cascade described in the review.
#     Added: emission_temperature τ ≥ 1 to dampen overconfident approximate
#     emissions near MF criticality.
# ─────────────────────────────────────────────────────────────────────────────

class HMMFilter:

    def __init__(
        self,
        K: int,
        A: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
        emission_temperature: float = 1.0,   # FIX 2.3 — new parameter
    ):
        self.K = K
        self.A = A if A is not None else _sticky_transition(K)
        self.pi = pi if pi is not None else np.ones(K) / K
        self.emission_temperature = max(1.0, float(emission_temperature))

        # Internal state kept in LOG-SPACE for numerical stability.
        self._log_alpha: np.ndarray = np.log(self.pi + 1e-300)

    @property
    def _alpha(self) -> np.ndarray:
        """Return current belief as normalised probabilities (for compatibility)."""
        log_a = self._log_alpha - logsumexp(self._log_alpha)
        return np.exp(log_a)

    @_alpha.setter
    def _alpha(self, value: np.ndarray) -> None:
        """Allow external code to set _alpha in probability space."""
        self._log_alpha = np.log(np.clip(value, 1e-300, None))

    def reset(self) -> None:
        self._log_alpha = np.log(self.pi + 1e-300)

    def step(self, emit: np.ndarray) -> np.ndarray:
        
        log_emit = np.log(np.clip(emit, 1e-300, None)) / self.emission_temperature

        # Log-space prediction: log p(s_t) = log ∑_i A[i, s_t] · alpha[i]
        log_A = np.log(self.A + 1e-300)              # (K, K)
        # log_A[i, j] + log_alpha[i] for each i, then logsumexp over i → (K,)
        log_pred = logsumexp(
            log_A + self._log_alpha[:, None], axis=0  # broadcast (K,1)+(K,K)
        )

        # Update: log alpha_new[j] = log_pred[j] + log_emit[j]
        log_alpha_new = log_pred + log_emit

        # Normalise in log-space (subtract log-sum to get log-probabilities).
        log_alpha_new = log_alpha_new - logsumexp(log_alpha_new)
        self._log_alpha = log_alpha_new

        return self._alpha.copy()

    def predict_regime(self, steps: int = 1) -> np.ndarray:
        """Multi-step ahead regime probability forecast."""
        log_b = self._log_alpha.copy()
        log_A = np.log(self.A + 1e-300)
        for _ in range(steps):
            log_b = logsumexp(log_A + log_b[:, None], axis=0)
        log_b = log_b - logsumexp(log_b)
        return np.exp(log_b)


def _sticky_transition(K: int, diag: float = 0.70) -> np.ndarray:
    off = (1.0 - diag) / max(K - 1, 1)
    A = np.full((K, K), off)
    np.fill_diagonal(A, diag)
    return A

# ─────────────────────────────────────────────────────────────────────────────
# 6)  ONLINE UPDATERS
#     FIX 1.2 (partial): Added post-update d-centering in OMDUpdater3 to
#     prevent all d_i drifting together and to improve interpretability.
#     This does NOT exploit a gauge symmetry (which doesn't exist here) but
#     is a useful mean-zero constraint that stabilises learning.
# ─────────────────────────────────────────────────────────────────────────────

class OMDUpdater3:
    def __init__(
        self,
        N: int,
        eta: float    = 0.01,
        h_max: float  = 2.0,
        J_max: float  = 1.0,
        d_max: float  = 2.0,
        l2: float     = 1e-3,
        l1_J: float   = 0.05,
        l2_d: float   = 1e-3,
        center_d: bool = True,    # FIX 1.2 — new parameter
    ):
        self.N = N
        self.eta  = eta
        self.h_max = h_max
        self.J_max = J_max
        self.d_max = d_max
        self.l2    = l2
        self.l1_J  = l1_J
        self.l2_d  = l2_d
        self.center_d = center_d
        self._Gh = np.ones(N)
        self._Gd = np.ones(N)
        self._GJ = np.ones((N, N))

    @staticmethod
    def _soft(x: np.ndarray, lam: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

    def step(self, model: MeanFieldBlumeCapel, x: np.ndarray, w: float = 1.0) -> None:
        """
        One AdaGrad step maximising weighted PLL for spin vector x with
        HMM responsibility weight w.
        """
        gh, gd, gJ = model.pseudo_log_likelihood_gradient(x)

        ng_h = -w * gh + self.l2   * model.h
        ng_d = -w * gd + self.l2_d * model.d
        ng_J = -w * gJ + self.l2   * model.J

        self._Gh += ng_h ** 2
        self._Gd += ng_d ** 2
        self._GJ += ng_J ** 2

        h_new = model.h - self.eta * ng_h / (np.sqrt(self._Gh) + 1e-8)
        d_new = model.d - self.eta * ng_d / (np.sqrt(self._Gd) + 1e-8)
        J_new = model.J - self.eta * ng_J / (np.sqrt(self._GJ) + 1e-8)

        J_new = self._soft(J_new, self.l1_J)
        h_new = np.clip(h_new, -self.h_max, self.h_max)
        d_new = np.clip(d_new, -self.d_max, self.d_max)

        # FIX 1.2: Mean-zero constraint on d for numerical conditioning.
        # Prevents all d_i drifting together (correlated activity gradients).
        if self.center_d:
            d_new = d_new - d_new.mean()

        Jf = np.linalg.norm(J_new, "fro")
        if Jf > self.J_max:
            J_new *= self.J_max / (Jf + 1e-12)

        model.h = h_new
        model.d = d_new
        model.J = _sym_zero_diag(J_new)


class TransitionEG:
    """Exponentiated Gradient update for the HMM transition matrix."""

    def __init__(self, K: int, eta_A: float = 0.05):
        self.K = K
        self.eta_A = eta_A

    def step(
        self,
        A: np.ndarray,
        alpha_prev: np.ndarray,
        emit: np.ndarray,
    ) -> np.ndarray:
        xi  = alpha_prev[:, None] * A * emit[None, :]
        xi /= xi.sum(axis=1, keepdims=True) + 1e-300
        gA  = -xi / (A + 1e-300)
        lA  = np.log(A + 1e-300) - self.eta_A * gA
        lA -= lA.max(axis=1, keepdims=True)
        An  = np.exp(lA)
        return An / An.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7)  RARSI MODEL  (Blume–Capel backbone)
#     FIX 2.4: Improved convergence tracking — rolling average PLL (not
#     cumulative), warmup period before stopping, and clearer criterion.
# ─────────────────────────────────────────────────────────────────────────────

class RARSIModel:
    _RESP_FLOOR = 0.01

    def __init__(
        self,
        N: int,
        K: int        = 3,
        eta: float    = 0.01,
        eta_A: float  = 0.05,
        h_max: float  = 2.0,
        J_max: float  = 1.0,
        d_max: float  = 2.0,
        l2: float     = 1e-3,
        l1_J: float   = 0.05,
        l2_d: float   = 1e-3,
        mf_iter: int  = 25,
        n_passes: int = 15,
        patience: int = 3,
        n_warmup_passes: int = 2,          # FIX 2.4 — new parameter
        center_d: bool = True,             # passed through to OMDUpdater3
        emission_temperature: float = 1.0, # passed through to HMMFilter
    ):
        self.N = N
        self.K = K
        self.n_passes = n_passes
        self.patience = patience
        self.n_warmup_passes = max(1, n_warmup_passes)

        self.models: List[MeanFieldBlumeCapel] = []
        for k in range(K):
            p = MeanFieldBlumeCapel(N, max_iter=mf_iter)
            rng = np.random.default_rng(42 + k)

            h_bias = [+0.30, -0.30,  0.00][k % 3]
            p.h = rng.normal(h_bias, 0.10, N)

            d_bias = [+0.30, +0.30, -0.30][k % 3]
            p.d = rng.normal(d_bias, 0.05, N)

            j_scale = [0.02, 0.02, 0.08][k % 3]
            p.J = _sym_zero_diag(rng.normal(0.0, j_scale, (N, N)))

            self.models.append(p)

        # Backward-compat alias.
        self.potts = self.models

        self.hmm      = HMMFilter(K, emission_temperature=emission_temperature)
        self.omds     = [
            OMDUpdater3(N, eta, h_max, J_max, d_max, l2, l1_J, l2_d,
                        center_d=center_d)
            for _ in range(K)
        ]
        self.trans_eg    = TransitionEG(K, eta_A)
        self.belief_path: List[np.ndarray] = []

    def fit(self, spins: np.ndarray) -> "RARSIModel":
        T = len(spins)
        pll_history: List[float] = []   # per-pass average PLL
        no_improve = 0

        for pass_idx in range(self.n_passes):
            self.hmm.reset()
            self.belief_path = []
            pass_pll = 0.0

            for t in range(T):
                x  = spins[t]
                ap = self._alpha_copy()

                emit = np.array([p.emission_prob(x) for p in self.models])
                wt   = self.hmm.step(emit)
                wt   = np.maximum(wt, self._RESP_FLOOR)
                wt  /= wt.sum()
                self.hmm._alpha = wt
                self.belief_path.append(wt)

                for k, (p, omd) in enumerate(zip(self.models, self.omds)):
                    omd.step(p, x, w=wt[k])

                self.hmm.A = self.trans_eg.step(self.hmm.A, ap, emit)

                for k, p in enumerate(self.models):
                    pass_pll += wt[k] * p.emission_log_prob(x)

            avg_pll = pass_pll / max(T, 1)
            pll_history.append(avg_pll)

            # FIX 2.4: Only check convergence after warmup.
            if pass_idx >= self.n_warmup_passes:
                prev_avg = np.mean(pll_history[-self.patience - 1: -1]) if len(pll_history) > 1 else -np.inf
                if avg_pll - prev_avg < 1e-4:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve >= self.patience:
                    break

        return self

    def _alpha_copy(self) -> np.ndarray:
        return self.hmm._alpha.copy()

    def forward_filter_step(self, x: np.ndarray) -> None:
        emit = np.array([p.emission_prob(x) for p in self.models])
        wt   = self.hmm.step(emit)
        wt   = np.maximum(wt, self._RESP_FLOOR)
        wt  /= wt.sum()
        self.hmm._alpha = wt

    def predict_expected_spins(self, horizon: int) -> np.ndarray:
        """Horizon-step ahead expected spins: E[s|H] = ∑_k p(k|H)·⟨s⟩_k."""
        fb = self.hmm.predict_regime(steps=horizon)
        return sum(fb[k] * self.models[k].expected_spins() for k in range(self.K))

    def current_regime(self) -> int:
        return int(np.argmax(self.hmm._alpha))

    def current_belief(self) -> np.ndarray:
        return self.hmm._alpha.copy()

# ─────────────────────────────────────────────────────────────────────────────
# 8)  ADAPTIVE BLENDER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveBlender:
    def __init__(
        self,
        ic_window: int      = 60,
        min_weight: float   = 0.0,
        base_ising_w: float = 0.0,
        base_mom_w: float   = 1.0,
    ):
        self.ic_window    = ic_window
        self.min_weight   = min_weight
        self.base_ising_w = base_ising_w
        self.base_mom_w   = base_mom_w

        self._ic_ising: deque = deque(maxlen=ic_window)
        self._ic_mom:   deque = deque(maxlen=ic_window)
        self._pending_ising: deque = deque(maxlen=ic_window + 20)
        self._pending_mom:   deque = deque(maxlen=ic_window + 20)
        self._pending_step:  deque = deque(maxlen=ic_window + 20)
        self._step: int = 0

    def record_signal(self, bc_sig: np.ndarray, mom_sig: np.ndarray) -> None:
        self._pending_ising.append(bc_sig.copy())
        self._pending_mom.append(mom_sig.copy())
        self._pending_step.append(self._step)
        self._step += 1

    def observe_outcome(self, realized: np.ndarray, horizon: int) -> None:
        target_step = self._step - horizon
        if not self._pending_step:
            return
        for i in range(len(self._pending_step) - 1, -1, -1):
            if self._pending_step[i] <= target_step:
                si = self._pending_ising[i]
                sm = self._pending_mom[i]
                if np.abs(si).sum() > 1e-8 and np.abs(realized).sum() > 1e-8:
                    ri, _ = spearmanr(si, realized)
                    rm, _ = spearmanr(sm, realized)
                    if not np.isnan(ri): self._ic_ising.append(float(ri))
                    if not np.isnan(rm): self._ic_mom.append(float(rm))
                break

    def weights(self) -> Tuple[float, float]:
        if len(self._ic_ising) < 10:
            return self.base_ising_w, self.base_mom_w
        wi    = max(float(np.mean(self._ic_ising)), self.min_weight)
        wm    = max(float(np.mean(self._ic_mom)),   self.min_weight)
        total = wi + wm
        if total < 1e-8:
            return self.base_ising_w, self.base_mom_w
        return wi / total, wm / total

    @property
    def mean_ic_potts(self) -> float:
        return float(np.mean(self._ic_ising)) if self._ic_ising else float("nan")

    @property
    def mean_ic_mom(self) -> float:
        return float(np.mean(self._ic_mom)) if self._ic_mom else float("nan")

    mean_ic_ising = mean_ic_potts   # legacy alias

# ─────────────────────────────────────────────────────────────────────────────
# 9)  REGIME GATE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeGate:
    def __init__(
        self,
        K: int         = 3,
        window: int    = 120,
        min_accuracy: float = 0.51,
        warmup: int    = 40,
        gate_scale: float   = 0.35,
        entry_threshold: float = 0.12,
    ):
        self.K = K; self.window = window
        self.min_accuracy = min_accuracy; self.warmup = warmup
        self.gate_scale = gate_scale; self.entry_threshold = entry_threshold
        self._hits: Dict[int, deque] = {k: deque(maxlen=window) for k in range(K)}

    def observe(self, regime: int, signal: np.ndarray, realized: np.ndarray) -> None:
        active = np.abs(signal) > self.entry_threshold
        if not active.any():
            return
        correct_frac = float(
            (np.sign(signal[active]) == np.sign(realized[active])).mean()
        )
        self._hits[regime].append(correct_frac)

    def scale_for(self, regime: int) -> float:
        hits = self._hits[regime]
        if len(hits) < self.warmup:
            return 1.0
        return 1.0 if float(np.mean(hits)) >= self.min_accuracy else max(self.gate_scale, 0.35)

    def accuracy(self, regime: int) -> float:
        h = self._hits[regime]
        return float(np.mean(h)) if h else float("nan")

    def n_observations(self, regime: int) -> int:
        return len(self._hits[regime])

# ─────────────────────────────────────────────────────────────────────────────
# 10)  ROLLING WINDOW TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class RollingWindowTrainer:

    def __init__(
        self,
        train_window: int      = 126,
        horizon: int           = 5,
        K: int                 = 3,
        refit_freq: int        = 5,
        eta: float             = 0.01,
        eta_A: float           = 0.05,
        h_max: float           = 2.0,
        J_max: float           = 1.0,
        d_max: float           = 2.0,
        l2: float              = 1e-3,
        l1_J: float            = 0.05,
        l2_d: float            = 1e-3,
        mf_iter: int           = 25,
        n_passes: int          = 15,
        patience: int          = 3,
        n_warmup_passes: int   = 2,           # FIX 2.4
        center_d: bool         = True,        # FIX 1.2
        emission_temperature: float = 1.0,    # FIX 2.3
        n_pca: Union[int, str] = "auto",
        pca_variance_target: float = 0.75,
        spin_quantile: float   = 0.33,
        spin_encoding: str     = "quantile",  # FIX 2.1 ("zscore" recommended)
        ic_window: int         = 60,
        base_momentum_w: float = 1.0,
        mom_windows: Tuple     = (20, 60, 120),
        use_regime_gate: bool      = False,
        min_regime_accuracy: float = 0.51,
        gate_scale: float      = 0.35,
        gate_warmup: int       = 40,
        entry_threshold: float = 0.20,
    ):
        self.train_window = max(21, min(train_window, 504))
        self.horizon      = horizon
        self.K            = K
        self.refit_freq   = refit_freq
        self.eta          = eta
        self.eta_A        = eta_A
        self.h_max        = h_max
        self.J_max        = J_max
        self.d_max        = d_max
        self.l2           = l2
        self.l1_J         = l1_J
        self.l2_d         = l2_d
        self.mf_iter      = mf_iter
        self.n_passes     = n_passes
        self.patience     = patience
        self.n_warmup_passes     = n_warmup_passes
        self.center_d            = center_d
        self.emission_temperature = emission_temperature
        self.n_pca               = n_pca
        self.pca_variance_target = pca_variance_target
        self.spin_quantile       = spin_quantile
        self.spin_encoding       = spin_encoding
        self.ic_window           = ic_window
        self.base_momentum_w     = base_momentum_w
        self.mom_windows         = mom_windows
        self.use_regime_gate         = use_regime_gate
        self.min_regime_accuracy     = min_regime_accuracy
        self.gate_scale          = gate_scale
        self.gate_warmup         = gate_warmup
        self.entry_threshold     = entry_threshold

        # Outputs populated by .run()
        self.signals:        List[np.ndarray] = []
        self.potts_signals:  List[np.ndarray] = []
        self.mom_signals:    List[np.ndarray] = []
        self.ising_signals:  List[np.ndarray] = []   # alias
        self.ising_weights:  List[float]      = []
        self.mom_weights:    List[float]      = []
        self.regime_labels:  List[int]        = []
        self.belief_history: List[np.ndarray] = []
        self.gate_scales:    List[float]      = []
        self.refit_steps:    List[int]        = []
        self.signal_dates:   List[pd.Timestamp] = []
        self.n_pca_history:  List[int]        = []
        self.ev_history:     List[float]      = []
        self.blender:   Optional[AdaptiveBlender] = None
        self.regime_gate: Optional[RegimeGate]   = None

    def _build_mom_signal(self, rets_window: np.ndarray) -> np.ndarray:
        N = rets_window.shape[1]
        sigs = []
        for w in self.mom_windows:
            w = min(w, len(rets_window))
            if w < 2:
                continue
            m   = rets_window[-w:].mean(axis=0)
            std = m.std() + 1e-8
            sigs.append((m - m.mean()) / std)
        if not sigs:
            return np.zeros(N)
        combined = np.mean(sigs, axis=0)
        mx = np.abs(combined).max()
        return np.clip(combined / (mx + 1e-8), -1, 1)

    def run(self, rets: pd.DataFrame, verbose: bool = True) -> "RollingWindowTrainer":
        T, N  = rets.shape
        arr   = rets.values
        dates = rets.index

        self.blender = AdaptiveBlender(
            ic_window    = self.ic_window,
            base_ising_w = max(0.0, 1.0 - self.base_momentum_w),
            base_mom_w   = self.base_momentum_w,
        )
        self.regime_gate = (
            RegimeGate(
                K=self.K, window=120,
                min_accuracy=self.min_regime_accuracy,
                warmup=self.gate_warmup,
                gate_scale=self.gate_scale,
                entry_threshold=self.entry_threshold,
            ) if self.use_regime_gate else None
        )

        model: Optional[RARSIModel]       = None
        pca:   Optional[PCAPreprocessor]  = None
        steps  = range(self.train_window, T - self.horizon + 1)
        it     = tqdm(
            steps,
            desc=(
                f"BlumeCapel W={self.train_window} H={self.horizon} K={self.K} "
                f"enc={self.spin_encoding}"
            ),
            disable=not verbose,
        )

        for t in it:
            step_idx   = t - self.train_window
            do_refit   = (step_idx % self.refit_freq == 0) or (model is None)
            train_rets = arr[t - self.train_window: t]

            if step_idx >= self.horizon and len(self.signals) >= self.horizon:
                past_idx = t - self.horizon
                future   = arr[past_idx + 1: past_idx + 1 + self.horizon]
                realized = np.prod(1 + future, axis=0) - 1
                self.blender.observe_outcome(realized, self.horizon)
                if self.regime_gate is not None:
                    past_signal = self.signals[-self.horizon]
                    past_regime = self.regime_labels[-self.horizon]
                    self.regime_gate.observe(past_regime, past_signal, realized)

            if do_refit:
                pca = PCAPreprocessor(
                    n_pca               = self.n_pca,
                    pca_variance_target = self.pca_variance_target,
                    spin_quantile       = self.spin_quantile,
                    spin_encoding       = self.spin_encoding,   # FIX 2.1
                ).fit(train_rets)

                train_spins = pca.transform_spins(train_rets)
                eff_dim     = train_spins.shape[1]

                model = RARSIModel(
                    N        = eff_dim,
                    K        = self.K,
                    eta      = self.eta,
                    eta_A    = self.eta_A,
                    h_max    = self.h_max,
                    J_max    = self.J_max,
                    d_max    = self.d_max,
                    l2       = self.l2,
                    l1_J     = self.l1_J,
                    l2_d     = self.l2_d,
                    mf_iter  = self.mf_iter,
                    n_passes = self.n_passes,
                    patience = self.patience,
                    n_warmup_passes     = self.n_warmup_passes,     # FIX 2.4
                    center_d            = self.center_d,            # FIX 1.2
                    emission_temperature = self.emission_temperature, # FIX 2.3
                )
                model.fit(train_spins)
                self.refit_steps.append(t)

                if verbose and step_idx % (self.refit_freq * 15) == 0:
                    it.set_postfix({
                        "n_pca": eff_dim,
                        "EV":    f"{pca.explained_variance_total:.0%}",
                        "IC_p":  f"{self.blender.mean_ic_potts:.3f}",
                        "IC_m":  f"{self.blender.mean_ic_mom:.3f}",
                    })
            else:
                x_new = pca.transform_spins(arr[t - 1: t])[0]
                model.forward_filter_step(x_new)

            pc_sig       = model.predict_expected_spins(self.horizon)
            potts_signal = pca.inverse_signal(pc_sig)
            mom_signal   = self._build_mom_signal(train_rets)

            self.blender.record_signal(potts_signal, mom_signal)
            wi, wm = self.blender.weights()
            blended = wi * potts_signal + wm * mom_signal
            mx      = np.abs(blended).max()
            signal  = blended / (mx + 1e-8) if mx > 1e-8 else blended

            regime  = model.current_regime()
            g_scale = self.regime_gate.scale_for(regime) if self.regime_gate else 1.0
            signal  = signal * g_scale

            self.signals.append(signal)
            self.potts_signals.append(potts_signal.copy())
            self.ising_signals.append(potts_signal.copy())
            self.mom_signals.append(mom_signal.copy())
            self.ising_weights.append(wi)
            self.mom_weights.append(wm)
            self.regime_labels.append(regime)
            self.belief_history.append(model.current_belief())
            self.gate_scales.append(g_scale)
            self.signal_dates.append(dates[t])
            self.n_pca_history.append(pca.n_components_chosen_)
            self.ev_history.append(pca.explained_variance_total)

        print(
            f"  {len(self.signals)} signals | {len(self.refit_steps)} refits | "
            f"avg n_pca={np.mean(self.n_pca_history):.1f} | "
            f"avg EV={np.mean(self.ev_history):.1%} | "
            f"enc={self.spin_encoding}"
        )
        print(
            f"  Final IC → BlumeCapel: {self.blender.mean_ic_potts:.4f} | "
            f"Mom: {self.blender.mean_ic_mom:.4f}"
        )
        if self.regime_gate:
            gate_str = " | ".join(
                f"R{r}: {self.regime_gate.accuracy(r):.1%} "
                f"(n={self.regime_gate.n_observations(r)})"
                for r in range(self.K)
            )
            print(f"  Gate accuracy → {gate_str}")
        return self

# ─────────────────────────────────────────────────────────────────────────────
# 11)  PORTFOLIO ENGINE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tz is not None else idx

def _normalize_ts(ts) -> pd.Timestamp:
    return ts.tz_localize(None) if (hasattr(ts, "tz") and ts.tz is not None) else ts


class PortfolioEngine:
    def __init__(
        self,
        entry_threshold: float  = 0.20,
        max_position: float     = 0.10,
        cost_bps: float         = 10.0,
        weight_mode: str        = "long_only",
        vol_window: int         = 21,
        min_gross_signal: float = 0.05,
        short_scale: float      = 0.0,
        vol_target: Optional[float] = 0.15,
        vol_lookback: int       = 63,
        max_leverage: float     = 2.0,
        regime_confidence_scaling: bool = False,
        min_conf_scale: float   = 0.35,
    ):
        self.entry_threshold  = entry_threshold
        self.max_position     = max_position
        self.cost_bps         = cost_bps / 10_000.0
        self.weight_mode      = weight_mode
        self.vol_window       = vol_window
        self.min_gross_signal = min_gross_signal
        self.short_scale      = short_scale
        self.vol_target       = vol_target
        self.vol_lookback     = vol_lookback
        self.max_leverage     = max_leverage
        self.regime_confidence_scaling = regime_confidence_scaling
        self.min_conf_scale   = min_conf_scale

        self.weights_history: List[np.ndarray] = []
        self.pnl_series:      pd.Series = pd.Series(dtype=float)
        self.turnover_series: pd.Series = pd.Series(dtype=float)
        self.gross_exposure:  pd.Series = pd.Series(dtype=float)
        self.net_exposure:    pd.Series = pd.Series(dtype=float)

    def _build_weights(self, signal: np.ndarray, rets_hist: np.ndarray) -> np.ndarray:
        N = len(signal)
        w = np.zeros(N)
        active = np.abs(signal) > self.entry_threshold
        if not active.any():
            return w
        if np.abs(signal[active]).sum() < self.min_gross_signal:
            return w

        if self.weight_mode == "signal":
            w[active] = signal[active]
        elif self.weight_mode == "equal":
            longs  = active & (signal > 0)
            shorts = active & (signal < 0)
            if longs.any():  w[longs]  =  1.0 / longs.sum()
            if shorts.any(): w[shorts] = -1.0 / shorts.sum()
        elif self.weight_mode == "long_only":
            pos = active & (signal > 0)
            if pos.any(): w[pos] = signal[pos]
            w = np.maximum(w, 0)
        elif self.weight_mode == "vol_scaled":
            vols = (rets_hist.std(axis=0) + 1e-8
                    if rets_hist.shape[0] >= 2 else np.ones(N))
            w[active] = signal[active] / vols[active]
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

        w[w < 0] *= self.short_scale
        gross = np.abs(w).sum()
        if gross < 1e-8:
            return np.zeros(N)
        w /= gross
        w  = np.clip(w, -self.max_position, self.max_position)
        g2 = np.abs(w).sum()
        if g2 > 1e-8:
            w /= g2
        return w

    def _vol_scale(self, buf: List[float]) -> float:
        if self.vol_target is None or len(buf) < 20:
            return 1.0
        rv = np.array(buf[-self.vol_lookback:]).std() * np.sqrt(252)
        return float(np.clip(self.vol_target / (rv + 1e-6), 0.1, self.max_leverage))

    def _conf_scale(self, belief: np.ndarray) -> float:
        if not self.regime_confidence_scaling:
            return 1.0
        K    = len(belief)
        mx   = float(belief.max())
        unif = 1.0 / K
        conf = float(np.clip((mx - unif) / (1 - unif + 1e-8), 0, 1))
        return self.min_conf_scale + (1 - self.min_conf_scale) * conf

    def simulate(
        self,
        trainer: RollingWindowTrainer,
        rets: pd.DataFrame,
    ) -> "PortfolioEngine":
        signals   = trainer.signals
        dates_sig = trainer.signal_dates
        beliefs   = trainer.belief_history
        horizon   = trainer.horizon
        rets_arr  = rets.values
        rdn       = _normalize_tz(rets.index)
        T_full    = len(rets_arr)
        N         = rets_arr.shape[1]

        daily_pnl    = np.zeros(T_full)
        n_open       = np.zeros(T_full, dtype=np.int32)
        pnl_buf:     List[float]    = []
        sig_records: List[Tuple]    = []
        prev_w       = np.zeros(N)

        for sig, sd, belief in zip(signals, dates_sig, beliefs):
            sn = _normalize_ts(sd)
            if sn not in rdn:
                continue
            ti = rdn.get_loc(sn)
            if ti + horizon >= T_full:
                continue
            hs = max(0, ti - self.vol_window)
            w  = self._build_weights(sig, rets_arr[hs:ti])
            w  = w * self._conf_scale(belief) * self._vol_scale(pnl_buf)
            turn = float(np.abs(w - prev_w).sum())
            cost = turn * self.cost_bps
            wb   = w / horizon
            for d in range(1, horizon + 1):
                idx = ti + d
                if idx < T_full:
                    dr = float(np.dot(wb, rets_arr[idx]))
                    if d == 1:
                        dr -= cost
                    daily_pnl[idx] += dr
                    n_open[idx]    += 1
            if ti > 0:
                pnl_buf.append(daily_pnl[ti])
            self.weights_history.append(w)
            sig_records.append((ti, w, turn, sn))
            prev_w = w.copy()

        ai = np.where(n_open > 0)[0]
        if len(ai) == 0:
            return self
        self.pnl_series = pd.Series(daily_pnl[ai], index=rets.index[ai])
        if sig_records:
            od = [r[3] for r in sig_records]
            self.turnover_series = pd.Series([r[2] for r in sig_records], index=od)
            self.gross_exposure  = pd.Series([np.abs(r[1]).sum() for r in sig_records], index=od)
            self.net_exposure    = pd.Series([r[1].sum() for r in sig_records], index=od)
        return self

    def equity_curve(self, start_nav: float = 100.0) -> pd.Series:
        return start_nav * (1 + self.pnl_series).cumprod()

# ─────────────────────────────────────────────────────────────────────────────
# 12)  PREDICTION ACCURACY  (updated labels)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_accuracy(
    trainer: RollingWindowTrainer,
    rets: pd.DataFrame,
    entry_threshold: float = 0.20,
) -> Dict:
    signals    = trainer.signals
    dates      = trainer.signal_dates
    regimes    = trainer.regime_labels
    horizon    = trainer.horizon
    potts_sigs = trainer.potts_signals
    mom_sigs   = trainer.mom_signals
    rets_arr   = rets.values
    rdn        = _normalize_tz(rets.index)
    T          = len(rets_arr)
    K          = trainer.K

    all_corr, all_act               = [], []
    long_acc,  short_acc            = [], []
    str_list,  cor_list             = [], []
    reg_corr:  Dict[int, List]      = defaultdict(list)
    reg_tot:   Dict[int, List]      = defaultdict(list)
    ic_blended, ic_potts_l, ic_mom_l = [], [], []

    for sig, sd, regime, sp, sm in zip(
        signals, dates, regimes, potts_sigs, mom_sigs
    ):
        sn = _normalize_ts(sd)
        if sn not in rdn:
            continue
        ti = rdn.get_loc(sn)
        if ti + horizon >= T:
            continue
        future = rets_arr[ti + 1: ti + 1 + horizon]
        real   = np.prod(1 + future, axis=0) - 1

        for s_arr, ic_list in [(sig, ic_blended), (sp, ic_potts_l), (sm, ic_mom_l)]:
            if np.abs(s_arr).sum() > 1e-8 and np.abs(real).sum() > 1e-8:
                rho, _ = spearmanr(s_arr, real)
                if not np.isnan(rho):
                    ic_list.append(rho)

        active = np.abs(sig) > entry_threshold
        if not active.any():
            continue
        pd_  = np.sign(sig[active])
        rd_  = np.sign(real[active])
        corr = (pd_ == rd_)
        all_corr.append(int(corr.sum()))
        all_act.append(int(active.sum()))
        lm  = sig[active] > 0
        sm_ = ~lm
        if lm.any():  long_acc.append(float(corr[lm].mean()))
        if sm_.any(): short_acc.append(float(corr[sm_].mean()))
        reg_corr[regime].append(int(corr.sum()))
        reg_tot[regime].append(int(active.sum()))
        str_list.extend(np.abs(sig[active]).tolist())
        cor_list.extend(corr.tolist())

    if not all_act or sum(all_act) == 0:
        return {"Overall Accuracy": float("nan")}

    overall    = sum(all_corr) / sum(all_act)
    per_regime = {
        r: sum(reg_corr[r]) / sum(reg_tot[r])
        for r in sorted(reg_corr) if sum(reg_tot[r]) > 0
    }
    strs = np.array(str_list)
    cors = np.array(cor_list, dtype=float)
    top_acc = bot_acc = float("nan")
    if len(strs) >= 3:
        lo = np.percentile(strs, 33)
        hi = np.percentile(strs, 67)
        top_acc = cors[strs >= hi].mean()
        bot_acc = cors[strs <= lo].mean()

    def _ic_stats(arr):
        if not arr:
            return float("nan"), float("nan")
        a = np.array(arr)
        return float(a.mean()), float(a.mean() / (a.std() + 1e-8))

    ic_b, ir_b = _ic_stats(ic_blended)
    ic_p, ir_p = _ic_stats(ic_potts_l)
    ic_m, ir_m = _ic_stats(ic_mom_l)

    gate_stats = {}
    if trainer.regime_gate:
        gate_stats = {
            f"Gate acc R{r}": trainer.regime_gate.accuracy(r)
            for r in range(K)
        }
        n_gated = sum(1 for g in trainer.gate_scales if g < 1.0)
        gate_stats["Gated signals (%)"] = n_gated / max(len(trainer.gate_scales), 1)

    return {
        "Overall Accuracy":            overall,
        "Long Accuracy":               float(np.mean(long_acc)) if long_acc else float("nan"),
        "Short Accuracy":              float(np.mean(short_acc)) if short_acc else float("nan"),
        "Top-Tertile Signal Accuracy": float(top_acc),
        "Bot-Tertile Signal Accuracy": float(bot_acc),
        "Blended IC (Spearman)":       ic_b,
        "Blended IC-IR":               ir_b,
        "BlumeCapel IC":               ic_p,
        "BlumeCapel IC-IR":            ir_p,
        "Potts IC":                    ic_p,   # backward-compat alias
        "Ising IC":                    ic_p,   # backward-compat alias
        "Momentum IC":                 ic_m,
        "Momentum IC-IR":              ir_m,
        "Per-Regime Accuracy":         per_regime,
        "Total Active Predictions":    int(sum(all_act)),
        "Total Correct":               int(sum(all_corr)),
        "Spin Encoding":               trainer.spin_encoding,
        **gate_stats,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 13)  METRICS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pnl: pd.Series, risk_free_annual: float = 0.065) -> Dict:
    if len(pnl) == 0:
        return {}
    ppa = 252
    r   = pnl.values
    n   = len(r)
    eq  = (1 + r).cumprod()
    ann_ret = float((1 + r).prod() ** (ppa / n) - 1)
    ann_vol = float(r.std() * np.sqrt(ppa))
    rf      = risk_free_annual / ppa
    exc     = r - rf
    sharpe  = float(exc.mean() / (exc.std() + 1e-8) * np.sqrt(ppa))
    dn      = exc[exc < 0]
    dv      = float(dn.std() * np.sqrt(ppa)) if len(dn) > 1 else 1e-8
    sortino = float(exc.mean() / (dv + 1e-8) * np.sqrt(ppa))
    rm      = np.maximum.accumulate(eq)
    dd      = (eq - rm) / (rm + 1e-8)
    mdd     = float(dd.min())
    calmar  = ann_ret / abs(mdd + 1e-8)
    wins    = r[r > 0]
    losses  = r[r < 0]
    return {
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Sortino Ratio":   sortino,
        "Max Drawdown":    mdd,
        "Calmar Ratio":    calmar,
        "Hit Rate":        float((r > 0).mean()),
        "Avg Win":         float(wins.mean())   if len(wins)   > 0 else 0.0,
        "Avg Loss":        float(losses.mean()) if len(losses) > 0 else 0.0,
        "Profit Factor":   float(wins.sum() / abs(losses.sum() + 1e-8))
                           if len(losses) > 0 else np.inf,
        "Num Periods":     n,
    }


def buy_and_hold_pnl(
    rets: pd.DataFrame,
    signal_dates: List,
    horizon: int,
) -> pd.Series:
    ra  = rets.values
    rdn = _normalize_tz(rets.index)
    T   = len(ra)
    N   = ra.shape[1]
    dp  = np.zeros(T)
    no  = np.zeros(T, dtype=np.int32)
    for sd in signal_dates:
        sn = _normalize_ts(sd)
        if sn not in rdn:
            continue
        ti = rdn.get_loc(sn)
        if ti + horizon >= T:
            continue
        wbh = np.ones(N) / (N * horizon)
        for d in range(1, horizon + 1):
            idx = ti + d
            if idx < T:
                dp[idx] += float(np.dot(wbh, ra[idx]))
                no[idx] += 1
    ai = np.where(no > 0)[0]
    return (pd.Series(dp[ai], index=rets.index[ai])
            if len(ai) > 0 else pd.Series(dtype=float))

# ─────────────────────────────────────────────────────────────────────────────
# 14)  REPORTING  (updated labels)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLOURS = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2"]
REGIME_NAMES   = {
    2: ["Bull", "Bear"],
    3: ["Bull/Low-Vol", "Bear/Stress", "Transition"],
    4: ["Strong Bull", "Bear", "High-Vol", "Sideways"],
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
})


def _rlabels(K: int) -> List[str]:
    return REGIME_NAMES.get(K, [f"Regime {k}" for k in range(K)])


def print_metrics_table(sm: Dict, bm: Dict, name: str) -> None:
    W = 64
    print(f"\n{'─'*W}\n  Performance Metrics — {name.upper()}\n{'─'*W}")
    print(f"  {'Metric':<26}  {'RARSI-BlumeCapel':>15}  {'B&H':>15}\n{'─'*W}")
    fmts = {
        "Ann. Return":    ".2%",  "Ann. Volatility": ".2%",
        "Sharpe Ratio":   ".3f",  "Sortino Ratio":   ".3f",
        "Max Drawdown":   ".2%",  "Calmar Ratio":    ".3f",
        "Hit Rate":       ".2%",  "Avg Win":         ".3%",
        "Avg Loss":       ".3%",  "Profit Factor":   ".3f",
        "Num Periods":    "d",
    }
    for k, fmt in fmts.items():
        sv = sm.get(k, float("nan"))
        bv = bm.get(k, float("nan"))
        try:
            ss = format(sv, fmt); bs = format(bv, fmt)
        except Exception:
            ss = str(sv);          bs = str(bv)
        print(f"  {k:<26}  {ss:>15}  {bs:>15}")
    print(f"{'─'*W}\n")


def print_accuracy_table(acc: Dict, name: str, K: int = 3) -> None:
    labels = _rlabels(K)
    print(f"{'─'*58}\n  Accuracy — {name.upper()}\n{'─'*58}")

    def row(lbl, val, pct=True):
        s = (
            (f"{val:.2%}" if pct else f"{val:.4f}")
            if (isinstance(val, float) and not np.isnan(val)) else "n/a"
        )
        print(f"  {lbl:<44}  {s:>10}")

    row("Overall Directional Accuracy", acc.get("Overall Accuracy", float("nan")))
    row("Long-side Accuracy",           acc.get("Long Accuracy",    float("nan")))
    row("Short-side Accuracy",          acc.get("Short Accuracy",   float("nan")))
    print()
    row("Top-Tertile Signal Accuracy",  acc.get("Top-Tertile Signal Accuracy", float("nan")))
    row("Bot-Tertile Signal Accuracy",  acc.get("Bot-Tertile Signal Accuracy", float("nan")))
    print()
    row("Blended IC",       acc.get("Blended IC (Spearman)", float("nan")), pct=False)
    row("Blended IC-IR",    acc.get("Blended IC-IR",         float("nan")), pct=False)
    row("BlumeCapel IC",    acc.get("BlumeCapel IC",         float("nan")), pct=False)
    row("Momentum IC",      acc.get("Momentum IC",           float("nan")), pct=False)
    print(f"  Spin encoding: {acc.get('Spin Encoding', 'quantile')}")
    print()
    for r, v in acc.get("Per-Regime Accuracy", {}).items():
        lbl = labels[r] if r < len(labels) else f"R{r}"
        row(f"  Regime {r} — {lbl}", v)
    print()
    if "Gated signals (%)" in acc:
        row("Signals gated (%)", acc["Gated signals (%)"])
        for r in range(K):
            k = f"Gate acc R{r}"
            if k in acc:
                row(f"  Gate acc R{r}", acc[k])
    print(
        f"\n  Active: {acc.get('Total Active Predictions', 0):,} | "
        f"Correct: {acc.get('Total Correct', 0):,}\n{'─'*58}\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 15)  PLOTS  (updated labels)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_show_save(fig, path: str, show: bool) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_universe_dashboard(result: Dict, name: str, show: bool = True) -> None:
    trainer   = result["trainer"]
    portfolio = result["portfolio"]
    rets      = result["rets"]
    acc       = result["accuracy"]
    bh_pnl    = result["bh_pnl"]
    horizon   = trainer.horizon
    K         = trainer.K

    eq    = portfolio.equity_curve()
    bh_eq = 100.0 * (1 + bh_pnl).cumprod()
    labels  = _rlabels(K)
    colours = REGIME_COLOURS[:K]

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(
        f"Universe Dashboard (Blume–Capel) — {name.upper()}  |  "
        f"W={trainer.train_window}d  H={horizon}d  K={K}  "
        f"enc={trainer.spin_encoding}  q={trainer.spin_quantile:.2f}  "
        f"τ={trainer.emission_temperature:.1f}  "
        f"avg_n_pca={np.mean(trainer.n_pca_history):.0f}  "
        f"avg_EV={np.mean(trainer.ev_history):.0%}",
        fontsize=11, fontweight="bold", y=0.99,
    )
    gs = fig.add_gridspec(5, 2, hspace=0.55, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values, color="#1565C0", lw=1.8, label="RARSI-BlumeCapel")
    ax1.plot(bh_eq.index, bh_eq.values, color="#B71C1C",
             lw=1.4, ls="--", alpha=0.8, label="Equal-weight B&H")
    sm = result["metrics"]["strategy"]
    bm = result["metrics"]["benchmark"]
    ax1.set_title(
        f"NAV  |  Sharpe: RARSI={sm.get('Sharpe Ratio',0):.3f}  "
        f"B&H={bm.get('Sharpe Ratio',0):.3f}  |  "
        f"MaxDD: {sm.get('Max Drawdown',0):.1%}",
        fontsize=9,
    )
    ax1.set_ylabel("NAV (base=100)")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, :])
    ea = eq.values
    rm = np.maximum.accumulate(ea)
    dd = (ea - rm) / (rm + 1e-8) * 100
    ax2.fill_between(eq.index, dd, 0, color="#EF5350", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Strategy Drawdown")

    ax3 = fig.add_subplot(gs[2, 0])
    rdn = _normalize_tz(rets.index)
    T   = len(rets)
    ra  = rets.values
    ic_p_l, ic_m_l, ic_b_l, ic_dates = [], [], [], []
    for sp, sm2, sb, sd in zip(
        trainer.potts_signals, trainer.mom_signals,
        trainer.signals, trainer.signal_dates,
    ):
        sn = _normalize_ts(sd)
        if sn not in rdn:
            continue
        ti = rdn.get_loc(sn)
        if ti + horizon >= T:
            continue
        fut  = ra[ti + 1: ti + 1 + horizon]
        real = np.prod(1 + fut, axis=0) - 1
        if np.abs(sb).sum() < 1e-8 or np.abs(real).sum() < 1e-8:
            continue
        rp, _ = spearmanr(sp, real)
        rm2, _ = spearmanr(sm2, real)
        rb, _ = spearmanr(sb, real)
        if any(np.isnan([rp, rm2, rb])):
            continue
        ic_p_l.append(rp); ic_m_l.append(rm2)
        ic_b_l.append(rb); ic_dates.append(sd)
    if ic_dates:
        idx  = pd.Index(ic_dates)
        roll = 40
        for arr_, c, lbl in [
            (ic_p_l, "#B71C1C", "BlumeCapel"),
            (ic_m_l, "#1565C0", "Momentum"),
            (ic_b_l, "#2E7D32", "Blended"),
        ]:
            s = pd.Series(arr_, index=idx).rolling(roll, min_periods=5).mean()
            ax3.plot(s, lw=1.5, color=c, label=f"{lbl} (μ={np.mean(arr_):.4f})")
    ax3.axhline(0, color="grey", lw=0.8, ls=":")
    ax3.set_title("Rolling 40-step IC by component")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[2, 1])
    if trainer.ising_weights:
        dates_sig = list(trainer.signal_dates)
        ax4.fill_between(dates_sig, trainer.ising_weights,
                         alpha=0.6, color="#B71C1C", label="BlumeCapel weight")
        ax4.fill_between(dates_sig, [1-w for w in trainer.ising_weights],
                         alpha=0.6, color="#1565C0", label="Momentum weight")
    ax4.set_ylim(0, 1)
    ax4.set_title("Adaptive blend weights over time")
    ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(gs[3, 0])
    per_r = acc.get("Per-Regime Accuracy", {})
    if per_r:
        rlbls = [labels[r] if r < len(labels) else f"R{r}" for r in sorted(per_r)]
        rvals = [per_r[r] for r in sorted(per_r)]
        bars  = ax5.bar(rlbls, rvals,
                        color=[colours[r] for r in sorted(per_r)], alpha=0.8)
        ax5.axhline(0.5, color="grey", lw=1.0, ls=":", label="Random (50%)")
        ax5.axhline(acc.get("Overall Accuracy", 0.5), color="black",
                    lw=1.2, ls="--",
                    label=f"Overall {acc.get('Overall Accuracy',0):.1%}")
        for bar, v in zip(bars, rvals):
            ax5.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                     f"{v:.1%}", ha="center", fontsize=8)
        ax5.set_ylim(0.3, 0.75)
        ax5.set_title("Prediction Accuracy by Regime")
        ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(gs[3, 1])
    regimes = trainer.regime_labels
    dates_r = list(trainer.signal_dates)
    if regimes:
        prev_r = regimes[0]; seg_start = 0
        for i in range(1, len(regimes)):
            if regimes[i] != prev_r:
                ax6.axvspan(dates_r[seg_start], dates_r[i],
                            color=colours[prev_r], alpha=0.4)
                seg_start = i; prev_r = regimes[i]
        ax6.axvspan(dates_r[seg_start], dates_r[-1],
                    color=colours[prev_r], alpha=0.4)
        for k in range(K):
            ax6.bar(0, 0, color=colours[k], alpha=0.7, label=labels[k])
        if trainer.gate_scales:
            ax6.plot(dates_r, np.array(trainer.gate_scales),
                     color="black", lw=0.9, label="Gate scale")
    ax6.set_ylim(0, 1.1)
    ax6.set_title("Regime Map + Gate Scale")
    ax6.legend(fontsize=6, ncol=2)

    ax7 = fig.add_subplot(gs[4, :])
    rdn2 = _normalize_tz(rets.index)
    pct_neutral, pct_dates = [], []
    for sp, sd in zip(trainer.potts_signals, trainer.signal_dates):
        sn = _normalize_ts(sd)
        if sn not in rdn2:
            continue
        frac_neutral = float((np.abs(sp) < trainer.entry_threshold).mean())
        pct_neutral.append(frac_neutral)
        pct_dates.append(sd)
    if pct_dates:
        roll_n = pd.Series(pct_neutral, index=pd.Index(pct_dates)).rolling(40, min_periods=5).mean()
        ax7.fill_between(roll_n.index, roll_n.values, alpha=0.5, color="#388E3C")
        ax7.set_ylabel("Fraction near-neutral")
        ax7.set_title(
            "Rolling fraction of assets in near-neutral state "
            "(Blume–Capel crystal-field diagnostic)"
        )
        ax7.set_ylim(0, 1)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.tick_params(labelsize=7)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

    path = os.path.join("results", f"{name}_blumecapel_dashboard.png")
    _safe_show_save(fig, path, show)


def plot_suite_comparison(results: Dict[str, Dict], show: bool = True) -> None:
    names      = list(results.keys())
    sharpes    = [results[n]["metrics"]["strategy"].get("Sharpe Ratio",   float("nan")) for n in names]
    returns    = [results[n]["metrics"]["strategy"].get("Ann. Return",    float("nan")) for n in names]
    maxdds     = [results[n]["metrics"]["strategy"].get("Max Drawdown",   float("nan")) for n in names]
    accs       = [results[n]["accuracy"].get("Overall Accuracy",          float("nan")) for n in names]
    bh_sharpes = [results[n]["metrics"]["benchmark"].get("Sharpe Ratio",  float("nan")) for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Suite Comparison — Blume–Capel RARSI vs B&H", fontsize=13, fontweight="bold")

    def _hbar(ax, vals, bvals, title, xlabel, fmt, ref=None):
        y    = np.arange(len(names))
        bars = ax.barh(y, vals, alpha=0.8, color="#1565C0", label="RARSI-BlumeCapel")
        if bvals is not None:
            ax.barh(y, bvals, alpha=0.5, color="#B71C1C", label="B&H")
        if ref is not None:
            ax.axvline(ref, color="grey", lw=0.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("sector_", "S/") for n in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        format(v, fmt), va="center", fontsize=7)
        ax.legend(fontsize=7)

    _hbar(axes[0,0], sharpes,   bh_sharpes, "Sharpe Ratio",      "Sharpe", ".3f", ref=0)
    _hbar(axes[0,1], returns,   None,        "Annualised Return", "Return", ".1%", ref=0)
    _hbar(axes[1,0], maxdds,    None,        "Max Drawdown",      "DD",     ".1%")
    _hbar(axes[1,1], accs,      None,        "Directional Accuracy","Acc",  ".1%", ref=0.5)

    plt.tight_layout()
    path = os.path.join("results", "suite_blumecapel_comparison.png")
    _safe_show_save(fig, path, show)

# ─────────────────────────────────────────────────────────────────────────────
# 16)  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset_name,
    train_window: int      = 126,
    horizon: int           = 5,
    refit_freq: int        = 5,
    K: int                 = 3,
    n_passes: int          = 15,
    eta: float             = 0.01,
    eta_A: float           = 0.05,
    h_max: float           = 2.0,
    J_max: float           = 1.0,
    d_max: float           = 2.0,
    l2: float              = 1e-3,
    l1_J: float            = 0.05,
    l2_d: float            = 1e-3,
    mf_iter: int           = 25,
    n_pca: Union[int, str] = "auto",
    pca_variance_target: float = 0.75,
    spin_quantile: float   = 0.33,
    spin_encoding: str     = "quantile",      # FIX 2.1
    n_warmup_passes: int   = 2,               # FIX 2.4
    center_d: bool         = True,            # FIX 1.2
    emission_temperature: float = 1.0,        # FIX 2.3
    ic_window: int         = 60,
    base_momentum_w: float = 1.0,
    mom_windows: Tuple     = (20, 60, 120),
    use_regime_gate: bool      = False,
    min_regime_accuracy: float = 0.51,
    gate_scale: float      = 0.35,
    gate_warmup: int       = 40,
    entry_threshold: float = 0.20,
    max_position: float    = 0.25,
    min_gross_signal: float = 0.05,
    short_scale: float     = 0.0,
    cost_bps: float        = 0.0,
    weight_mode: str       = "long_only",
    vol_target: Optional[float] = 0.15,
    max_leverage: float    = 2.0,
    regime_confidence_scaling: bool = False,
    min_conf_scale: float  = 0.35,
    start: str             = "2015-01-01",
    end: str               = "2025-01-01",
    verbose: bool          = True,
    show_plots: bool       = True,
) -> Dict:

    print(f"\n{'═'*65}")
    print(
        f"  {dataset_name}  W={train_window}d H={horizon}d K={K} "
        f"q={spin_quantile:.2f} enc={spin_encoding} τ={emission_temperature:.1f} "
        f"gate={use_regime_gate}"
    )
    print(f"{'═'*65}")

    _, rets, _, tickers = load_dataset(dataset_name, start=start, end=end)
    print(f"  {rets.shape[0]} × {rets.shape[1]}")

    trainer = RollingWindowTrainer(
        train_window=train_window, horizon=horizon, K=K, refit_freq=refit_freq,
        eta=eta, eta_A=eta_A, h_max=h_max, J_max=J_max, d_max=d_max,
        l2=l2, l1_J=l1_J, l2_d=l2_d, mf_iter=mf_iter, n_passes=n_passes,
        n_pca=n_pca, pca_variance_target=pca_variance_target,
        spin_quantile=spin_quantile, spin_encoding=spin_encoding,
        n_warmup_passes=n_warmup_passes, center_d=center_d,
        emission_temperature=emission_temperature,
        ic_window=ic_window, base_momentum_w=base_momentum_w,
        mom_windows=mom_windows,
        use_regime_gate=use_regime_gate, min_regime_accuracy=min_regime_accuracy,
        gate_scale=gate_scale, gate_warmup=gate_warmup,
        entry_threshold=entry_threshold,
    )
    trainer.run(rets, verbose=verbose)

    portfolio = PortfolioEngine(
        entry_threshold=entry_threshold, max_position=max_position,
        cost_bps=cost_bps, weight_mode=weight_mode,
        min_gross_signal=min_gross_signal, short_scale=short_scale,
        vol_target=vol_target, max_leverage=max_leverage,
        regime_confidence_scaling=regime_confidence_scaling,
        min_conf_scale=min_conf_scale,
    )
    portfolio.simulate(trainer, rets)
    bh_pnl = buy_and_hold_pnl(rets, trainer.signal_dates, horizon)

    sm = compute_metrics(portfolio.pnl_series)
    bm = compute_metrics(bh_pnl)
    print_metrics_table(sm, bm, str(dataset_name))

    acc = compute_prediction_accuracy(trainer, rets, entry_threshold)
    print_accuracy_table(acc, str(dataset_name), K=K)

    result = {
        "trainer":   trainer,
        "portfolio": portfolio,
        "rets":      rets,
        "tickers":   tickers,
        "metrics":   {"strategy": sm, "benchmark": bm},
        "accuracy":  acc,
        "bh_pnl":    bh_pnl,
    }
    if show_plots:
        plot_universe_dashboard(result, str(dataset_name), show=True)
    return result


def run_dataset_suite(
    dataset_names: Sequence[str],
    start="2015-01-01", end="2025-01-01",
    verbose=True, show_plots=True, **kwargs,
) -> Dict[str, Dict]:
    results = {}
    for name in dataset_names:
        print(f"\n\n{'#'*10} {name.upper()} {'#'*10}")
        results[name] = run_experiment(
            name, start=start, end=end,
            verbose=verbose, show_plots=show_plots, **kwargs,
        )
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 17)  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite = [
        "nifty50", 
        # "nifty100",
        # "sector_it", "sector_banking", "sector_pharma", "sector_energy",
        # "sector_auto", "sector_fmcg", "sector_metals",
        # "large_cap", "mid_cap", "small_cap",
    ]

    results = run_dataset_suite(
        suite,
        # ── Trainer ──────────────────────────────────────────────
        train_window    = 126,
        horizon         = 5,
        K               = 3,
        refit_freq      = 5,
        n_passes        = 15,
        mf_iter         = 25,
        # ── PCA + spin encoding ───────────────────────────────────
        n_pca                = "auto",
        pca_variance_target  = 0.75,
        spin_quantile        = 0.33,
        spin_encoding        = "zscore",  # change to "zscore" for FIX 2.1
        # ── Blume–Capel model ─────────────────────────────────────
        eta    = 0.01,
        eta_A  = 0.05,
        h_max  = 2.0,
        J_max  = 1.0,
        d_max  = 2.0,
        l2     = 1e-3,
        l1_J   = 0.05,
        l2_d   = 1e-3,
        center_d           = True,   # FIX 1.2: d mean-zero constraint
        n_warmup_passes    = 2,      # FIX 2.4: convergence warmup
        emission_temperature = 1.0,  # FIX 2.3: set > 1 near criticality
        # ── Blender ───────────────────────────────────────────────
        ic_window       = 60,
        base_momentum_w = 1.0,
        mom_windows     = (20, 60, 120),
        # ── Regime gate (off by default) ──────────────────────────
        use_regime_gate     = False,
        min_regime_accuracy = 0.51,
        gate_scale          = 0.35,
        gate_warmup         = 40,
        # ── Portfolio ─────────────────────────────────────────────
        entry_threshold  = 0.20,
        max_position     = 0.25,
        min_gross_signal = 0.05,
        short_scale      = 0.0,
        cost_bps         = 0.0,
        weight_mode      = "long_only",
        vol_target       = 0.15,
        max_leverage     = 2.0,
        regime_confidence_scaling = False,
        min_conf_scale   = 0.35,
        # ── Date range ────────────────────────────────────────────
        start = "2015-01-01",
        end   = "2025-01-01",
        verbose    = True,
        show_plots = True,
    )

    rows = []
    for name, res in results.items():
        s = res["metrics"]["strategy"]
        b = res["metrics"]["benchmark"]
        a = res["accuracy"]
        rows.append({
            "Universe":     name,
            "Sharpe":       s.get("Sharpe Ratio",          float("nan")),
            "Ann.Ret":      s.get("Ann. Return",            float("nan")),
            "MaxDD":        s.get("Max Drawdown",           float("nan")),
            "HitRate":      s.get("Hit Rate",               float("nan")),
            "BH.Sharpe":    b.get("Sharpe Ratio",           float("nan")),
            "Accuracy":     a.get("Overall Accuracy",       float("nan")),
            "LongAcc":      a.get("Long Accuracy",          float("nan")),
            "BlendedIC":    a.get("Blended IC (Spearman)",  float("nan")),
            "BlumeCapelIC": a.get("BlumeCapel IC",          float("nan")),
            "MomIC":        a.get("Momentum IC",            float("nan")),
            "Gated%":       a.get("Gated signals (%)",      float("nan")),
            "SpinEnc":      a.get("Spin Encoding",          "quantile"),
            "Assets":       len(res["tickers"]),
        })

    summary      = pd.DataFrame(rows)
    summary_path = os.path.join("results", "blumecapel_suite_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nSuite summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved → {summary_path}")

    plot_suite_comparison(results, show=True)
