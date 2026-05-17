# Markets in a Spin / RARSI v4.1

A regime-aware, cross-sectional equity return model that combines adaptive PCA, three-state spin encoding, a Blume–Capel interaction model, a sticky Hidden Markov Model, online parameter learning, and risk-managed portfolio construction.

## Overview

This project models financial markets as interacting asset configurations. The core idea is to:
1. compress cross-sectional returns into a low-dimensional latent space,
2. convert latent scores into discrete spin states `{-1, 0, +1}`,
3. learn regime-specific interaction structure with a Blume–Capel model,
4. infer latent market regimes with an HMM,
5. blend model signals with momentum, and
6. map forecasts into portfolio weights with volatility and transaction-cost controls.

## Why this model?

The binary up/down market view can be too coarse when returns cluster near zero. Introducing a neutral state gives a better representation of weak movements and naturally leads to a three-state spin system. The model also separates market behaviour into latent regimes such as bullish, bearish, and transition states.  

## Main components

### 1. Adaptive PCA preprocessing
The model first applies PCA to a rolling training window of cross-sectional returns. This reduces dimensionality and produces latent principal component scores.

### 2. Three-state spin encoding
Each principal component score is mapped to:
- `-1` for sufficiently negative values,
- `0` for neutral / low-activity values,
- `+1` for sufficiently positive values.

### 3. Blume–Capel interaction model
The latent spin vector is modeled with a 3-state Blume–Capel Hamiltonian:
- linear field `h`
- crystal field `d`
- symmetric coupling matrix `J`

Mean-field inference is used to keep the model tractable.

### 4. Regime-switching HMM
A Hidden Markov Model captures slow regime changes over time. Each regime has its own Blume–Capel parameters, and a sticky transition prior encourages persistence.

### 5. Online learning
Parameters are updated with weighted pseudo-likelihood objectives and AdaGrad-style online mirror descent. The transition matrix is updated with exponentiated gradient descent.

### 6. Signal generation and blending
The model produces:
- a Blume–Capel forecast from regime magnetisations,
- a momentum signal from multiple lookback windows,

and blends them using rolling information coefficients.

### 7. Portfolio construction
Final signals are converted into positions with:
- entry thresholds,
- long-only / signal / equal-weight / volatility-scaled modes,
- gross exposure normalization,
- per-asset caps,
- volatility targeting,
- optional regime-confidence scaling,
- transaction-cost adjustment.

## Workflow

1. Load and clean adjusted price data.
2. Compute cross-sectional returns.
3. Fit PCA on the rolling training window.
4. Encode latent scores into three-state spins.
5. Fit regime-specific Blume–Capel models.
6. Infer regimes with a log-space HMM filter.
7. Update parameters online.
8. Generate blended forecasts.
9. Convert forecasts into portfolio weights.
10. Evaluate out-of-sample under walk-forward validation.

## Walk-forward validation

The protocol is fully walk-forward:
- no peeking into the future,
- PCA and thresholds are fit only on past data,
- parameters are updated online between refits,
- evaluation is always performed on future returns.

## Performance metrics

The project tracks:
- annualised return,
- annualised volatility,
- Sharpe ratio,
- Sortino ratio,
- maximum drawdown,
- Calmar ratio,
- directional accuracy,
- information coefficient and IC-IR.

## Default control parameters

Common control parameters include:
- `H`: forecast horizon and holding period
- `F`: refit frequency
- `W`: training window length

## Numerical stability features

The implementation uses several stability safeguards:
- log-sum-exp for probability normalisation,
- clipped emissions,
- gradient clipping,
- Frobenius-norm projection for `J`,
- belief flooring in the HMM,
- centered `d` updates,
- regularisation on `h`, `d`, and `J`.

## Complexity notes

The heaviest steps are PCA/SVD, mean-field iterations, PLL gradients, and HMM updates. The model is designed to remain numerically stable while still being practical for online walk-forward use.

## Project structure

Suggested structure:
- `data/` — market data and preprocessing
- `models/` — PCA, spin encoding, Blume–Capel, HMM, signal blending
- `strategies/` — portfolio construction and risk layers
- `evaluation/` — metrics and backtest routines
- `notebooks/` — experiments and analysis
- `docs/` — technical notes and presentations

## Getting started

This repository should include:
1. data ingestion and cleaning,
2. a walk-forward training script,
3. signal generation,
4. a backtest / evaluation pipeline,
5. configuration for model hyperparameters.

## Notes

This README is intentionally project-facing and concise. The technical details of the model are documented in the supporting write-up, while the presentation supplies the intuitive market-to-spin framing and evaluation storyline.

## References

- Technical documentation for **RARSI v4.1**
- Presentation: **Markets in a Spin**
