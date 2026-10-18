from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeFit:
    means: np.ndarray
    sigmas: np.ndarray
    transition: np.ndarray
    filtered: np.ndarray
    smoothed: np.ndarray
    log_likelihood: float
    iterations: int
    volatility_only: bool


def _density(x: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    z = (x[:, None] - means[None, :]) / sigmas[None, :]
    return np.exp(-0.5 * z**2) / (sigmas[None, :] * math.sqrt(2 * math.pi)) + 1e-300


def stationary(transition: np.ndarray) -> np.ndarray:
    a = np.array([[1 - transition[0, 0], -transition[1, 0]], [1.0, 1.0]])
    return np.linalg.solve(a, np.array([0.0, 1.0]))


def forward_backward(
    x: np.ndarray, means: np.ndarray, sigmas: np.ndarray, transition: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(x)
    density = _density(x, means, sigmas)
    d0, d1 = density[:, 0].tolist(), density[:, 1].tolist()
    p00, p01, p10, p11 = (float(v) for v in transition.ravel())
    start = stationary(transition)
    f0 = [0.0] * n
    f1 = [0.0] * n
    q0 = [0.0] * n
    q1 = [0.0] * n
    log_likelihood = 0.0
    q0[0], q1[0] = float(start[0]), float(start[1])
    for t in range(n):
        if t:
            q0[t] = f0[t - 1] * p00 + f1[t - 1] * p10
            q1[t] = f0[t - 1] * p01 + f1[t - 1] * p11
        j0, j1 = q0[t] * d0[t], q1[t] * d1[t]
        total = j0 + j1
        f0[t], f1[t] = j0 / total, j1 / total
        log_likelihood += math.log(total)
    s0 = [0.0] * n
    s1 = [0.0] * n
    s0[-1], s1[-1] = f0[-1], f1[-1]
    c00 = c01 = c10 = c11 = 0.0
    for t in range(n - 2, -1, -1):
        r0 = s0[t + 1] / max(q0[t + 1], 1e-300)
        r1 = s1[t + 1] / max(q1[t + 1], 1e-300)
        w00, w01 = f0[t] * p00 * r0, f0[t] * p01 * r1
        w10, w11 = f1[t] * p10 * r0, f1[t] * p11 * r1
        s0[t], s1[t] = w00 + w01, w10 + w11
        c00 += w00
        c01 += w01
        c10 += w10
        c11 += w11
    filtered = np.column_stack([f0, f1])
    smoothed = np.column_stack([s0, s1])
    counts = np.array([[c00, c01], [c10, c11]])
    return filtered, smoothed, counts, float(log_likelihood)


def _start(x: np.ndarray, rng: np.random.Generator, volatility_only: bool):
    spread = x.std()
    sigmas = np.array([0.6, 1.6]) * spread * rng.uniform(0.7, 1.3, 2)
    means = np.zeros(2) if volatility_only else np.quantile(x, [0.3, 0.7])
    stay = rng.uniform(0.9, 0.99, 2)
    transition = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    return means, sigmas, transition


def fit_markov(
    x: np.ndarray,
    volatility_only: bool = False,
    starts: int = 6,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
    seed: int = 0,
) -> RegimeFit:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    best: RegimeFit | None = None
    floor = x.std() * 0.02
    for _ in range(starts):
        means, sigmas, transition = _start(x, rng, volatility_only)
        previous = -np.inf
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            filtered, smoothed, counts, loglik = forward_backward(x, means, sigmas, transition)
            weight = smoothed.sum(axis=0)
            if not volatility_only:
                means = (smoothed * x[:, None]).sum(axis=0) / weight
            variance = (smoothed * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / weight
            sigmas = np.sqrt(np.maximum(variance, floor**2))
            transition = counts / counts.sum(axis=1, keepdims=True)
            if abs(loglik - previous) < tolerance * max(1.0, abs(loglik)):
                break
            previous = loglik
        filtered, smoothed, _, loglik = forward_backward(x, means, sigmas, transition)
        order = np.argsort(sigmas) if volatility_only else np.argsort(means)
        candidate = RegimeFit(
            means=means[order],
            sigmas=sigmas[order],
            transition=transition[np.ix_(order, order)],
            filtered=filtered[:, order],
            smoothed=smoothed[:, order],
            log_likelihood=loglik,
            iterations=iteration,
            volatility_only=volatility_only,
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate
    return best


def describe_regimes(
    fit: RegimeFit, min_vol_ratio: float = 1.3, min_probability: float = 0.6
) -> dict:
    ratio = float(fit.sigmas[1] / fit.sigmas[0])
    durations = 1.0 / np.maximum(1.0 - np.diag(fit.transition), 1e-9)
    current = fit.filtered[-1]
    state = int(np.argmax(current))
    separated = (
        ratio >= min_vol_ratio
        if fit.volatility_only
        else (
            abs(fit.means[1] - fit.means[0]) > 0.5 * float(fit.sigmas.mean())
            or ratio >= min_vol_ratio
        )
    )
    labels = ("calm", "turbulent") if fit.volatility_only else ("low", "high")
    confident = bool(separated and current[state] >= min_probability)
    return {
        "state": labels[state] if separated else "undetermined",
        "probability": float(current[state]),
        "probability_high": float(current[1]),
        "confident": confident,
        "separated": bool(separated),
        "vol_ratio": ratio,
        "means": fit.means.tolist(),
        "sigmas": fit.sigmas.tolist(),
        "expected_duration": durations.tolist(),
        "persistence": np.diag(fit.transition).tolist(),
        "stationary": stationary(fit.transition).tolist(),
        "log_likelihood": fit.log_likelihood,
    }


def classify(returns: np.ndarray, volatility_only: bool = True, **options) -> dict:
    fit = fit_markov(returns, volatility_only=volatility_only, **options)
    summary = describe_regimes(fit)
    summary["smoothed_high"] = fit.smoothed[:, 1]
    return summary
