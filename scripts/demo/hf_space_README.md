---
title: itbound Demo
emoji: 📉
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
sdk_version: "5.50.0"
app_file: app.py
pinned: true
tags:
- causal-inference
- causal-bounds
- unmeasured-confounding
- treatment-effects
- statistics
---

# itbound Demo

Upload a CSV and compute data-driven lower and upper causal bounds under unmeasured confounding.

This Space demonstrates the core idea of [Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding](https://arxiv.org/abs/2601.17160): when strong identification assumptions are not credible, causal intervals can still be valid and informative.

## What this demo does

- Uploads an observational CSV
- Selects treatment, outcome, and covariates
- Computes lower and upper causal bounds
- Returns a width histogram and claims summary
- Produces a downloadable artifact bundle

## Suggested first run

- treatment column: `a`
- outcome column: `y`
- covariates: `x1,x2`
- divergences: `KL`, `TV`
- aggregation mode: `paper_adaptive_k`

## Links

- Paper: <https://arxiv.org/abs/2601.17160>
- Package: <https://pypi.org/project/itbound/>
- Code: <https://github.com/yonghanjung/Information-Theretic-Bounds>
