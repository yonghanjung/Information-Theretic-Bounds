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

[Paper](https://arxiv.org/abs/2601.17160) | [GitHub](https://github.com/yonghanjung/Information-Theretic-Bounds) | [PyPI](https://pypi.org/project/itbound/)

This Space demonstrates the core idea of [Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding](https://arxiv.org/abs/2601.17160): when strong identification assumptions are not credible, causal intervals can still be valid and informative.

## What this demo does

- Uses the canonical IHDP example when no file is uploaded
- Uploads an observational CSV when you want to override the default example
- Selects treatment, outcome, and covariates
- Computes lower and upper causal bounds
- Shows a ribbon plot against `x0` for the canonical IHDP example, with ground-truth effect and bounds
- Returns a width histogram and claims summary
- Produces a downloadable artifact bundle

## Suggested first run

- treatment column: `treatment`
- outcome column: `y_factual`
- covariates: `x1,x2,x3,x4,x5`
- ribbon x-axis column: `x0`
- divergences: `KL`, `TV`
- aggregation mode: `paper_adaptive_k`

## Links

- Paper: <https://arxiv.org/abs/2601.17160>
- Package: <https://pypi.org/project/itbound/>
- Code: <https://github.com/yonghanjung/Information-Theretic-Bounds>
