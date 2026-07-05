# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Experiments testing whether migration topology determines solution diversity in multi-agent LLM systems, extending the "From Games to Graphs" paper. Key finding: **the ordering inverts** compared to genetic algorithms.

- GAs: none > ring > star > FC for diversity
- LLMs: FC > star ≈ ring > none (Kendall's W = 0.91 across 6 clean domains)

The sign-flip is explained by contravariant functoriality: GA migration is a left Kan extension (colimit-like, homogenizes); LLM "be different" context is a right Kan extension (limit-like, diversifies).

## Stack

- Python 3, OpenAI API (GPT-4o-mini)
- Sentence embeddings (all-MiniLM-L6-v2) for cosine distance metrics
- Haskell for formal theory

## Architecture

- `experiment.py` — 8 coding tasks × 4 topologies × 10 seeds = 1,600 API calls
- `THEORY.md` — Formal theory: Kan extensions, adjoint functors, landscape preconditions
- `AgentOrchestration.hs` — Haskell formalization
- `letter_to_lyra_claudius.md` — Experimental narrative and intuitive explanation
- `output*/` — CSV + JSON results

## Preconditions

Effect requires (1) peaked prior (low entropy default) and (2) multi-modal solution space. Fails gracefully on unimodal tasks or already-diverse priors.
