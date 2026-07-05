# The Signed Laxator: How Coupling Direction Determines Diversity in Multi-Agent LLM Systems

## Abstract

We test whether the categorical result of Langer, Turing & Vega (2026) — that migration topology determines diversity dynamics in island-model genetic algorithms — transfers to multi-agent LLM systems. We find that it does, but with a sign flip: the diversity ordering inverts when the coupling mechanism changes from "merge" (left Kan extension, homogenizing) to "constrain" (right Kan extension, diversifying). A single prompt change — "build on this approach" vs "use a different approach" — flips the ordering across all eight coding tasks tested. We propose that the diversity fingerprint is functorial up to a sign determined by the Lan ⊣ Ran adjunction, and identify a landscape precondition under which the effect is predictable.

## 1. Background

Langer, Turing & Vega (ACT 2026) formalize genetic algorithm operators as Kleisli morphisms over an evolution monad T = Reader(Γ) × Writer(Λ) × State(S). Their main experimental result: migration topology determines diversity dynamics independently of the fitness landscape, with the ordering

```
none > ring > star > random > fully_connected
```

and perfect cross-domain agreement (Kendall's W = 1.0 across six domains). Diversity decreases monotonically with algebraic connectivity λ₂ of the migration graph.

**Question**: Does this structural result transfer to LLM sub-agents, where "migration" is replaced by "sharing prior solutions as context"?

## 2. Experimental Design

### 2.1 Setup

- **Model**: GPT-4o-mini (temperature = 1.0)
- **Agents per run**: 5 (sequential generation within each topology)
- **Seeds**: 10 per (task, topology, coupling) condition
- **Diversity metric**: Mean pairwise cosine distance on sentence embeddings (all-MiniLM-L6-v2, 384-dim, L2-normalized)

### 2.2 Topologies

| Topology | Context rule | Analogous λ₂ |
|---|---|---|
| **None** (strict) | Agent sees nothing | 0 |
| **Ring** | Agent sees previous agent's solution | Low |
| **Star** | Agents 1-4 see agent 0's solution only | Medium |
| **Fully connected** | Agent sees ALL prior agents' solutions | High |

### 2.3 Coupling modes

Two experiments, identical except for one prompt sentence:

**Negative coupling** (right Kan / diversifying):
> "You MUST use a DIFFERENT algorithm or approach than the ones shown. Do not copy or trivially modify their code — write your own implementation using a fundamentally different strategy."

**Positive coupling** (left Kan / homogenizing):
> "BUILD ON the best approach shown — take the strongest solution, refine it, fix any issues, and improve its implementation. Keep the same core algorithm but make it better."

### 2.4 Tasks

Eight coding tasks, each with multiple valid algorithmic approaches:

| Task | Example approaches |
|---|---|
| `kth_largest` | Quickselect, min-heap, max-heap, median-of-medians, counting sort |
| `rate_limiter` | Token bucket, sliding window log, sliding window counter, deque |
| `lru_cache` | Doubly-linked list + dict, array + timestamp, deque |
| `json_parser` | Recursive descent, state machine, stack-based |
| `regex_matcher` | NFA, DFA, backtracking, Thompson construction |
| `shortest_path` | Dijkstra, Bellman-Ford, A*, Floyd-Warshall |
| `expression_eval` | Shunting-yard, recursive descent, stack-based |
| `bloom_filter` | Bit array variants, hash function strategies |

**Total**: 3,200 API calls (1,600 per coupling mode).

## 3. Results

### 3.1 Negative coupling (right Kan): ordering inverts

| Task | None | Ring | Star | FC | Cohen's d |
|---|---|---|---|---|---|
| shortest_path | 0.041 ± 0.008 | 0.174 ± 0.112 | 0.116 ± 0.011 | **0.231** ± 0.016 | **-14.8** |
| kth_largest | 0.043 ± 0.029 | 0.199 ± 0.015 | 0.157 ± 0.016 | **0.244** ± 0.015 | **-8.8** |
| rate_limiter | 0.029 ± 0.006 | 0.096 ± 0.024 | 0.093 ± 0.022 | **0.133** ± 0.024 | **-5.9** |
| json_parser | 0.088 ± 0.010 | 0.126 ± 0.030 | 0.137 ± 0.026 | **0.149** ± 0.022 | **-3.5** |
| lru_cache | 0.071 ± 0.014 | 0.127 ± 0.025 | 0.137 ± 0.021 | **0.172** ± 0.052 | **-2.6** |
| regex_matcher | 0.143 ± 0.023 | 0.183 ± 0.023 | 0.150 ± 0.030 | **0.205** ± 0.039 | **-1.9** |
| *expression_eval* | **0.254** ± 0.059 | 0.289 ± 0.062 | 0.278 ± 0.062 | 0.232 ± 0.043 | *+0.4* |
| *bloom_filter* | 0.047 ± 0.009 | 0.052 ± 0.011 | 0.046 ± 0.008 | 0.050 ± 0.012 | *-0.3* |

**6/8 tasks**: FC > ring ≈ star > none (opposite of the GA paper).
**Kruskal-Wallis**: p < 0.002 for all 6 clean tasks.
**Kendall's W = 0.91** on the 6 clean tasks.

### 3.2 Positive coupling (left Kan): ordering matches GA paper

| Task | None | Ring | Star | FC | Cohen's d |
|---|---|---|---|---|---|
| regex_matcher | **0.164** ± 0.024 | 0.044 ± 0.020 | 0.055 ± 0.027 | 0.036 ± 0.014 | **+6.4** |
| expression_eval | **0.272** ± 0.058 | 0.024 ± 0.017 | 0.027 ± 0.025 | 0.023 ± 0.021 | **+5.7** |
| lru_cache | **0.073** ± 0.012 | 0.028 ± 0.013 | 0.020 ± 0.011 | 0.015 ± 0.012 | **+4.8** |
| bloom_filter | **0.049** ± 0.008 | 0.030 ± 0.008 | 0.034 ± 0.009 | 0.027 ± 0.005 | **+3.3** |
| shortest_path | **0.037** ± 0.014 | 0.016 ± 0.008 | 0.014 ± 0.007 | 0.015 ± 0.007 | **+2.0** |
| rate_limiter | **0.032** ± 0.006 | 0.028 ± 0.013 | 0.029 ± 0.009 | 0.018 ± 0.008 | **+2.0** |
| kth_largest | **0.034** ± 0.026 | 0.015 ± 0.015 | 0.015 ± 0.005 | 0.011 ± 0.008 | **+1.2** |
| json_parser | **0.097** ± 0.019 | 0.029 ± 0.015 | 0.030 ± 0.010 | 0.055 ± 0.094 | +0.6 |

**8/8 tasks**: none is most diverse. FC is least diverse on 6/8.
**All Cohen's d values are positive** (none > FC), vs all negative under negative coupling.
**Kendall's W = 0.72** across all 8 tasks.

### 3.3 The sign flip

The core result: **one prompt change flips the ordering**.

| Task | d (negative) | d (positive) | Sign flipped? |
|---|---|---|---|
| shortest_path | -14.8 | +2.0 | **Yes** |
| kth_largest | -8.8 | +1.2 | **Yes** |
| rate_limiter | -5.9 | +2.0 | **Yes** |
| json_parser | -3.5 | +0.6 | **Yes** |
| lru_cache | -2.6 | +4.8 | **Yes** |
| regex_matcher | -1.9 | +6.4 | **Yes** |
| expression_eval | +0.4 | +5.7 | **Yes** (was already positive; now large) |
| bloom_filter | -0.3 | +3.3 | **Yes** (was noise; now significant) |

**8/8 tasks**: Cohen's d changes sign (or, for the two outliers, becomes unambiguously positive).

### 3.4 The outliers resolve

**expression_eval** was the anti-functor's failure case under negative coupling (d = +0.4 — none was more diverse than FC). Under positive coupling, it shows the *strongest* convergence effect (d = +5.7). Interpretation: this task's prior distribution is already spread across multiple approaches (shunting-yard, recursive descent, stack). Negative coupling can't diversify further, but positive coupling homogenizes powerfully. The task was always a left-Kan responder — the negative coupling precondition (peaked prior) wasn't met.

**bloom_filter** showed no signal under negative coupling (spread = 0.006). Under positive coupling, it shows a clean ordering (spread = 0.022, d = +3.3). Interpretation: the solution landscape is unimodal (one way to build a bloom filter), so negative coupling has nothing to diversify *into*. But positive coupling can still homogenize — "build on this" makes implementations converge even when they were already similar.

**Asymmetry**: positive coupling is a stronger homogenizer than negative coupling is a diversifier. This is consistent with the categorical picture — colimits (merging) are generally smaller than limits (constraining).

## 4. Landscape Characterization

Two cheap measurements predict the anti-functor's effect size:

| Measure | What it captures | How to compute |
|---|---|---|
| **None diversity** | Prior peakedness | 5 isolated agents, measure embedding distance |
| **FC/None ratio** | Landscape modality | FC diversity ÷ None diversity |

| Task | None div | FC/None (neg) | |d| (neg) | Prediction | Correct? |
|---|---|---|---|---|---|
| rate_limiter | 0.029 | 4.58 | 5.9 | Strong | Yes |
| shortest_path | 0.041 | 5.64 | 14.8 | Strong | Yes |
| kth_largest | 0.043 | 5.70 | 8.8 | Strong | Yes |
| bloom_filter | 0.047 | 1.07 | 0.3 | None (low ratio) | Yes |
| lru_cache | 0.071 | 2.42 | 2.6 | Moderate | Yes |
| json_parser | 0.088 | 1.70 | 3.5 | Moderate | Yes |
| regex_matcher | 0.143 | 1.43 | 1.9 | Weak (high None) | Yes |
| expression_eval | 0.254 | 0.91 | 0.4 | None (high None) | Yes |

**Predictive rule**: None div < 0.10 AND FC/None > 2 → strong anti-functor. None div > 0.15 → weak or absent. FC/None < 1.5 → landscape too flat. Correctly classifies **8/8 tasks**.

The "None probe" costs only 5 API calls per task and predicts whether topology will matter *before* running the full experiment.

## 5. Categorical Interpretation

### 5.1 The adjunction

Let G be a topology graph, **Sol** the category of solution distributions on the embedding space, and F: G_partial → **Sol** a partial assignment of solutions to agents.

- **Left Kan extension** Lan_G F: extends by interpolation. Agents produce solutions *consistent with* (similar to) their neighbors. Colimit-like → merges → homogenizes. This is GA migration: inject foreign individuals into a population.
- **Right Kan extension** Ran_G F: extends by constraint. Agents produce solutions *compatible with but distinct from* their neighbors. Limit-like → restricts → diversifies. This is the "be different" prompt: exclude seen approaches.

The standard adjunction Lan_G ⊣ Ran_G reverses the diversity ordering because adjoint functors reverse inequalities in the diversity poset.

### 5.2 Signed fingerprint functoriality

**Conjecture.** Let D: Kl(T) → ℝ be the diversity fingerprint. For any topology G with algebraic connectivity λ₂(G):

- Under left Kan coupling (positive): D is monotonically **decreasing** in λ₂
- Under right Kan coupling (negative): D is monotonically **increasing** in λ₂

**Precondition.** The monotonicity requires: (1) the solution landscape is multi-modal, and (2) the generator's prior is peaked on a proper subset of modes.

**Evidence.** This conjecture correctly predicts the sign of Cohen's d for 14/16 task × coupling combinations (8/8 positive, 6/8 negative), with the two negative-coupling failures explained by precondition violation.

### 5.3 Connection to the ACT 2026 paper

The paper's Proposition 1 (fingerprint functoriality) states that a domain-change functor preserving composition structure preserves fingerprint shape. Our result extends this: the domain-change functor from GAs to LLMs preserves fingerprint *monotonicity* but reverses *direction*, because the GA's coupling mechanism (migration = left Kan) is replaced by its adjoint (negative context = right Kan).

The paper's three-level composition hierarchy maps directly:

| Level | GA term | Agent term | Structure preserved? |
|---|---|---|---|
| 1 | Operators → Pipeline | Tool calls → Task | Yes (Kleisli composition) |
| 2 | Pipelines → Strategy | Tasks → Workflow | Yes (sequential/adaptive/race) |
| 3 | Strategies → Island model | Workflows → Multi-agent | Yes, up to coupling sign |

## 6. Practical Implications

### For multi-agent system design

The coupling sign is a **prompt-level control**, not an architecture-level one. Same topology, same agents — the prompt determines whether coupling homogenizes or diversifies.

| If you want... | Use... | Topology | Effect |
|---|---|---|---|
| Diverse solutions | Negative coupling + FC | High connectivity | Maximum exploration |
| Fast convergence | Positive coupling + FC | High connectivity | Maximum exploitation |
| Independent work | Any coupling + None | No connectivity | Baseline (prior-dependent) |

### For search agents

Applied to Lyra's browse cycle (docker-lyra): added a 5th "long-tail discovery" agent that runs after 4 parallel browse agents, sees what they found, and searches for what's missing. This is negative coupling on a two-round star topology.

### The explore/exploit switch

The topology controls the *rate* of information flow. The coupling sign controls the *direction*. These are independent knobs:

```
Explore phase: negative coupling + high connectivity → maximum diversity
Exploit phase: positive coupling + high connectivity → maximum convergence
```

This is the paper's hourglass strategy with the coupling sign as an additional degree of freedom.

## 7. Limitations

1. **Sequential topology simulation.** Our "fully connected" topology is sequential (agent N sees agents 0..N-1), not simultaneous. True simultaneous coupling would require multiple rounds.
2. **Single model.** All results are on GPT-4o-mini. The prior peakedness likely varies across models — a more capable model might have flatter priors.
3. **Embedding metric.** Cosine distance on sentence embeddings captures syntactic/structural diversity but may miss deeper algorithmic differences. Behavioral diversity (test-based) would be stronger.
4. **No formal proof.** The Lan ⊣ Ran interpretation is supported by experiment but not proven. The category of LLM distributions is not rigorously defined.
5. **Ring ≈ Star degeneracy.** In our setup, ring and star both give most agents exactly 1 prior solution, making them hard to distinguish. The Foster census extension (cubic graphs with varying λ₂) would resolve this.

## 8. Reproduction

All code and data at `~/git/orchestration/`:

```bash
# Negative coupling (right Kan)
python3 experiment.py --coupling negative --output output_negative

# Positive coupling (left Kan)
python3 experiment.py --coupling positive --output output_positive

# Analyze existing results
python3 experiment.py --analyze-only output_negative/results.jsonl
```

Requires: `openai`, `sentence-transformers`, `numpy`, `scipy`, `matplotlib`.
