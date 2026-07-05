"""
LLM Topology Diversity Experiment
=================================

Tests whether migration topology determines solution diversity in multi-agent
LLM systems, analogous to the ACT 2026 paper "From Games to Graphs".

The paper shows: none > ring > star > fully_connected for diversity,
with Kendall's W = 1.0 across six domains.

We test: do LLM sub-agents follow the same ordering when "migration"
is replaced by "showing prior solutions as context"?

Diversity metric: mean pairwise cosine distance on sentence embeddings.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import combinations

import numpy as np
import openai

# ============================================================
# Configuration
# ============================================================

TASKS = {
    "kth_largest": (
        "Write a Python function `kth_largest(nums: list[int], k: int) -> int` "
        "that returns the k-th largest element in an unsorted list. "
        "Do NOT use the built-in sort or sorted. Implement the core algorithm yourself. "
        "Include a brief docstring explaining which algorithm you chose and its time complexity. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "rate_limiter": (
        "Write a Python class `RateLimiter` that allows at most `max_requests` calls "
        "to an `allow()` method within a sliding `window_seconds` time window. "
        "The constructor signature is `__init__(self, max_requests: int, window_seconds: float)`. "
        "The `allow()` method returns True if the request is allowed, False otherwise. "
        "Include a brief docstring explaining which algorithm you chose. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "lru_cache": (
        "Write a Python class `LRUCache` with `__init__(self, capacity: int)`, "
        "`get(self, key: int) -> int` (returns -1 if not found), and "
        "`put(self, key: int, value: int) -> None`. Both get and put must be O(1). "
        "Do NOT use `functools.lru_cache` or `collections.OrderedDict`. "
        "Include a brief docstring explaining your data structure choice. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "json_parser": (
        "Write a Python function `parse_json(s: str) -> object` that parses a JSON string "
        "into Python objects (dict, list, str, int/float, bool, None). "
        "Do NOT use the `json` module or any external libraries. Implement the parser yourself. "
        "Handle strings (with escapes), numbers, booleans, null, arrays, and objects. "
        "Include a brief docstring explaining which parsing technique you chose. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "regex_matcher": (
        "Write a Python function `match(pattern: str, text: str) -> bool` that implements "
        "basic regex matching supporting: literal characters, '.' (any char), '*' (zero or more "
        "of the preceding element), and '+' (one or more of the preceding). "
        "Do NOT use the `re` module. Implement the matching algorithm yourself. "
        "Include a brief docstring explaining which algorithm you chose. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "shortest_path": (
        "Write a Python function `shortest_path(graph: dict[str, list[tuple[str, float]]], "
        "start: str, end: str) -> tuple[float, list[str]]` that returns the shortest distance "
        "and the path as a list of node names in a weighted directed graph. "
        "The graph is an adjacency list: {node: [(neighbor, weight), ...]}. "
        "Return (float('inf'), []) if no path exists. "
        "Include a brief docstring explaining which algorithm you chose. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "expression_eval": (
        "Write a Python function `evaluate(expr: str) -> float` that evaluates arithmetic "
        "expressions with +, -, *, /, parentheses, and floating-point numbers. "
        "Respect standard operator precedence (* and / before + and -). "
        "Do NOT use `eval()` or `ast.literal_eval()`. Implement the parser yourself. "
        "Include a brief docstring explaining which parsing/evaluation technique you chose. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
    "bloom_filter": (
        "Write a Python class `BloomFilter` with `__init__(self, expected_items: int, "
        "false_positive_rate: float)`, `add(self, item: str) -> None`, and "
        "`contains(self, item: str) -> bool`. Calculate optimal bit array size and number "
        "of hash functions from the parameters. Do NOT use external libraries for the "
        "bit array or hashing — implement them yourself (you may use hashlib). "
        "Include a brief docstring explaining your design choices. "
        "Return ONLY the Python code, no explanation outside the code."
    ),
}

COUPLING_PROMPTS = {
    "negative": (
        "\n\nHere are solutions from other engineers on your team. "
        "You MUST use a DIFFERENT algorithm or approach than the ones shown. "
        "Do not copy or trivially modify their code — write your own implementation "
        "using a fundamentally different strategy.\n"
    ),
    "positive": (
        "\n\nHere are solutions from other engineers on your team. "
        "BUILD ON the best approach shown — take the strongest solution, "
        "refine it, fix any issues, and improve its implementation. "
        "Keep the same core algorithm but make it better.\n"
    ),
}

@dataclass
class ExperimentConfig:
    n_agents: int = 5
    n_seeds: int = 10
    n_rounds: int = 1          # Start with 1 round (simpler, still tests the core claim)
    topologies: list = field(default_factory=lambda: ["none", "ring", "star", "fully_connected"])
    tasks: list = field(default_factory=lambda: list(TASKS.keys()))
    coupling: str = "negative"  # "negative" (be different / right Kan) or "positive" (build on / left Kan)
    model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 2048
    embedding_model: str = "all-MiniLM-L6-v2"


# ============================================================
# Topology: determines what context each agent sees
# ============================================================

def get_context(topology: str, agent_idx: int, solutions: list[str]) -> list[str]:
    """Return prior solutions this agent should see, based on topology.

    `solutions` contains the solutions generated so far in this round
    (agents 0..agent_idx-1).

    Topologies:
      none:             [] always — complete isolation
      ring:             [solutions[agent_idx - 1]] — see one predecessor
      star:             [solutions[0]] for agents 1+ — hub only
      fully_connected:  solutions[:agent_idx] — see all predecessors
    """
    if not solutions or agent_idx == 0:
        return []

    if topology == "none":
        return []
    elif topology == "ring":
        return [solutions[agent_idx - 1]]
    elif topology == "star":
        return [solutions[0]]
    elif topology == "fully_connected":
        return list(solutions[:agent_idx])
    else:
        raise ValueError(f"Unknown topology: {topology}")


# ============================================================
# Agent: calls the Anthropic API
# ============================================================

def generate_solution(
    client: openai.OpenAI,
    task_prompt: str,
    prior_solutions: list[str],
    config: ExperimentConfig,
) -> str:
    """Generate a solution, optionally conditioned on prior solutions."""

    system = (
        "You are a senior software engineer. Write clean, working Python code. "
        "Be creative and choose your own algorithmic approach. "
        "Do not explain your code outside of docstrings."
    )

    user_parts = [task_prompt]

    if prior_solutions:
        user_parts.append(COUPLING_PROMPTS[config.coupling])
        for i, sol in enumerate(prior_solutions):
            user_parts.append(f"\n--- Solution {i+1} ---\n```python\n{sol}\n```\n")

    resp = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
    )

    text = resp.choices[0].message.content

    # Extract code from markdown fences if present
    if "```python" in text:
        code = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        code = text.split("```", 1)[1].split("```", 1)[0]
    else:
        code = text

    return code.strip()


# ============================================================
# Diversity: embedding cosine distance
# ============================================================

class DiversityMeter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def pairwise_cosine_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine distance = 1 - cosine_similarity. Embeddings are L2-normalized."""
        sim = embeddings @ embeddings.T
        return 1.0 - sim

    def mean_pairwise_distance(self, texts: list[str]) -> tuple[float, np.ndarray]:
        """Compute mean pairwise cosine distance. Returns (scalar, distance_matrix)."""
        emb = self.embed(texts)
        dist = self.pairwise_cosine_distance(emb)
        n = len(texts)
        # Upper triangle (excluding diagonal)
        indices = list(combinations(range(n), 2))
        distances = [dist[i, j] for i, j in indices]
        return float(np.mean(distances)), dist


# ============================================================
# Main experiment loop
# ============================================================

def load_existing(output_dir: Path) -> tuple[list[dict], dict]:
    """Load previously completed results for resume capability."""
    results = []
    all_solutions = {}

    results_path = output_dir / "results.jsonl"
    solutions_dir = output_dir / "solutions"

    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    if solutions_dir.exists():
        for p in solutions_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
            key = (data["task"], data["topology"], data["seed"])
            all_solutions[key] = data["solutions"]

    return results, all_solutions


def save_result(output_dir: Path, result: dict, solutions: list[str],
                task: str, topology: str, seed: int):
    """Save a single run's results immediately (append-only, crash-safe)."""
    # Append result to JSONL (one line per run, append-safe)
    results_path = output_dir / "results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    # Save solutions to individual file
    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(exist_ok=True)
    sol_path = solutions_dir / f"{task}_{topology}_seed{seed}.json"
    with open(sol_path, "w") as f:
        json.dump({"task": task, "topology": topology, "seed": seed,
                    "solutions": solutions}, f, indent=2)


def run_experiment(config: ExperimentConfig, output_dir: Path):
    client = openai.OpenAI()
    meter = DiversityMeter(config.embedding_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results and skip completed runs
    results, all_solutions = load_existing(output_dir)
    completed = {(r["task"], r["topology"], r["seed"]) for r in results}
    if completed:
        print(f"Resuming: {len(completed)} runs already completed, skipping them.")

    total_runs = len(config.tasks) * len(config.topologies) * config.n_seeds
    run_count = len(completed)

    for task_name in config.tasks:
        task_prompt = TASKS[task_name]

        for topology in config.topologies:
            for seed in range(config.n_seeds):
                if (task_name, topology, seed) in completed:
                    continue

                run_count += 1
                print(f"\n[{run_count}/{total_runs}] task={task_name} "
                      f"topo={topology} seed={seed}")

                solutions = []
                for agent_idx in range(config.n_agents):
                    context = get_context(topology, agent_idx, solutions)
                    print(f"  Agent {agent_idx}: {len(context)} prior solutions in context")

                    try:
                        sol = generate_solution(
                            client, task_prompt, context, config
                        )
                        solutions.append(sol)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        solutions.append(f"# Error: {e}")

                # Compute diversity
                diversity, dist_matrix = meter.mean_pairwise_distance(solutions)

                result = {
                    "task": task_name,
                    "topology": topology,
                    "seed": seed,
                    "coupling": config.coupling,
                    "n_agents": config.n_agents,
                    "diversity": diversity,
                    "solution_hashes": [
                        hashlib.md5(s.encode()).hexdigest()[:8] for s in solutions
                    ],
                    "solution_lengths": [len(s) for s in solutions],
                }
                results.append(result)
                all_solutions[(task_name, topology, seed)] = solutions

                # Save immediately — crash-safe
                save_result(output_dir, result, solutions,
                            task_name, topology, seed)

                print(f"  Diversity: {diversity:.4f}")

    print(f"\nAll {len(results)} runs complete. Results in {output_dir}/")
    return results


# ============================================================
# Analysis
# ============================================================

def analyze(results: list[dict]):
    """Print summary statistics and test the topology ordering hypothesis."""
    from scipy import stats

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    tasks = sorted(set(r["task"] for r in results))
    topologies = ["none", "ring", "star", "fully_connected"]

    # Per-task diversity by topology
    for task in tasks:
        print(f"\n--- {task} ---")
        task_results = [r for r in results if r["task"] == task]

        topo_diversities = {}
        for topo in topologies:
            divs = [r["diversity"] for r in task_results if r["topology"] == topo]
            if divs:
                topo_diversities[topo] = divs
                mean = np.mean(divs)
                std = np.std(divs)
                print(f"  {topo:20s}: {mean:.4f} ± {std:.4f}  (n={len(divs)})")

        # Kruskal-Wallis H test
        groups = [topo_diversities[t] for t in topologies if t in topo_diversities]
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            try:
                H, p = stats.kruskal(*groups)
                print(f"\n  Kruskal-Wallis H = {H:.3f}, p = {p:.6f}")
            except ValueError as e:
                print(f"\n  Kruskal-Wallis: {e}")

            # Pairwise Mann-Whitney U
            print("  Pairwise Mann-Whitney U:")
            for i, t1 in enumerate(topologies):
                for t2 in topologies[i+1:]:
                    if t1 in topo_diversities and t2 in topo_diversities:
                        U, p_mw = stats.mannwhitneyu(
                            topo_diversities[t1], topo_diversities[t2],
                            alternative="two-sided"
                        )
                        print(f"    {t1} vs {t2}: U={U:.1f}, p={p_mw:.4f}")

        # Cohen's d: none vs fully_connected
        if "none" in topo_diversities and "fully_connected" in topo_diversities:
            d_none = np.array(topo_diversities["none"])
            d_full = np.array(topo_diversities["fully_connected"])
            pooled_std = np.sqrt((d_none.var() + d_full.var()) / 2)
            if pooled_std > 0:
                cohens_d = (d_none.mean() - d_full.mean()) / pooled_std
                print(f"\n  Cohen's d (none vs fully_connected) = {cohens_d:.3f}")

    # Cross-task topology ordering (Kendall's W)
    if len(tasks) >= 2:
        print(f"\n--- Cross-Task Agreement ---")
        rankings = []
        for task in tasks:
            task_results = [r for r in results if r["task"] == task]
            means = {}
            for topo in topologies:
                divs = [r["diversity"] for r in task_results if r["topology"] == topo]
                if divs:
                    means[topo] = np.mean(divs)
            if len(means) == len(topologies):
                # Rank: higher diversity = rank 1
                sorted_topos = sorted(means.keys(), key=lambda t: -means[t])
                ranking = [sorted_topos.index(t) + 1 for t in topologies]
                rankings.append(ranking)
                print(f"  {task}: {' > '.join(sorted_topos)} "
                      f"(diversities: {[f'{means[t]:.4f}' for t in sorted_topos]})")

        if len(rankings) >= 2:
            # Kendall's W
            rankings_array = np.array(rankings)
            m, k = rankings_array.shape  # m judges, k items
            rank_sums = rankings_array.sum(axis=0)
            mean_rank_sum = np.mean(rank_sums)
            S = np.sum((rank_sums - mean_rank_sum) ** 2)
            W = 12 * S / (m**2 * (k**3 - k))
            print(f"\n  Kendall's W = {W:.4f}")
            print(f"  (1.0 = perfect agreement across tasks; "
                  f"paper reports W = 1.0 across 6 GA domains)")


def plot_results(results: list[dict], output_dir: Path):
    """Generate a bar chart of diversity by topology, per task."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    tasks = sorted(set(r["task"] for r in results))
    topologies = ["none", "ring", "star", "fully_connected"]
    topo_labels = ["None\n(strict)", "Ring", "Star", "Fully\nConnected"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]

    for ax, task in zip(axes, tasks):
        task_results = [r for r in results if r["task"] == task]
        means, stds = [], []
        for topo in topologies:
            divs = [r["diversity"] for r in task_results if r["topology"] == topo]
            means.append(np.mean(divs) if divs else 0)
            stds.append(np.std(divs) if divs else 0)

        bars = ax.bar(range(len(topologies)), means, yerr=stds,
                      color=colors, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(topologies)))
        ax.set_xticklabels(topo_labels, fontsize=9)
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Pairwise Cosine Distance" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    coupling = results[0].get("coupling", "negative") if results else "negative"
    kan_label = "Right Kan / be different" if coupling == "negative" else "Left Kan / build on"
    fig.suptitle(
        f"Solution Diversity by Agent Topology — {coupling.upper()} coupling\n"
        f"({kan_label})",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    plot_path = output_dir / "diversity_by_topology.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Topology Diversity Experiment")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--task", type=str, default=None,
                        help="Run only this task (default: all)")
    parser.add_argument("--topology", type=str, default=None,
                        help="Run only this topology (default: all)")
    parser.add_argument("--coupling", type=str, default="negative",
                        choices=["negative", "positive"],
                        help="Coupling sign: 'negative' (be different / right Kan) "
                             "or 'positive' (build on / left Kan)")
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Skip experiment, analyze existing results.jsonl")
    parser.add_argument("--output", type=str,
                        default="output")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.analyze_only:
        results = []
        with open(args.analyze_only) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        if not results:
            with open(args.analyze_only) as f:
                results = json.load(f)
        analyze(results)
        plot_results(results, Path(args.analyze_only).parent)
    else:
        config = ExperimentConfig(
            n_agents=args.n_agents,
            n_seeds=args.n_seeds,
            coupling=args.coupling,
        )
        if args.task:
            config.tasks = [args.task]
        if args.topology:
            config.topologies = [args.topology]

        results = run_experiment(config, output_dir)
        analyze(results)
        plot_results(results, output_dir)
