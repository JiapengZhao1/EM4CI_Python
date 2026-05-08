"""
em4ci_core.py — self-contained EM4CI learning + causal inference pipeline.

Usage examples
--------------
# Learn one template
python em4ci_core.py learn models_xdsl/em_ex1_TD2_10_ED2_0.xdsl data/100/ex1_TD2_10.csv 100

# Infer (compare true vs learned model)
python em4ci_core.py infer models_xdsl/ex1_TD2_10.xdsl <learned>.xdsl Y --do X --num-samples 100

# Run full pipeline for experiment set across sample sizes
python em4ci_core.py run --experiment-set small --sample-sizes 100 1000 --domains 2 4 --restarts 10
python em4ci_core.py run --experiments ex1_TD2_10 ex3_TD2_10 --sample-sizes 100 --domains 2
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd

# Silence pgmpy's verbose datatype-inference INFO messages
logging.getLogger("pgmpy").setLevel(logging.WARNING)

from pgmpy.estimators import ExpectationMaximization, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

try:
    from pgmpy.models import DiscreteBayesianNetwork as _BN
except ImportError:
    from pgmpy.models import BayesianNetwork as _BN

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class NodeSpec:
    node_id: str
    states: List[str]
    parents: List[str]
    probabilities: List[float]

@dataclass
class NetworkSpec:
    name: str
    nodes: Dict[str, NodeSpec]
    node_order: List[str]

    def latent_nodes(self, observed: Sequence[str]) -> List[str]:
        obs = set(observed)
        return [n for n in self.node_order if n not in obs]

# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def read_xdsl(path: str | Path) -> Tuple[ET.ElementTree, NetworkSpec]:
    tree = ET.parse(path)
    root = tree.getroot()
    nodes_root = root.find("nodes")
    nodes: Dict[str, NodeSpec] = {}
    order: List[str] = []
    for cpt in nodes_root.findall("cpt"):
        nid = cpt.attrib["id"]
        states = [s.attrib["id"] for s in cpt.findall("state")]
        parents = (cpt.findtext("parents") or "").split() or []
        probs = [float(v) for v in (cpt.findtext("probabilities") or "").split()] or []
        nodes[nid] = NodeSpec(nid, states, parents, probs)
        order.append(nid)
    return tree, NetworkSpec(root.attrib.get("id", Path(path).stem), nodes, order)


def _make_cpd(spec: NetworkSpec, node: NodeSpec) -> TabularCPD:
    vc = len(node.states)
    ev = node.parents
    ev_card = [len(spec.nodes[p].states) for p in ev]
    cols = math.prod(ev_card) if ev_card else 1
    probs = np.asarray(node.probabilities, dtype=float)
    if len(probs) == vc * cols:
        values = probs.reshape(cols, vc).T
    else:  # probabilities omit last row (XDSL compact format)
        partial = probs.reshape(cols, vc - 1).T
        values = np.vstack([partial, np.clip(1 - partial.sum(0, keepdims=True), 0, 1)])
    sn = {node.node_id: node.states, **{p: spec.nodes[p].states for p in ev}}
    return TabularCPD(node.node_id, vc, values, evidence=ev or None,
                      evidence_card=ev_card or None, state_names=sn)


def build_model(spec: NetworkSpec, latent_nodes: Sequence[str] = ()) -> _BN:
    edges = [(p, n.node_id) for n in spec.nodes.values() for p in n.parents]
    try:
        model = _BN(edges, latents=set(latent_nodes))
    except TypeError:
        model = _BN(edges)
        model.latents = set(latent_nodes)
    for nid in spec.node_order:
        if nid not in model.nodes():
            model.add_node(nid)
    return model


def load_xdsl(path: str | Path) -> _BN:
    """Load a fully-parameterised BN from an XDSL file."""
    _, spec = read_xdsl(path)
    model = build_model(spec)
    model.add_cpds(*[_make_cpd(spec, spec.nodes[nid]) for nid in spec.node_order])
    model.check_model()
    return model


def save_xdsl(template_tree: ET.ElementTree, model: _BN, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = copy.deepcopy(template_tree)
    nodes_root = tree.getroot().find("nodes")
    for cpt_el in nodes_root.findall("cpt"):
        cpd = model.get_cpds(cpt_el.attrib["id"])
        if cpd is None:
            continue
        prob_text = " ".join(f"{v:.10g}" for v in np.asarray(cpd.get_values()).T.reshape(-1))
        el = cpt_el.find("probabilities")
        if el is None:
            el = ET.SubElement(cpt_el, "probabilities")
        el.text = prob_text
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path

# ─────────────────────────────────────────────
# Learning
# ─────────────────────────────────────────────

_TEMPLATE_RE = re.compile(r"^em_(?P<model>.+)_ED(?P<domain>\d+)_(?P<run>\d+)$")


def bic(ll: float, domain: int, n: int, num_latent: int) -> float:
    return -2 * ll + num_latent * (domain - 1) * math.log(n)


def marginal_ll(model: _BN, data: pd.DataFrame) -> float:
    """Observed-data log-likelihood via Variable Elimination."""
    latents = set(getattr(model, "latents", set()) or set())
    obs_cols = [c for c in data.columns if c not in latents]
    infer = VariableElimination(model)
    joint = infer.query(obs_cols, evidence={}, show_progress=False)
    eps = np.finfo(float).tiny
    total = 0.0
    for row in data[obs_cols].to_dict(orient="records"):
        try:
            p = float(joint.get_value(**row))
        except Exception:
            p = eps
        total += math.log(max(p, eps))
    return total


def learn(
    template_path: str | Path,
    data_path: str | Path,
    num_samples: int,
    output_dir: str | Path | None = None,
    max_iter: int = 100,
) -> dict:
    template_path = Path(template_path)
    data_path = Path(data_path)
    template_tree, spec = read_xdsl(template_path)

    data = pd.read_csv(data_path, sep=None, engine="python")
    for col in data.columns:
        data[col] = data[col].astype(str).astype("category")

    template_name = template_path.stem
    m = _TEMPLATE_RE.match(template_name)
    model_name = m.group("model") if m else template_name
    domain = int(m.group("domain")) if m else None

    latent_vars = spec.latent_nodes(data.columns)
    latent_card = {v: len(spec.nodes[v].states) for v in latent_vars}
    model = build_model(spec, latent_nodes=latent_vars)

    t0 = time.perf_counter()
    if latent_vars:
        est = ExpectationMaximization(model, data)
        sig = set(inspect.signature(est.get_parameters).parameters)
        kwargs = {k: v for k, v in dict(latent_card=latent_card, max_iter=max_iter,
                                         n_jobs=1, show_progress=False).items() if k in sig}
        cpds = est.get_parameters(**kwargs)
    else:
        cpds = MaximumLikelihoodEstimator(model, data).get_parameters()
    elapsed = time.perf_counter() - t0

    model.add_cpds(*cpds)
    model.check_model()
    ll = marginal_ll(model, data)

    if output_dir is None:
        output_dir = Path("learned_models") / model_name / str(num_samples) / template_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_xdsl(template_tree, model, output_dir / f"{template_name}.xdsl")

    result = dict(
        model_name=model_name,
        template_name=template_name,
        log_likelihood=ll,
        bic_score=bic(ll, domain, num_samples, len(latent_vars)) if domain else None,
        elapsed_seconds=elapsed,
        num_hidden_nodes=len(latent_vars),
        hidden_nodes=latent_vars,
        learned_model_path=str(output_dir / f"{template_name}.xdsl"),
    )
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

# ─────────────────────────────────────────────
# Causal inference
# ─────────────────────────────────────────────

def _state_names(model: _BN, var: str) -> List[str]:
    cpd = model.get_cpds(var)
    sn = getattr(cpd, "state_names", {}) or {}
    return list(sn.get(var, [str(i) for i in range(cpd.variable_card)]))


def _mutilate(model: _BN, do_vars: Sequence[str]) -> _BN:
    """Graph surgery: remove parents of do_vars and replace CPDs with uniform."""
    mut = model.copy()
    for var in do_vars:
        cpd = mut.get_cpds(var)
        states = _state_names(mut, var)
        parents = list(mut.get_parents(var))
        if parents:
            mut.remove_edges_from([(p, var) for p in parents])
        mut.remove_cpds(cpd)
        u = 1.0 / len(states)
        mut.add_cpds(TabularCPD(var, len(states), [[u]] * len(states),
                                state_names={var: states}))
    mut.check_model()
    return mut


def _interventional_dists(model: _BN, query_var: str,
                           do_vars: Sequence[str]) -> Dict[tuple, List[float]]:
    """Returns {(do_val, ...): [P(query=s0|do), P(query=s1|do), ...]}"""
    mut = _mutilate(model, do_vars)
    infer = VariableElimination(mut)
    results = {}
    for assignment in product(*[_state_names(model, v) for v in do_vars]):
        evidence = dict(zip(do_vars, assignment))
        factor = infer.query([query_var], evidence=evidence, show_progress=False)
        results[assignment] = np.asarray(factor.values).reshape(-1).tolist()
    return results


def infer(
    true_model_path: str | Path,
    learned_model_path: str | Path,
    query_var: str,
    do_vars: Sequence[str],
    num_samples: int,
    output_dir: str | Path | None = None,
) -> dict:
    true_model = load_xdsl(true_model_path)
    learned_model = load_xdsl(learned_model_path)

    t0 = time.perf_counter()
    true_dists = _interventional_dists(true_model, query_var, do_vars)
    learned_dists = _interventional_dists(learned_model, query_var, do_vars)

    # Weights: P_true(do_vars) for weighted MAE
    infer_true = VariableElimination(true_model)
    joint = infer_true.query(list(do_vars), show_progress=False)
    weights = {a: float(joint.get_value(**dict(zip(do_vars, a)))) for a in true_dists}

    flat_true, flat_est, flat_w = [], [], []
    for a in true_dists:
        tv, ev = true_dists[a], learned_dists[a]
        w = weights[a]
        flat_true.extend(tv); flat_est.extend(ev)
        flat_w.extend([w] * len(tv))

    n = len(flat_true)
    avg_err = sum(abs(t - e) for t, e in zip(flat_true, flat_est)) / n
    w_err = sum(abs(t - e) * w for t, e, w in zip(flat_true, flat_est, flat_w))

    # ATE error (first do_var, first pair of states)
    do0 = do_vars[0]
    states0 = _state_names(true_model, do0)
    def _expected(dists, assignment):
        q_states = _state_names(true_model, query_var)
        return sum(p * (float(s) if s.replace('.','',1).isdigit() else float(i))
                   for i, (p, s) in enumerate(zip(dists[assignment], q_states)))
    ate_true = _expected(true_dists, (states0[0],)) - _expected(true_dists, (states0[1],)) if len(states0) >= 2 else 0.0
    ate_est  = _expected(learned_dists, (states0[0],)) - _expected(learned_dists, (states0[1],)) if len(states0) >= 2 else 0.0
    ate_err  = abs(ate_true - ate_est)
    elapsed = time.perf_counter() - t0

    result = dict(
        query_var=query_var, do_vars=list(do_vars), num_samples=int(num_samples),
        average_error=avg_err, weighted_error=w_err, ate_error=ate_err,
        elapsed_seconds=elapsed,
        true_dists={" | ".join(a): v for a, v in true_dists.items()},
        learned_dists={" | ".join(a): v for a, v in learned_dists.items()},
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "inference_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result

# ─────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────

SMALL_EXPERIMENTS = [
    "ex1_TD2_10", "ex2_TD2_10", "ex3_TD2_10", "ex4_TD2_10",
    "ex5_TD2_10", "ex6_TD2_10", "ex7_TD2_10", "ex8_TD2_10",
]
LARGE_EXPERIMENTS = [
    "49_chain_TD4_10", "99_chain_TD4_10",
    "17_diamond_TD4_10", "65_diamond_TD4_10",
    "6_cone_cloud_TD4_10", "15_cone_cloud_TD4_10",
]
EXPERIMENT_SETS = {
    "small": SMALL_EXPERIMENTS,
    "large": LARGE_EXPERIMENTS,
    "all": SMALL_EXPERIMENTS + LARGE_EXPERIMENTS,
}
DO_VARS = {
    "ex1_TD2_10": ["X"],  "ex2_TD2_10": ["X1"], "ex3_TD2_10": ["X"],
    "ex4_TD2_10": ["X1"], "ex5_TD2_10": ["X"],  "ex6_TD2_10": ["X1"],
    "ex7_TD2_10": ["X1"], "ex8_TD2_10": ["X"],
    "49_chain_TD4_10": ["V0"],  "99_chain_TD4_10": ["V0"],
    "17_diamond_TD4_10": ["V0"], "65_diamond_TD4_10": ["V0"],
    "6_cone_cloud_TD4_10": ["V5"], "15_cone_cloud_TD4_10": ["V14"],
}
QUERY_VARS = {
    "ex1_TD2_10": "Y", "ex2_TD2_10": "Y", "ex3_TD2_10": "Y", "ex4_TD2_10": "Y",
    "ex5_TD2_10": "Y", "ex6_TD2_10": "Y", "ex7_TD2_10": "Y", "ex8_TD2_10": "Y",
    "49_chain_TD4_10": "V48",  "99_chain_TD4_10": "V98",
    "17_diamond_TD4_10": "V16", "65_diamond_TD4_10": "V64",
    "6_cone_cloud_TD4_10": "V0", "15_cone_cloud_TD4_10": "V0",
}

# ─────────────────────────────────────────────
# Full pipeline: learn (multi-restart) + infer
# ─────────────────────────────────────────────

def _run_single_trial(
    experiment: str,
    num_samples: int,
    domains: Sequence[int],
    restarts: int,
    model_dir: Path,
    data_dir: Path,
    learned_model_dir: Path,
    output_dir: Path,
    max_iter: int,
    force_learn: bool,
) -> dict:
    """One learn+infer pass (no trial looping). Returns a result dict."""
    true_model_path = model_dir / f"{experiment}.xdsl"
    data_path = data_dir / str(num_samples) / f"{experiment}.csv"
    query_var = QUERY_VARS.get(experiment, "Y")
    do_vars   = DO_VARS.get(experiment, ["X"])

    best_bic, best_model_path = float("inf"), None

    for domain in domains:
        for run_idx in range(restarts):
            template_name = f"em_{experiment}_ED{domain}_{run_idx}"
            template_path = model_dir / f"{template_name}.xdsl"
            if not template_path.exists():
                print(f"  [skip] template not found: {template_path}")
                continue

            run_output_dir = learned_model_dir / experiment / template_name
            metrics_file = run_output_dir / "metrics.json"

            if metrics_file.exists() and not force_learn:
                result = json.loads(metrics_file.read_text())
                # Old pipeline metrics.json may lack "learned_model_path"; derive it
                if "learned_model_path" not in result:
                    result["learned_model_path"] = str(run_output_dir / f"{template_name}.xdsl")
            else:
                print(f"  learning {template_name} ...", flush=True)
                result = learn(template_path, data_path, num_samples,
                               output_dir=run_output_dir, max_iter=max_iter)
                ll = result.get("log_likelihood")
                bic_val_print = result.get("bic_score")
                if ll is not None:
                    bic_str = f"  BIC={bic_val_print:.4f}" if bic_val_print is not None else ""
                    print(f"    LL={ll:.4f}{bic_str}", flush=True)

            bic_val = result.get("bic_score")
            candidate = Path(result["learned_model_path"])
            if bic_val is not None and bic_val < best_bic and candidate.exists():
                best_bic = bic_val
                best_model_path = candidate

    if best_model_path is None or not best_model_path.exists():
        return {"experiment": experiment, "num_samples": num_samples, "status": "no_model"}

    inf_output_dir = output_dir / experiment
    inf_result = infer(true_model_path, best_model_path, query_var, do_vars,
                       num_samples, output_dir=inf_output_dir)
    return {
        "experiment": experiment,
        "num_samples": num_samples,
        "best_bic": best_bic,
        "best_model": str(best_model_path),
        **{k: inf_result[k] for k in ("average_error", "weighted_error", "ate_error")},
        "status": "ok",
    }


def run_experiment(
    experiment: str,
    num_samples: int,
    domains: Sequence[int],
    restarts: int,
    model_dir: str | Path = "models_xdsl",
    data_dir: str | Path = "data",
    learned_model_dir: str | Path = "learned_models",
    output_dir: str | Path = "output",
    max_iter: int = 100,
    force_learn: bool = False,
    trials: int = 1,
) -> dict:
    """Learn (all restarts × domains, pick best BIC) then run causal inference.

    When trials > 1, each trial runs in its own sub-directory with force_learn=True
    (independent random EM seeds) and results are aggregated as mean ± std.
    """
    model_dir = Path(model_dir)
    data_dir  = Path(data_dir)
    learned_model_dir = Path(learned_model_dir)
    output_dir = Path(output_dir)

    if trials == 1:
        return _run_single_trial(
            experiment, num_samples, domains, restarts,
            model_dir, data_dir, learned_model_dir, output_dir, max_iter, force_learn,
        )

    # ── multi-trial: each trial gets its own directories ──
    ERROR_KEYS = ("average_error", "weighted_error", "ate_error")
    trial_rows: List[dict] = []
    for trial_i in range(trials):
        print(f"  [trial {trial_i + 1}/{trials}]", flush=True)
        t_learned_dir = learned_model_dir / f"trial_{trial_i:02d}"
        t_output_dir  = output_dir / f"trial_{trial_i:02d}"
        row = _run_single_trial(
            experiment, num_samples, domains, restarts,
            model_dir, data_dir, t_learned_dir, t_output_dir,
            max_iter, force_learn=True,   # always re-learn for independent seeds
        )
        trial_rows.append(row)

    ok_rows = [r for r in trial_rows if r.get("status") == "ok"]
    if not ok_rows:
        return {"experiment": experiment, "num_samples": num_samples,
                "trials": trials, "status": "no_model"}

    agg: dict = {"experiment": experiment, "num_samples": num_samples,
                 "trials": trials, "num_ok_trials": len(ok_rows), "status": "ok",
                 "trial_results": trial_rows}
    for key in ERROR_KEYS:
        vals = [r[key] for r in ok_rows]
        agg[key]             = float(sum(vals) / len(vals))
        agg[f"{key}_std"]   = float((sum((v - agg[key]) ** 2 for v in vals) / len(vals)) ** 0.5)
        agg[f"{key}_min"]   = float(min(vals))
        agg[f"{key}_max"]   = float(max(vals))
    return agg


def run_all(
    experiments: Sequence[str],
    sample_sizes: Sequence[int],
    domains: Sequence[int],
    restarts: int,
    model_dir: str | Path = "models_xdsl",
    data_dir: str | Path = "data",
    learned_model_dir: str | Path = "learned_models",
    output_dir: str | Path = "output",
    max_iter: int = 100,
    force_learn: bool = False,
    trials: int = 1,
) -> List[dict]:
    rows = []
    for exp in experiments:
        for n in sample_sizes:
            print(f"\n=== {exp}  N={n}  trials={trials} ===", flush=True)
            row = run_experiment(exp, n, domains, restarts,
                                 model_dir=model_dir, data_dir=data_dir,
                                 learned_model_dir=learned_model_dir,
                                 output_dir=output_dir, max_iter=max_iter,
                                 force_learn=force_learn, trials=trials)
            rows.append(row)
            if row.get("status") == "ok":
                std = row.get("average_error_std")
                if std is not None:
                    print(f"  avg_err={row['average_error']:.6f} ± {std:.6f} "
                          f"(n={row['num_ok_trials']})", flush=True)
                else:
                    print(f"  avg_err={row['average_error']:.6f}", flush=True)
            else:
                print(f"  status={row['status']}", flush=True)
    return rows

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _cli_learn(args):
    result = learn(args.template, args.data, args.num_samples,
                   output_dir=args.output_dir, max_iter=args.max_iter)
    print(json.dumps({k: result[k] for k in
                      ("model_name", "template_name", "log_likelihood", "bic_score",
                       "elapsed_seconds", "num_hidden_nodes", "learned_model_path")},
                     indent=2))


def _cli_infer(args):
    result = infer(args.true_model, args.learned_model, args.query_var,
                   args.do, args.num_samples, output_dir=args.output_dir)
    print(json.dumps({k: result[k] for k in
                      ("average_error", "weighted_error", "ate_error", "elapsed_seconds")},
                     indent=2))


def _cli_run(args):
    if args.experiments:
        experiments = args.experiments
        exp_tag = "+".join(args.experiments)
    else:
        experiments = EXPERIMENT_SETS.get(args.experiment_set, SMALL_EXPERIMENTS)
        exp_tag = args.experiment_set

    sizes_tag = "+".join(str(s) for s in args.sample_sizes)
    custom_tag = args.name if args.name else "run"
    run_name = f"{exp_tag}_{sizes_tag}_{custom_tag}"

    learned_model_dir = Path(args.learned_model_dir) / run_name
    output_dir = Path(args.output_dir) / run_name

    rows = run_all(
        experiments=experiments,
        sample_sizes=args.sample_sizes,
        domains=args.domains,
        restarts=args.restarts,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        learned_model_dir=learned_model_dir,
        output_dir=output_dir,
        max_iter=args.max_iter,
        force_learn=args.force_learn,
        trials=args.trials,
    )
    # Save summary
    summary_path = output_dir / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    multi_trial = args.trials > 1
    if multi_trial:
        hdr = f"{'Experiment':<25} {'N':>6} {'T':>4} {'avg_err':>10} {'±std':>10} {'w_err':>10} {'ate_err':>10}"
    else:
        hdr = f"{'Experiment':<25} {'N':>6} {'avg_err':>10} {'w_err':>10} {'ate_err':>10}"
    print(f"\n=== SUMMARY ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("status") == "ok":
            if multi_trial:
                print(f"{r['experiment']:<25} {r['num_samples']:>6} {r.get('num_ok_trials', 1):>4} "
                      f"{r['average_error']:>10.6f} {r.get('average_error_std', 0.0):>10.6f} "
                      f"{r['weighted_error']:>10.6f} {r['ate_error']:>10.6f}")
            else:
                print(f"{r['experiment']:<25} {r['num_samples']:>6} "
                      f"{r['average_error']:>10.6f} {r['weighted_error']:>10.6f} {r['ate_error']:>10.6f}")
        else:
            print(f"{r['experiment']:<25} {r['num_samples']:>6}  {r['status']}")
    print(f"\nRun name : {run_name}")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="EM4CI: EM learning + causal inference")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── learn ──
    p_learn = sub.add_parser("learn", help="Fit one EM template model")
    p_learn.add_argument("template", help="Template XDSL (structure + latent markers)")
    p_learn.add_argument("data", help="Observed data CSV")
    p_learn.add_argument("num_samples", type=int, help="Declared sample size (for BIC)")
    p_learn.add_argument("--output-dir", default=None)
    p_learn.add_argument("--max-iter", type=int, default=100)
    p_learn.set_defaults(func=_cli_learn)

    # ── infer ──
    p_infer = sub.add_parser("infer", help="Compute causal error vs true model")
    p_infer.add_argument("true_model", help="Ground-truth XDSL")
    p_infer.add_argument("learned_model", help="Learned XDSL")
    p_infer.add_argument("query_var", help="Query variable Y in P(Y|do(X))")
    p_infer.add_argument("--do", nargs="+", required=True, dest="do", help="Intervention variable(s)")
    p_infer.add_argument("--num-samples", type=int, default=100)
    p_infer.add_argument("--output-dir", default=None)
    p_infer.set_defaults(func=_cli_infer)

    # ── run ──
    p_run = sub.add_parser("run", help="Full pipeline: learn all restarts + infer, for multiple experiments")
    p_run.add_argument("--experiments", nargs="+", default=None,
                       help="Explicit experiment names (overrides --experiment-set)")
    p_run.add_argument("--experiment-set", choices=list(EXPERIMENT_SETS), default="small",
                       help="Predefined set: small (8), large (6), all (14)  [default: small]")
    p_run.add_argument("--sample-sizes", nargs="+", type=int, default=[100, 1000])
    p_run.add_argument("--domains", nargs="+", type=int, default=[2, 4])
    p_run.add_argument("--restarts", type=int, default=10)
    p_run.add_argument("--max-iter", type=int, default=100)
    p_run.add_argument("--model-dir", default="models_xdsl")
    p_run.add_argument("--data-dir", default="data")
    p_run.add_argument("--learned-model-dir", default="learned_models")
    p_run.add_argument("--output-dir", default="output")
    p_run.add_argument("--name", default=None,
                       help="Custom run name used as top-level folder "
                            "(default: experiment-set name, or 'custom' when --experiments is used)")
    p_run.add_argument("--force-learn", action="store_true",
                       help="Re-run learning even if metrics.json already exists")
    p_run.add_argument("--trials", type=int, default=10, metavar="N",
                       help="Independent trials per (experiment, sample_size). "
                            "Each trial uses force_learn=True and its own sub-directory. "
                            "Results are averaged with mean ± std reported. (default: 1)")
    p_run.set_defaults(func=_cli_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
