"""Model construction, grid search on validation and the final test run."""

from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models import EASE, IALS, CustomALS, FactorizationMachine, ItemKNN, Popularity, ScoreFusion

from .data import Phase
from .evaluation import EvaluationSet, evaluate, paired_bootstrap

MAIN_MODELS = ["ItemKNN", "IALS", "EASE", "FM", "CustomALS"]
FM_FEATURES = {"age": ("age",), "all": ("total", "recent", "age")}


def grid(space: dict) -> list[dict]:
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def fit(name: str, params: dict, phase: Phase, seed: int = 42, on_epoch=None):
    """Train one model on the phase history; every model sees the same history."""
    p = dict(params)
    observed, positive = phase.matrix("observed"), phase.matrix("positive")
    if name.startswith("Popularity"):
        return Popularity().fit(phase.popularity(p["window_days"]), phase.n_users)
    if name in ("ItemKNN", "EASE"):
        signal = phase.matrix(p.pop("signal"))
        return (ItemKNN if name == "ItemKNN" else EASE)(**p).fit(signal)
    if name == "IALS":
        return IALS(**p, seed=seed).fit(observed, positive)
    if name.startswith("CustomALS"):
        targets = CustomALS.build_targets(
            phase.matrix("rating"), phase.matrix("time"), phase.cutoff,
            half_life_days=p.pop("half_life_days"), recency_mix=p.pop("recency_mix"))
        return CustomALS(**p, seed=seed).fit(observed, targets)
    if name.startswith("FM"):
        fields = p.pop("item_features")
        features = None if fields is None else phase.item_history(FM_FEATURES[fields])
        return FactorizationMachine(**p, seed=seed).fit(
            observed, positive, features, phase.matrix("time"), phase.cutoff, on_epoch)
    raise ValueError(name)


def _row(name, params, result, seconds, **extra):
    return {"model": name, **extra, **result.metrics, "fit_seconds": seconds, "params": params}


def _best(rows: list[dict], metric: str) -> dict:
    return min(rows, key=lambda r: (-r[metric], json.dumps(r["params"], sort_keys=True)))


_WORKER: dict = {}


def _init_worker(phase: Phase, config: dict) -> None:
    _WORKER.update(phase=phase, data=EvaluationSet.from_phase(phase, config["k"]), seed=config["seed"],
                   limits=threadpool_limits(config["threads"]))


def _run_trial(task) -> list[dict]:
    """Fit one configuration; FM is scored after each epoch count in ``checkpoints``."""
    name, params, checkpoints = task
    phase, data, rows = _WORKER["phase"], _WORKER["data"], []
    start = time.perf_counter()

    def record(model, trial):
        seconds = time.perf_counter() - start
        rows.append({**_row(name, trial, evaluate(model, data), seconds),
                     "loss_history": list(getattr(model, "loss_history_", []))})

    if checkpoints:
        def on_epoch(epoch, model):
            if epoch in checkpoints:
                record(model, {**params, "epochs": epoch})
        fit(name, {**params, "epochs": max(checkpoints)}, phase, _WORKER["seed"], on_epoch)
    else:
        record(fit(name, params, phase, _WORKER["seed"]), params)
    return rows


def run_validation(phase: Phase, config: dict, out: Path) -> tuple[pd.DataFrame, dict]:
    """Evaluate every grid point on validation in parallel; save trials and the selection.

    Every trial is seeded independently, so results do not depend on the number of workers.
    """
    tasks = []
    for name, space in config["search"].items():
        if name == "FM":  # one run per setting, scored at each epoch count of the grid
            rest = {key: values for key, values in space.items() if key != "epochs"}
            tasks += [(name, params, set(space["epochs"])) for params in grid(rest)]
        else:
            tasks += [(name, params, None) for params in grid(space)]
    order = sorted(range(len(tasks)), key=lambda j: tasks[j][0] != "FM")  # long FM runs first
    with ProcessPoolExecutor(config["workers"], initializer=_init_worker, initargs=(phase, config)) as pool:
        done = dict(zip(order, pool.map(_run_trial, [tasks[j] for j in order])))
    rows = [row for j in range(len(tasks)) for row in done[j]]
    metric = config["primary_metric"]
    selected = {name: _best([r for r in rows if r["model"] == name], metric) for name in config["search"]}

    # Fusion of the two best main models; the weight is chosen on validation too.
    data = EvaluationSet.from_phase(phase, config["k"])
    first, second = sorted(MAIN_MODELS, key=lambda n: -selected[n][metric])[:2]
    with threadpool_limits(config["threads"]):
        parts = [fit(n, selected[n]["params"], phase, config["seed"]) for n in (first, second)]
        for weight in config["fusion_weights"]:
            params = {"models": [first, second], "weight": weight}
            fused = ScoreFusion(parts, [weight, 1 - weight])
            rows.append({**_row("Fusion", params, evaluate(fused, data), 0.0), "loss_history": []})
    selected["Fusion"] = _best([r for r in rows if r["model"] == "Fusion"], metric)

    trials = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    trials.drop(columns="loss_history").assign(params=trials.params.map(json.dumps)).to_csv(
        out / "validation_trials.csv", index=False)
    selection = {name: {"params": r["params"], metric: r[metric], "recall@10": r["recall@10"]}
                 for name, r in selected.items()}
    (out / "selected.json").write_text(json.dumps(selection, indent=2) + "\n")
    return trials, selection


def _test_specs(selected: dict, config: dict) -> list[tuple[str, dict, int, str]]:
    seed = config["seed"]
    custom, ials = selected["CustomALS"]["params"], selected["IALS"]["params"]
    specs = [(name, selected[name]["params"], seed, "main") for name in MAIN_MODELS]
    specs += [
        ("Popularity", selected["Popularity"]["params"], seed, "baseline"),
        ("Popularity_all_time", {"window_days": None}, seed, "baseline"),
        ("Fusion", selected["Fusion"]["params"], seed, "baseline"),
        ("FM_ID_only", {**selected["FM"]["params"], "item_features": None}, seed, "control"),
        ("CustomALS_no_time", {**custom, "recency_mix": 0.0}, seed, "control"),
        ("CustomALS_matched", {**ials, "half_life_days": custom["half_life_days"],
                               "recency_mix": custom["recency_mix"]}, seed, "control"),
    ]
    for repeat in config["repeat_seeds"]:
        specs += [(name, selected[name]["params"], repeat, "repeat") for name in ("IALS", "CustomALS", "FM")]
        specs.append(("FM_ID_only", {**selected["FM"]["params"], "item_features": None}, repeat, "repeat"))
    return specs


def run_test(phase: Phase, selection_path: Path, config: dict, out: Path):
    """Refit the saved selection on the test-phase history and evaluate once on test."""
    selected = json.loads(Path(selection_path).read_text())
    data = EvaluationSet.from_phase(phase, config["k"])
    rows, results, models = [], {}, {}
    with threadpool_limits(config["threads"]):
        for name, params, seed, role in _test_specs(selected, config):
            start = time.perf_counter()
            if name == "Fusion":
                model = ScoreFusion([models[n] for n in params["models"]],
                                    [params["weight"], 1 - params["weight"]])
                seconds = sum(r["fit_seconds"] for r in rows if r["model"] in params["models"])
            else:
                model = fit(name, params, phase, seed)
                seconds = time.perf_counter() - start
            result = evaluate(model, data)
            if seed == config["seed"]:
                results[name] = result
                if role == "main":
                    models[name] = model
            rows.append(_row(name, params, result, seconds, role=role, seed=seed))

    best = max(MAIN_MODELS, key=lambda n: selected[n][config["primary_metric"]])
    fusion_best = selected["Fusion"]["params"]["models"][0]
    pairs = [("CustomALS", "IALS"), ("CustomALS_matched", "IALS"), ("CustomALS", "CustomALS_no_time"),
             ("FM", "FM_ID_only"), ("Fusion", fusion_best), (best, "Popularity")]
    comparisons = pd.DataFrame([
        {"a": a, "b": b, "metric": metric,
         **paired_bootstrap(results[a].per_user[metric], results[b].per_user[metric],
                            config["bootstrap_repeats"], config["seed"])}
        for a, b in pairs for metric in ("ndcg@10", "recall@10")])
    table = pd.DataFrame(rows)
    table.assign(params=table.params.map(json.dumps)).to_csv(out / "test_results.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    return table, comparisons


FIXED = {"n_iters", "learning_rate", "negative_ratio", "batch_size", "num_threads", "weight_decay"}
SHORT = {"n_factors": "f", "factors": "f", "alpha": "α", "l2": "λ", "half_life_days": "h", "recency_mix": "η",
         "shrinkage": "γ", "window_days": "окно", "item_features": "признаки", "signal": "вход"}


def describe(params: dict) -> str:
    """Compact human-readable parameters for tables, e.g. 'f=64, α=10, λ=30'."""
    if "models" in params:
        a, b = params["models"]
        return f"{params['weight']:g}·{a} + {1 - params['weight']:g}·{b}"
    none = {"item_features": "нет", "window_days": "всё время"}
    return ", ".join(f"{SHORT.get(k, k)}={none[k] if v is None else v}"
                     for k, v in params.items() if k not in FIXED)
