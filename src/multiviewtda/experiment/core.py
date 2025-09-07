# generic ExperimentRunner
#
# src/yourpkg/experiment/core.py
from __future__ import annotations
import itertools, json, time, numpy as np
from pathlib import Path
from typing import Iterable
from polystack import Polystack, oof_by_view
from .config import ExperimentConfig
from .registry import create_dataset, get_view
from sklearn.model_selection import train_test_split

class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def _generate_combos(self, views: list[str]) -> list[tuple[str, ...]]:
        base_required = set(self.cfg.views.always_include)  # e.g., ["raw"]
        pool = [v for v in views if v not in base_required]
        # powerset limited to max_views / exact count, then sample 32 if > limit
        all_subsets = []
        for r in range(self.cfg.views.min_views, self.cfg.views.max_views + 1):
            for subset in itertools.combinations(pool, r - len(base_required)):
                all_subsets.append(tuple(sorted(base_required.union(subset))))
        rng = np.random.RandomState(self.cfg.random_seed)
        if len(all_subsets) > self.cfg.views.limit:
            all_subsets = list(rng.choice(all_subsets, size=self.cfg.views.limit, replace=False))
        return sorted(set(all_subsets))

    def run(self) -> dict:
        ds = create_dataset(self.cfg.dataset.name, **self.cfg.dataset.params)
        X, y = ds.load("train")

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=self.cfg.data.test_size,
            stratify=y if ds.task == "classification" else None,
            random_state=self.cfg.random_seed
        )

        # compute views once
        computed: dict[str, np.ndarray] = {}
        for v in self.cfg.views.available:  # e.g., ["raw","quadrants","tda"]
            V = get_view(v)
            out = V(Xtr)
            if isinstance(out, dict):  # quadrants -> expand names
                for k, arr in out.items(): computed[k] = arr
            else:
                computed[v] = out
        # mirror for test (reuse pipelines where possible; left as exercise)

        combos = self._generate_combos(list(computed.keys()))

        results = []
        for seed in self.cfg.seeds:
            for combo in combos:
                X_by_view = {k: computed[k] for k in combo}
                ests = self.cfg.models.make_base_estimators_for(combo)   # see §6
                final = self.cfg.models.make_final_estimator()
                ps = Polystack(
                    estimators=ests or None,
                    final_estimator=final,
                    cv=self.cfg.cv.to_sklearn(ds.task),
                    random_state=seed,
                    n_jobs=self.cfg.n_jobs,
                    meta=self.cfg.meta.name,
                ).fit(X_by_view, ytr)
                acc = float((ps.predict(X_by_view) == ytr).mean())
                results.append({"seed": seed, "views": combo, "acc_train": acc})

        out_dir = Path(self.cfg.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        (out_dir / "config_used.yaml").write_text(self.cfg.source_yaml)  # keep the exact cfg
        return {"n_runs": len(results), "out": str(out_dir)}
