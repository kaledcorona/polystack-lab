from __future__ import annotations
import argparse, json
from polystack import Polystack
from multiviewtda.experiment.views.registry import create_view
from multiviewtda.experiments.views.noise import apply_noise



specs = [
  {"name": "raw", "params": {"reshape_mode": "pad"}},
  {"name": "grid", "params": {"rows": 3, "cols": 3, "grid_mode": "pad", "prefix": "g3_"}},
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--meta", default="concat_proba")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--out", default="experiments/runs/latest.json")
    args = p.parse_args()

    ds = create_dataset(args.dataset)
    X_by_view, y = ds.load("train")

    clf = Polystack(
        estimators=None,            # default RF per view, or pass dict/list
        final_estimator=None,       # default RF
        cv=args.cv,
        random_state=42,
        meta=args.meta,
    ).fit(X_by_view, y)

    metrics = {"dataset": ds.name, "views": list(X_by_view), "classes": getattr(clf, "classes_", None)}
    print(json.dumps(metrics, indent=2))

def materialize_views(X_flat, specs):
    out = {}
    for s in specs:
        if isinstance(s, str):
            fn = create_view(s)
        else:
            fn = create_view(s["name"], **s.get("params", {}))
        produced = fn(X_flat)
        # avoid name collisions
        for k in produced:
            if k in out:
                raise ValueError(f"Duplicate view name '{k}'. Consider using 'prefix'.")
        out.update(produced)
    return out


if cfg.data.noise.active:
    X_train = apply_noise(
        X_train, cfg.data.noise.name,
        rng=np.random.default_rng(cfg.random_seed),
        image_shape=(28, 28),
        **cfg.data.noise.get("params", {})
    )


if __name__ == "__main__":
    main()



