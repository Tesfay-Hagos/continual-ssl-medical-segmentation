"""
Restore cv_results dict from WandB run summaries.

Imported at runtime from the git-cloned repo, so changes here take effect
after a git pull without needing to rebuild/re-upload the Kaggle notebook.
"""
import traceback
from pathlib import Path

_DONE_STATES = {"finished", "crashed", "killed", "failed"}
_CV_METHODS  = ["baseline", "ssl_only", "ssl_kd", "upper_bound"]


def restore_cv_from_wandb(cv_results: dict, wandb_project: str,
                           n_folds: int, best_ckpt_name: str,
                           out_dir: str) -> None:
    """
    Scan all WandB runs in wandb_project and populate cv_results in-place.

    Accepts runs in any terminal state (finished/crashed/killed/failed) because
    Kaggle sessions are killed before wandb.finish() is called, leaving runs as
    'crashed' even when training completed successfully.
    """
    try:
        import wandb
    except ImportError:
        print("  WandB not installed — skipping run restore.")
        return

    try:
        api    = wandb.Api()
        entity = (wandb.run.entity if wandb.run is not None
                  else api.default_entity or "")
        path   = f"{entity}/{wandb_project}" if entity else wandb_project
        print(f"  Querying WandB project: {path}")

        all_runs = list(api.runs(path))
        print(f"  Total runs found: {len(all_runs)}")

        # Index by display name; keep most recent if duplicates
        runs: dict = {}
        for r in all_runs:
            if r.name not in runs or r.createdAt > runs[r.name].createdAt:
                runs[r.name] = r
        print(f"  Run names: {sorted(runs)}")

        restored = 0
        for fold_idx in range(n_folds):
            fold_key = f"fold_{fold_idx}"
            for method in _CV_METHODS:
                if cv_results.get(fold_key, {}).get(method):
                    continue
                run_name = f"{method}_fold{fold_idx + 1}"
                run = runs.get(run_name)
                if run is None or run.state not in _DONE_STATES:
                    continue
                best_dsc  = run.summary.get(f"{run_name}/best_dsc")
                best_hd95 = run.summary.get(f"{run_name}/best_hd95")
                if best_dsc is None:
                    continue
                cv_results.setdefault(fold_key, {})[method] = {
                    "run":       run_name,
                    "best_dsc":  float(best_dsc),
                    "best_hd95": float(best_hd95) if best_hd95 is not None
                                 else float("inf"),
                    "ckpt":      str(Path(out_dir) / run_name / best_ckpt_name),
                }
                print(f"  ✅ {run_name} [{run.state}]: DSC={best_dsc:.4f}")
                restored += 1

        print(f"  Restored {restored} experiment(s) from WandB.")
    except Exception as e:
        print(f"  ⚠️  WandB run restore failed: {e}")
        traceback.print_exc()
