# %% [markdown]
# ## Improved Knowledge Distillation Experiment
#
# Must be run AFTER SSL_KD.py completes (needs cv_results, helpers, and all imports).
# Paste these cells at the bottom of SSL_KD.py on Kaggle, or run in the same kernel.
#
# Adds two new models on top of SSL_KD.py's three:
#   Model 4 — Improved KD  (T=4, alpha=0.7): large teacher → small student
#   Model 5 — Soft KD      (T=6, alpha=0.5): even softer targets, equal weighting
#
# Teacher flow (fixed):
#   SparK pretrained encoder (32ch) → fine-tune on Heart → freeze → distill to small student
#   The teacher uses the SAME architecture as SSL_KD.py so SSL weights load cleanly.

# %%
import torch.nn.functional as F
from pathlib import Path

# Restore cv_results from WandB if running in a fresh session
# (no-op if SSL_KD.py already ran in the same kernel)
if not _cv_results_path.exists():
    restore_checkpoint("ssl_kd_cv_results.json", Path(OUT_DIR),
                       "ssl-kd-cv-results", WANDB_PROJECT, "", "")
if _cv_results_path.exists() and not cv_results:
    import json as _json
    cv_results.update(_json.loads(_cv_results_path.read_text()))
    print(f"  Restored cv_results: {list(cv_results.keys())}")

# Student is smaller than the base model — teacher stays the same size as SSL_KD.py
# so that the SSL pretrained weights (32ch base) load without any size mismatch.
STUDENT_CFG = {
    "channels": [16, 32, 64, 128, 256],   # 2x smaller than base model
    "strides":  [2, 2, 2, 2],
}

# KD hyperparameter variants — this is what makes each model distinct
KD_VARIANTS = [
    {"temperature": 4.0, "alpha": 0.7, "tag": "kd_T4_a07"},  # Model 4
    {"temperature": 6.0, "alpha": 0.5, "tag": "kd_T6_a05"},  # Model 5
]

# One shared DiceCELoss instance — not recreated per batch
_kd_criterion = DiceCELoss(to_onehot_y=True, softmax=True)


def _kd_loss(student_logits, teacher_logits, labels, alpha, temperature):
    """Combined task + KD loss. Teacher logits must already be detached."""
    task_loss = _kd_criterion(student_logits, labels)

    t_soft  = F.softmax(teacher_logits / temperature, dim=1)
    s_log   = F.log_softmax(student_logits / temperature, dim=1)
    n_vox   = student_logits.shape[2] * student_logits.shape[3] * student_logits.shape[4]
    kd_loss = F.kl_div(s_log, t_soft, reduction="sum") / (student_logits.shape[0] * n_vox)
    kd_loss = kd_loss * (temperature ** 2)

    return (1 - alpha) * task_loss + alpha * kd_loss, task_loss.item(), kd_loss.item()


def _build_student():
    """Small student UNet — 2x fewer channels than base model."""
    unet = build_unet(in_channels=1, out_channels=2,
                      channels=tuple(STUDENT_CFG["channels"]),
                      strides=tuple(STUDENT_CFG["strides"]))
    return UNetWithEncoder(unet).to(DEVICE)


def _get_or_train_teacher(fold_idx: int, train_loader, val_loader):
    """
    Returns a fine-tuned, frozen teacher for this fold.
    Teacher = same architecture as SSL_KD.py base model so SSL weights load cleanly.
    Reuses the ssl_only checkpoint if already trained, otherwise trains from scratch.
    """
    fold_key = f"fold_{fold_idx}"

    # Reuse ssl_only checkpoint — it's already a fine-tuned SSL model on Heart
    ssl_ckpt = cv_results.get(fold_key, {}).get("ssl_only", {}).get("ckpt", "")
    teacher  = build_model(pretrained=False)  # same arch as SSL_KD.py

    if ssl_ckpt and Path(ssl_ckpt).exists():
        teacher.load_state_dict(torch.load(ssl_ckpt, map_location=DEVICE))
        print(f"  Teacher: loaded ssl_only checkpoint (fold {fold_idx+1})")
    else:
        # ssl_only not available — train teacher with SSL init from scratch
        print(f"  Teacher: ssl_only checkpoint missing, training from SSL init...")
        teacher = build_model(pretrained=True)
        result  = finetune(f"teacher_fold{fold_idx+1}", teacher, train_loader, val_loader)
        teacher.load_state_dict(torch.load(result["ckpt"], map_location=DEVICE))
        print(f"  Teacher trained: DSC={result['best_dsc']:.4f}")

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,} (all frozen)")
    return teacher


def run_kd_variant(fold_idx: int, variant: dict, train_loader, val_loader) -> dict:
    """
    Train one KD variant (defined by temperature + alpha) for a given fold.
    Skips if already in cv_results.
    """
    fold_key = f"fold_{fold_idx}"
    tag      = variant["tag"]
    T        = variant["temperature"]
    alpha    = variant["alpha"]
    run_name = f"{tag}_fold{fold_idx+1}"

    if cv_results.get(fold_key, {}).get(tag):
        print(f"  {tag} fold {fold_idx+1} already done — skipping.")
        return cv_results[fold_key][tag]

    print(f"\n=== Fold {fold_idx+1} — {tag} (T={T}, α={alpha}) ===")

    teacher = _get_or_train_teacher(fold_idx, train_loader, val_loader)
    student = _build_student()
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    ckpt_dir = Path(OUT_DIR) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = ckpt_dir / "latest.pth"

    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=TRAIN_CFG["lr"],
                                  weight_decay=TRAIN_CFG["weight_decay"])
    scheduler = make_scheduler(optimizer, TRAIN_CFG["epochs"], TRAIN_CFG["warmup_epochs"])
    scaler    = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    best_dsc, best_hd95, trigger, start_epoch = 0.0, float("inf"), 0, 0

    # Resume if interrupted
    if resume_ckpt.exists():
        state       = torch.load(resume_ckpt, map_location=DEVICE)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_dsc    = state["best_dsc"]
        best_hd95   = state["best_hd95"]
        trigger     = state["trigger"]
        print(f"  Resumed from epoch {start_epoch}, best_dsc={best_dsc:.4f}")

    for epoch in range(start_epoch, TRAIN_CFG["epochs"]):
        student.train()
        epoch_loss, epoch_task, epoch_kd, n_batches = 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]

            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            if labels.dim() == 4:
                labels = labels.unsqueeze(1)
            labels = labels.long()

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                s_logits = student(imgs)
                with torch.no_grad():
                    t_logits = teacher(imgs)
                loss, tl, kl = _kd_loss(s_logits, t_logits.detach(), labels, alpha, T)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_task += tl
            epoch_kd   += kl
            n_batches  += 1

        scheduler.step()
        nb = max(n_batches, 1)
        metrics  = evaluate(student, val_loader)
        dsc, hd95 = metrics["dice"], metrics["hd95"]

        print(f"  [{run_name}] Epoch {epoch+1:>3}/{TRAIN_CFG['epochs']} | "
              f"loss={epoch_loss/nb:.4f} | task={epoch_task/nb:.4f} | "
              f"kd={epoch_kd/nb:.4f} | DSC={dsc:.4f} | HD95={hd95:.1f} | "
              f"best={best_dsc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        log_wandb({f"{run_name}/loss": epoch_loss/nb, f"{run_name}/task_loss": epoch_task/nb,
                   f"{run_name}/kd_loss": epoch_kd/nb, f"{run_name}/dsc": dsc,
                   f"{run_name}/hd95": hd95, f"{run_name}/epoch": epoch + 1})

        if dsc >= best_dsc:
            best_dsc, best_hd95 = dsc, hd95
            torch.save(student.state_dict(), ckpt_dir / "best.pth")
            trigger = 0
        else:
            trigger += 1
            if trigger >= TRAIN_CFG["patience"]:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        # Save resume checkpoint
        torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_dsc": best_dsc, "best_hd95": best_hd95,
                    "trigger": trigger}, resume_ckpt)

    log_wandb({f"{run_name}/best_dsc": best_dsc, f"{run_name}/best_hd95": best_hd95})
    print(f"  [{run_name}] Done. Best DSC={best_dsc:.4f}  HD95={best_hd95:.1f}")

    result = {"run": run_name, "best_dsc": best_dsc, "best_hd95": best_hd95,
              "ckpt": str(ckpt_dir / "best.pth"), "temperature": T, "alpha": alpha}
    cv_results.setdefault(fold_key, {})[tag] = result
    _save_cv_results()
    # Push updated cv_results to WandB so next session can restore and skip
    save_checkpoint(_cv_results_path, "ssl-kd-cv-results", "", "")
    return result


# %%
# Run both KD variants across all folds
print("\n" + "="*70)
print("IMPROVED KD VARIANTS (Models 4 & 5)")
print("="*70)

for fold_idx in range(N_FOLDS):
    train_loader, val_loader = get_fold_loaders(fold_idx, use_all_train=False)
    for variant in KD_VARIANTS:
        if USE_WANDB:
            wandb.init(project=WANDB_PROJECT,
                       name=f"{variant['tag']}_fold{fold_idx+1}",
                       reinit=True,
                       config={**TRAIN_CFG, **variant,
                               "fold": fold_idx+1,
                               "student_channels": STUDENT_CFG["channels"]})
        try:
            run_kd_variant(fold_idx, variant, train_loader, val_loader)
        finally:
            if USE_WANDB:
                wandb.finish()

# %%
# Final 5-model comparison table (mean ± std across 3 folds)
print("\n" + "="*70)
print("FULL RESULTS — All 5 Models (3-fold CV)")
print("="*70)

all_methods = {
    "baseline":   "1. Baseline (random init)",
    "ssl_only":   "2. SSL-only (SparK)",
    "ssl_kd":     "3. SSL+KD  (T=2, α=1.0)",
    "kd_T4_a07":  "4. SSL+KD  (T=4, α=0.7)",
    "kd_T6_a05":  "5. SSL+KD  (T=6, α=0.5)",
    "upper_bound":"6. Supervised upper bound",
}

print(f"  {'Model':<35} {'DSC mean±std':<18} {'HD95 mean±std'}")
print("-" * 70)

for key, label in all_methods.items():
    dscs  = [cv_results[f"fold_{f}"][key]["best_dsc"]
             for f in range(N_FOLDS)
             if key in cv_results.get(f"fold_{f}", {})]
    hd95s = [cv_results[f"fold_{f}"][key]["best_hd95"]
             for f in range(N_FOLDS)
             if key in cv_results.get(f"fold_{f}", {})]
    if dscs:
        dsc_str  = f"{np.mean(dscs):.3f} ± {np.std(dscs, ddof=1) if len(dscs)>1 else 0:.3f}"
        hd95_str = f"{np.mean(hd95s):.1f} ± {np.std(hd95s, ddof=1) if len(hd95s)>1 else 0:.1f}"
    else:
        dsc_str = hd95_str = "N/A"
    print(f"  {label:<35} {dsc_str:<18} {hd95_str}")
