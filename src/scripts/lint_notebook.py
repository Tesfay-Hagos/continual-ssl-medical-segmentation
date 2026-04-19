#!/usr/bin/env python3
"""
Dry-run checker for kaggle_run.py.
Usage:  python scripts/lint_notebook.py src/notebooks/kaggle_run.py
"""

import ast
import importlib.util
import re
import sys
import os

NB_FILE = sys.argv[1] if len(sys.argv) > 1 else "src/notebooks/kaggle_run.py"

print(f"Checking: {NB_FILE}\n")
passed = True

with open(NB_FILE) as f:
    src = f.read()

# ── 1. Syntax ──────────────────────────────────────────────────────────────────
try:
    tree = ast.parse(src)
    print("✅  Syntax OK")
except SyntaxError as e:
    print(f"❌  SyntaxError line {e.lineno}: {e.msg}")
    sys.exit(1)

# ── 2. Cell count ─────────────────────────────────────────────────────────────
cells = [l for l in src.splitlines() if l.startswith("# %%")]
code  = [l for l in cells if "[markdown]" not in l]
md    = [l for l in cells if "[markdown]"     in l]
print(f"✅  {len(cells)} cells ({len(code)} code, {len(md)} markdown)")

# ── 3. Variable-before-assignment check (known patterns) ──────────────────────
class VarChecker(ast.NodeVisitor):
    def __init__(self, watch):
        self.watch   = set(watch)
        self.assigns = {}    # name → first assignment line
        self.issues  = []

    def visit_Assign(self, node):
        for t in node.targets:
            # Simple name:  x = ...
            if isinstance(t, ast.Name) and t.id in self.watch:
                self.assigns.setdefault(t.id, node.lineno)
            # Tuple unpack: a, b = ...
            elif isinstance(t, ast.Tuple):
                for elt in t.elts:
                    if isinstance(elt, ast.Name) and elt.id in self.watch:
                        self.assigns.setdefault(elt.id, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Load)
                and node.id in self.watch):
            first = self.assigns.get(node.id, None)
            if first is None or node.lineno < first:
                self.issues.append((node.id, node.lineno))
        self.generic_visit(node)

checker = VarChecker(["bwt_vals", "mt_aa", "all_val"])
checker.visit(tree)
if checker.issues:
    for name, line in checker.issues:
        print(f"⚠️   Line {line}: '{name}' used before assignment")
    passed = False
else:
    print("✅  Variable-before-assignment: OK")

# ── 4. strides vs channels depth consistency ──────────────────────────────────
stride_re  = re.compile(r'strides=\(([^)]+)\)')
channel_re = re.compile(r'channels=\(([^)]+)\)')
lines = src.splitlines()
problems = []
for i, line in enumerate(lines, 1):
    sm = stride_re.search(line)
    cm = channel_re.search(line)
    if sm and cm:
        ns = len(sm.group(1).split(","))
        nc = len(cm.group(1).split(","))
        if ns != nc - 1:
            problems.append(f"  Line {i}: channels depth={nc} needs strides={nc-1}, got {ns}")
if problems:
    for p in problems: print(f"❌  {p}")
    passed = False
else:
    print("✅  strides/channels depth: OK")

# ── 5. src module availability ────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(NB_FILE), "..", "..")
SRC_DIR = os.path.normpath(os.path.join(SRC_DIR, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODULES = [
    "data.datasets", "models.unet",
    "pretraining.pretrain", "pretraining.spark",
    "continual.ewc", "continual.lwf", "continual.replay",
    "evaluation.metrics", "scripts.train_continual",
]
missing_deps = []
for mod in MODULES:
    top = mod.split(".")[0]
    spec = importlib.util.find_spec(top) if top not in ("torch", "monai") else None
    if spec is not None:
        print(f"✅  import {mod}")
    else:
        # Check if the file exists even if deps aren't installed
        rel = mod.replace(".", os.sep) + ".py"
        full = os.path.join(SRC_DIR, rel)
        if os.path.exists(full):
            print(f"🔵  {mod} — file exists, deps need Kaggle/GPU env")
        else:
            print(f"❌  {mod} — FILE NOT FOUND: {full}")
            missing_deps.append(mod)
            passed = False

# ── 6. Summary ────────────────────────────────────────────────────────────────
print()
if passed:
    print("✅  All checks passed. Ready to convert with: make notebook")
else:
    print("❌  Some checks failed — fix before converting.")
    sys.exit(1)
