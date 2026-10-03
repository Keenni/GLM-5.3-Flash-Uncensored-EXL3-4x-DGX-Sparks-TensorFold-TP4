# Rebuilding the stack on TensorFold v0.6.2 (Mia's 53 patches, rebased)

**One-line conclusion:** TensorFold v0.6.1/0.6.2 overhauled the engine; Mia's 53 patches no longer applied.
We rebased all of them onto a clean v0.6.2 tree, ported the TP4 world-generalization into it, and burned
the result into a local image — no runtime overlay mounting anymore.

- Upstream base: TensorFold **v0.6.2** (the v0.6.0 tree in `baseline/` stays verbatim for the overlay route).
- Result: 51 of Mia's 53 patches applied (see triage below), 68 files touched, `+13,842 / −604` lines,
  all static gates green, and a live four-rank stack as the final arbiter.
- Image chain: `tensorfold-glm53:v0.6.0` (old production) → `v0.6.2-tp4` (rebase base) →
  `v0.6.2-tp4f` (drowzeys shared-EXL3 layer port) → **`v0.6.2-tp4j`** (current, see
  `scripts/start-glm53-flash-tf-unc-tp4-v062.sh`).

## 1. How the rebase was done

Four rounds of work, each round committing to disk and writing an incremental report, because the per-round
work exceeded any single session:

1. **Per-patch triage against a clean v0.6.2 tree** — apply each of Mia's 53 patches in isolation
   (`git apply -p1`, falling back to direct edits + regenerating the diff when the toolchain lied), and
   classify:
   - **11 file-level moves**: files the upstream had relocated or reorganized (`nofile` class).
   - **29 partially conflicting**, **11 fully conflicting** patches — roughly 124 conflicting hunks total,
     because v0.6.1 was described upstream as "engine work" and reshaped exactly the files Mia patches.
   - **Patches upstream absorbed** — the important class. Four of Mia's patches (0042 prompt-replay,
     0024 select-rows, 0052 roce-startup, and 0023 effort-max) have equivalent implementations in
     v0.6.2 already; hard-applying them would *revert* upstream improvements. Evidence for each ruling is
     in the report we kept; **0023 is dropped outright, 0051 is folded into 0053**.
2. **`git merge-file --diff3` as the primary tool**, not hand-reconciliation: for every file the overlay
   and patches touch, `ours` = the production tree (v0.6.0 + Mia 53 + TP4), `base` = pristine v0.6.0,
   `theirs` = pristine v0.6.2. The merge result is the correct "v0.6.2 + 53 + TP4" state; diffing it
   against our tree exposes every porting omission mechanically.
3. **Whole-file overwrite for untouched files**: for the ~7 files upstream did *not* change between
   v0.6.0 and v0.6.2 (`sparse.py`, the kda/decode/latent family, …), the correctly-patched v0.6.0 file
   can be copied in whole. Only files upstream touched (notably `engine.py`) need hunk-level work.
4. **A 22-hunk sweep of rename collisions** — upstream refactored `exl3_mm → exl3_generic`,
   `Scratch → Exl3Scratch`, etc.; naive merges left stale references.

One patch was worth a rewrite rather than a merge: **0049 multi-prefill**. Upstream's v0.6.2 engine/lane
restructuring invalidated its patch surface, so it was re-implemented as a new ~185-line
`multi_prefill.py` module against the new structure.

## 2. Verification: `py_compile` passing means nothing

The first "done" report claimed 52/52 files compile — and the stack then failed to boot three times in a
row, each time on a class of breakage compilation cannot see:

1. `NameError: shared_prefix` — an entire function-definition block lost, its call site kept.
2. `NameError: kv` — a variable block lost inside a patched function.
3. `AttributeError: self.split` — a *use* landed without its *assignment*; even pyflakes misses this.
4. `sparse.py` partially applied — duplicate definitions plus a dangling truncated block at module level.

The gates we now run before any image build (all cheap, all text-level):

- **pyflakes over the whole tree: zero undefined names** (catches 1 and 2).
- **Attribute/import gap gates**: compare the set of `self.X =` assignments and imported names per file
  against a reference tree — but *manually verify every hit*, because upstream renames produce false
  positives (catches 3, with review).
- **21 environment-variable gates must all be read** — if a knob a patch introduces is silently ignored,
  the patch did not land. `env | grep TF_GLM` matching the expected set is a build artifact check.
- **A real boot with the smoke test** — the only final truth. Static gates reduce iterations; they do not
  replace the boot.

## 3. The incremental-image trick

Rebuilding a 36.9 GB base image per iteration is a non-starter. While iterating we built a tiny image on
each node:

```dockerfile
FROM tensorfold-glm53:v0.6.2-tp4
COPY <the 2 files that changed> /usr/local/lib/python3.12/dist-packages/tensorfold/...
RUN find ... -name '__pycache__' -exec rm -rf {} +
```

Context of 31 KB → a few MB; each of the four nodes rebuilds in seconds. Only once the stack was final did
we build the authoritative full image (`v0.6.2-tp4j`, each node builds locally — the layer chain is shared,
only the label was pruned).

## 4. Porting the TP4 W4 generalization into the new tree

The nine-file world-generalization from our
[overlay route](tp4-world-generalization.md) was re-applied semantically to the rebased tree — including
the per-follower doorbell keys (adapted to the new lane mechanism) and the HC-split gather-mode code. Two
classes of porting bugs bit us here, both documented in detail in
[cache-eviction.md](cache-eviction.md) (the KV-pool halving) and [prefill-tuning.md](prefill-tuning.md)
(the HC-split saga):

> **The authoritative reference for porting is the production tree, not a self-made "golden" copy.**
> Our first comparison baseline was an incomplete snapshot (missing 13 files, still semantically TP2);
> using it for gap analysis silently hides omissions. `git merge-file --diff3` against the *actual running*
> tree is the systematic check.

The typical omission shape is **a parameter silently dropped**, which is exactly why it passes every
compile/attribute gate:

- `mla_geometry(..., kv=kv)` losing `kv=` ⇒ estimation prices the KV cache at bf16 while runtime runs fp8
  ⇒ per-token budget doubled, **pool halved** (8.08M → 4.12M tokens).
- A lost `+ mla_ring_bytes(...)` term in the workspace estimate.
- A lost `self.opener = ...` line — harmless at startup, `AttributeError` on the **first real request**.

Acceptance anchor: the boot log's `startup estimate X GiB within Y GiB` and `pool of N tokens` lines.
If the pool is roughly halved or the node retains tens of GiB after start, geometry was dropped.

## 5. Where things stand

The v0.6.2 line is the current production route; see `scripts/start-glm53-flash-tf-unc-tp4-v062.sh` for
the full knob set and [prefill-tuning.md](prefill-tuning.md) for what is enabled, verified, and why. The
v0.6.0 overlay route stays in this repository (`overlay/`) for reference — it is how this project started,
and its pitfall dossier is still accurate for that tree.
