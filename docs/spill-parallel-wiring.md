# The spill tier never wrote a byte in `--parallel` mode — and how it was wired

**One-line conclusion:** upstream #423's spill tier was only wired into the single-stream engine path. Under
`--parallel N` (which is how both our TP2 and TP4 stacks run), every eviction silently discarded the kept
prompt's rows: the `--spill-gib 240` directories stayed at **0 bytes for 30 hours** while long conversations
kept falling back to full re-prefills of **27–211 s**. The fix wires the multi-stream scheduler into the
spill writer and adds a disk-hit resume path, in two runtime-overlay files.

## Symptoms (observable)

- A conversation idle for 30–60 minutes resumes with `cached=0` and a full prefill; on our TP2 stack
  (2×DGX Spark, `--parallel 12`), 15 of 779 requests in a 10-hour window were full prefills, all of them
  large prompts (42k–220k tokens) costing **27.2–211.5 s** of prefill each.
- The spill directory (`--snapshot-dir`) exists and is mounted, but is empty:
  `ls /root/glm53-spill-*/` → nothing, ever. Not a malfunction of the disk; the write path is dead.
- Memory-side LRU eviction works fine (`TF_GLM_CACHE_ENTRIES=128` cap + `_evict why=room/takeover/home`
  traces) — the states just vanish when evicted instead of landing on disk.

## Root cause (with source anchors)

TensorFold v0.6.7 (`families/glm5_next/cuda/`):

1. **`engine.py`** — the spill hook exists only in the single-stream path:
   `engine._drop()` → `if self._spill is not None and self.spill_bytes > 0 and snap.rows is not None:
   self._spill.enqueue(snap)`. Disk reads (`_disk_resume` → `spill.best`/`spill.load`) are likewise only
   consulted by the single-stream `_resume`. Under `--parallel > 1` requests go through
   **`multi.py`'s `MultiDecoder`**, which has its own kept-prompt bookkeeping — and **`multi.py` contains
   not a single reference to `spill`** (verified by grep in the shipped image).
2. **Different resume vocabularies.** The single-stream engine saves/loads `snap.rows` (attention-row
   tensors extracted from the live caches). The multi-stream scheduler keeps states in pool extents and its
   snapshots carry `rec/conv/tail/head/drafter_rows` instead — `put_ring_tail(st, hit)` and
   `put_draft_window(...)` need `tail`/`head`, **which `spill.write()`/`spill.load()` never serialized**.
   So even a naive "enqueue in multi._drop" would produce files the multi path cannot resume from.
3. Shutdown-time spilling (`SPILL_STOP` → `save_sessions`) only fires on a *clean* shutdown; production
   restarts are `docker rm -f`, so nothing was ever written that way either.

## The fix (two runtime-overlay files, both stacks)

Deployed as a read-only overlay mounted over
`$TF_PKG/families/glm5_next/cuda/{multi.py,spill.py}` — same files for the TP2 (world=2) and TP4 (world=4)
stacks, md5 `921bbe7eb733353a424b61e79ff67e0c` / `b3dd5dc8c90fcec06c5d4e49c5a860e7`
(see [`overlay/`](../overlay/) resp. [`overlay-v067/`](../overlay-v067/), manifest included).

### `spill.py` — serialize the missing fields

- `SpillJob` additionally captures `snap.tail` (indexer-ring tail: `(slots, rows)` pairs) and
  `snap.head` (the logits row a replay of the whole prompt samples from).
- `write()` packs `tail.{i}.slots`, `tail.{i}.rows` and `head`; the metadata `state` records the tail count.
- `load()` restores them; **absent fields (files written by the old writer) stay `None`** — backward
  compatible with the single-stream engine's on-disk format.

### `multi.py` — wire the scheduler in

- `_drop(c)` now calls `_spill_snap(c)` **before** `_settle()` may shrink the extent: the prompt's rows are
  cloned out of the arena via a throwaway `State` bound to the extent's base (`_row_views(st, n, 0)`),
  wrapped in try/except — *a failed spill costs a later prefill, never the request*.
- `_resume()` consults the disk tier on a memory miss: longest strict prefix match via `spill.best()`
  (per-rank files, both ranks deciding alike), filtered by the same drafter/grid gates as memory hits.
- A disk hit is **materialized into a fresh extent** (`_spill_hydrate`): `load_rows()` writes the file's
  rows into the arena, and the snapshot becomes an ordinary kept prompt (`rec/conv/tail/head/drafter_rows`
  straight from the file). Hydration allocates with `pool.place()` **only — no evictions**, so both ranks
  stay in lockstep; a new `SPILL_LOAD = 11` op (emitted before the ADMIT that references the hydrated kid)
  tells rank 1 to materialize the same ids from *its* files.
- Image prompts are neither spilled nor resumed from disk: their rows belong to the picture's fingerprint,
  not to token ids (`vision_sig is not None` → skip).

### Verification signals (no probe traffic needed)

| Signal | Where |
|---|---|
| Spill actually writes | `*.rank{0,1}.safetensors` appear in the spill dir when an eviction happens |
| Disk resume works | rank-0 log: `[tensorfold] a kept prompt of N tokens came back from the spill tier` |
| And the money line | next `done req-…` for that conversation shows `cached=N` (not `cached=0`), TTFT back to ~1 s |

## ⚠️ Traps hit on the way

- **`pool.place()` must not evict inside hydration.** Rank 0 may only take lockstep-safe actions before
  emitting `SPILL_LOAD`; anything rank 1 cannot mirror identically (eviction choices, its own file
  possibly pruned) must fail loudly instead of silently diverging.
- **`save_rows(e, snap, drafter=None)` nukes `snap.drafter_rows`** (`drafter_end = -1`). The multi path's
  DFlash2 window lives in `drafter_rows` (a ring drafter), so the row extraction had to be done via
  `_row_views` on a temp `State`, leaving the snapshot's own fields intact.
- **Files from the single-stream writer lack `tail`; files from the multi writer lack nothing the other
  side needs** — the extra fields are additive, and `put_ring_tail` treats `None` as `[]`, so old files
  still load (they just can't satisfy a multi resume that needs a ring tail).
- The tier's only other write trigger is a clean shutdown (`SPILL_STOP`, rank 1 spills then stops) —
  `docker rm -f` never reaches it, which is why production ran 30 h with an empty directory.

## Deployment notes

- The overlay is mounted read-only by the start scripts (TP2: `OVERLAY_DIR=/root/tf-overlay-tp2-v067`;
  TP4: `TF_OVERLAY=/root/tf-overlay-tp4`), preflight-checked on every host. Rollback = remove the two
  `-v` lines and restart.
- TP4's overlay additionally promotes the v0.6.7-baked `multi.py` (sys-checkpoint, `SHARED_SLOTS=3`),
  which an older single-file overlay had been shadowing; the spill patch is the same file for both stacks.
