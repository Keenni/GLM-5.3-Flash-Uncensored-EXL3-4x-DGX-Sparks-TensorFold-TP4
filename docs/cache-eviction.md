# Cross-session cache eviction: why conversations kept wiping each other's prefix cache

**One-line conclusion:** under concurrent streams, TensorFold's kept-prompt eviction had three paths that
destroyed *other* sessions' cached state — an idle few minutes cost a full re-prefill (26–66 s) even with
nobody competing for memory. After the fixes, returning to a 5-turn-old conversation re-hits
43,904 cached tokens in 7.9 s instead of re-prefilling for 66 s.

## 1. Symptoms (all reproduced on the live stack)

- A session's `cached=` count only jumps between two "floor" values (e.g. `49024` / `66112`); after a few
  idle minutes, the next turn eats a 26–55 s full prefill.
- **Unrelated to contention for memory** — a completely idle system behaves the same.
- Two sessions alternating: whichever session went second wiped the first one's chain; returning to the
  old conversation cost a full re-prefill (66 s at a ~44k-token state).

Diagnostic tooling: `docker logs | grep "done req-"` shows the `cached=` trajectory per request; setting
`TF_GLM_TRACE_KEEP=1` emits `[keep]` lines at every `_resume/_keep/_evict/_room/_home` decision, e.g.
`EVICT why=takeover|home|room|KEEP-CAP` and `RESUME … -> HIT/MISS/REJECT`.

## 2. Bug 1 — `cache_bytes = 0` read as "zero budget" (engine level)

Mia's engine sets, when `parallel > 1`:

```python
self.cache_bytes = 0   # kept prompts live in the pool
```

But two eviction paths use `cache_bytes` as an **upper bound**:

```python
# _remember:   while len(cache) > 1 and (len > cache_entries or self._held_bytes() > self.cache_bytes):
# _take_over:  while self._held_bytes() + need > self.cache_bytes: ...
```

`_held_bytes()` counts KDA states / conv rows / MTP rows that `_drop()` does **not** release (it only
clears rows/nbytes/drafter rows), so `_held_bytes() > 0` is permanently true ⇒ `_remember` trims kept
prompts down to **one entry every turn**, and `_take_over` skips saving KV rows entirely.

**Fix (safe variant — do *not* "fix" it to unlimited):**

- `_remember`: guard the byte criterion with `self.cache_bytes and ...` (entry-count cap still applies).
- `_take_over`: when `cache_bytes == 0`, `self._drop(snap); continue` — drop *exactly the candidate under
  consideration*, nothing else.

❌ The tempting wrong fix — treat `cache_bytes == 0` as "no limit" — makes `_take_over` **copy** every
candidate's KV rows into the pool; one ~100k-token state is ≈ 675 MB, and 128 of them blow the pool.

## 3. Bugs 2 & 3 — the `multi.py` extent manager (the real killers)

> Lesson recorded the hard way: the engine-level fix above turned out to be **dead code** under
> `parallel > 1`, because `multi.py` maintains its *own* kept-prompt machinery (extents). Before changing a
> code path, `grep` for its call sites — a function existing does not mean it is the live one.

The invariant we landed on: **a session that has been kept owns its extent; no other request may write
into it.** Two paths violated it:

### 3.1 `takeover` — in-place adoption scythes other sessions

When a new request hits tokens inside an extent that belongs to another conversation, the old code adopted
it *in place* — and swept away the **entire kept chain** of the session that owned the extent, because
those rows fell inside the matched span.

**Fix:** if any doomed kept rows are non-empty, skip in-place takeover entirely and take the (already
existing, merely mis-prioritized) **copy-to-new-extent** path — copy only the rows the new request needs.

### 3.2 `home` — a lone request evicts what it just inherited

With a single active lane (the common case for one user), the manager "homes" the extent back to base 0 so
the single-stream CUDA graph can engage — and along the way **evicts every kept prompt in `[0, x.size)`**.
The previous turn's `home` had just *moved* the kept chain to base 0. Net effect: every new request erased
the previous conversation's cache. In a multi-tenant world that is fine-ish; for one user returning to
their own conversation it was fatal.

**Fix:** *move* the kept extent above the live one (`_emit(MOVE)` + `_move`) instead of evicting it — the
pool has millions of free rows and a handful in use, so space is not the constraint.

⚠ Two hard rules when moving extents (learned from a 500-error):

1. **Validate the destination with `pool.gaps()` before emitting the move.** If `_emit` has already been
   issued and `_move` then throws (`cannot move to [a, b): taken`), rank 0 and the followers diverge and
   only a restart cleans it up.
2. The destination must lie **entirely above the live lane's `x.end`** — the lane itself occupies space
   below, and naive `ceil` arithmetic that ignores the lane's own extent overflows into a taken range.

## 4. Verification (controlled A/B)

Reproduction: session A runs 5 turns → session B runs 2 turns → return to session A, turn 6.

| | Before | After |
|---|---|---|
| A turn 1 | cached=0 / 42.3 s | cached=0 / 13.4 s |
| A turns 2–5 | hits 9,984 … 35,456 | hits 9,984 … 35,456 |
| B turn 1 (fresh session) | hits system block, 1,536 | hits 1,536 |
| **A turn 6 (back to old session)** | **cached=1,536 / 66.0 s** | **cached=43,904 / 7.94 s** |

Post-fix trace: `TAKEOVER-SKIP in-place: 5 kept would die (longest=43904), copying 1536 rows instead`,
`HOME-MOVE extent=1 base=0 -> 53248`, `RESUME prompt=49433 kept=8 -> HIT ids=43904`; `EVICT
why=takeover|home` counts go to **zero**, no errors.

## 5. Context: why there is nothing upstream to copy

TensorFold's other TP4 effort (drowzeys' PR #159, the `glm_moe_dsa` family) states in its design docs that
shared-prefix caches and multi-stream caches are **explicitly deferred** — its streams get fixed, up-front
slot allocations and no cross-session reuse. Mia's kept/extent machinery is the only system doing
cross-session reuse here, so its bugs had no upstream fix to pull. For the full comparison, see
[prefill-tuning.md](prefill-tuning.md).
