# Prefill: what the knobs do, what we patched, and an HC-split postmortem

**One-line conclusion:** TensorFold's prefill on our Flash stack starts at ~700 tok/s on long prompts (our
own measurements, below) — the native baseline, not a defect. Three levels of fixes took it to
**~1,600 tok/s cold** (16k prompt → 1,575 tok/s; 8,192-row chunk → 1,654 tok/s): one-shot sparse, a
chunk-size kernel patch, higher concurrency — and two kernel-level fixes that dominate the account:
routing MoE prefill back to Mia's v19 prompt-expert kernel (**7.2×** on the dominant block, §7), and
root-causing + fixing our W4 hyper-connection row-split, which is **back in production** after an
embarrassing first round (§4). Decode is unchanged throughout.

> ⚠ **What the numbers are *not*: the other TP4 effort's benchmarks.** drowzeys' PR #159 works on a
> **different model** — the 753B GLM-5.3 (`glm_moe_dsa` family), not this Flash checkpoint. Its prefill
> numbers (native and optimized alike) describe a different model with a different compute profile and do
> **not** serve as our baseline or our ceiling. What transfers from that work is *methodology* (which
> kernels matter, how to sequence-parallelize glue, how to grade chunk sizes), not numbers.

Current production knobs (see `scripts/start-glm53-flash-tf-unc-tp4-v062.sh`):

| Knob | Value | Status |
|---|---|---|
| `TF_GLM_SPARSE_ONEPASS` | 1 | ✅ verified **bit-exact** (sentinel unchanged), kept |
| `TF_GLM_PREFILL_ROWS` | 8192 | ✅ verified, kept — required the kernel patch in §2 |
| `PARALLEL` (engine `PARALLEL_MOST`) | 12 | ✅ verified, kept — policy constant, see §3 |
| `TF_GLM_FILL_SHARE` | 0.3 | ✅ kept — prefill yields to active decode (idle streams don't participate) |
| `TF_GLM_HC_SPLIT` + `HC_EXCHANGE=gather` | 1 | ✅ **re-enabled after root-cause fix** — bit-exact vs the unsplit baseline (§4) |
| `TF_GLM_PREFILL_OVERLAP` | 0 | next candidate (§7 end) |

## 1. The baseline is ours, measured on this stack

Cold-prefill measurements on our own Flash stack, engine-side, before any of the changes below: 49,985
tok → 677 tok/s; 93,237 tok → 699 tok/s; 200,086 tok → 699 tok/s. That is the starting point these
patches are measured against — there is no upstream benchmark for this model on this hardware to compare
with (see the warning above: drowzeys' numbers belong to the 753B family).

Their optimization campaign (indexer top-k radix-select, per-shape prompt-GEMM tiles, UnpackCache,
8192-row chunks, sequence-parallel glue) is still the best available map of *which kernels matter* — but
the relative gains they report were measured on their model and only transfer as hints. What their work
establishes beyond doubt is the *kind* of effort required: writing kernels, not configuring.

`TF_GLM_PROFILE=1` gives per-block timing with device sync (diagnostic only — it perturbs timing and
needs a restart; do not run it in production config).

## 2. Chunk size: GB10's shared-memory ceiling and the L2 fallback

`cuda/exl3/experts.cu` stages the routing `pick` buffer (rows × `experts_per_tok+1` ints) in *dynamic*
shared memory: `smem = rows × 9 × 4` bytes, capped by `sharedMemPerBlockOptin − static` ≈ **101,248 B on
GB10** ⇒ a hard ceiling of **~2,812 rows per chunk**. Requesting `PREFILL_ROWS=4096` fails loudly:
`EXL3 grouping needs 147456 dynamic shared-memory bytes` (4096 × 9 × 4 = 147,456 > 101,248). drowzeys'
8192-row chunks work for them only because their any-K prompt kernel doesn't take this path.

The smem staging exists because the two-phase scan revisits the same data repeatedly — but the `pick`
buffer is tiny and L2-resident anyway. The patch (few lines in `experts.cu`):

- kernel takes an `int staged` flag; the scan reads `staged ? sh_pick : pick`;
- host side: when `smem > limit`, set `staged = 0, smem = 0` (direct global/L2 reads).

Same values, same plan, same bits — the row ceiling is simply gone. Verified with a 8,075-token request
whose first chunk is 4,096 rows (previously a guaranteed crash): no errors, `prefill=8.36s` ⇒ 966 tok/s
on that request; answers correct. Cost of larger chunks against the pool: `PREFILL_ROWS=2048 → 4096`
costs ~1.4 GiB (6,998,016 → ~6.65M at the earlier configuration); `4096 → 8192` brings the current
start-up pool to **6,238,208 tokens** (at `PARALLEL=12`, `CACHE_ENTRIES=128`).

## 3. Concurrency: a policy constant, not physics

`PARALLEL_MOST = 6` lives at one place in `engine.py` and is used by exactly one guard; the scheduler and
geometry take the runtime value (`--parallel 12` alone just errors with
`1 to 6 requests at once for GLM-5.3-Flash, not 12`). Raising the constant to 12 and restarting is the
whole change. Pool cost is nearly free: `PARALLEL 6 → 12` costs ~94k tokens (≈ 0.6 GiB), since each extra
stream adds one `slot_bytes` + ring buffer.

## 4. HC_SPLIT at world=4: the full postmortem (root-caused; see §4.1)

The hyper-connection row-split — splitting a prompt chunk's glue across ranks — is upstream a
**two-rank feature** (patch docstring: *"split by rows between the two ranks"*; the guard literally
raises `TF_GLM_HC_SPLIT needs two ranks`). Our first W4 port never ran: it died at the 64-row startup
warm-up prefill on two latent bugs:

1. **`self.peer = 1 - rank` is a 2-rank concept.** At world=4, ranks 2/3 got `peer = −1/−2`, producing a
   negative slice start into the all-gather buffer ⇒ an empty view ⇒
   `shape '[16, 4096]' is invalid for input of size 0` on ranks 2/3 only (master/first-follower looked
   healthy — a misleading signature).
2. **`_swap_rows` fed bf16 rows into fp32 buffer slices** ⇒
   `all_gather: recv must hold world x send of the same dtype`. The buffer is designed to be borrowed
   *through one bf16 slot* (the fp32 half-width) — the fix is a `.view(torch.bfloat16)`.

After fixing both, the stack booted, the smoke test passed, the bit-exact sentinel was unchanged — **and
long prompts still produced deterministic garbage**: from ~700 tokens up, answers ignored trailing
instructions, continued prose, emitted noise (`SPAO SPAO`). Short prompts (<64 rows, where the split never
engages) were fine; the same prompt repeated gave byte-identical wrong output (deterministic miscalc, not
a race); the 10-token smoke prompt never exercises the path, which is exactly why it passed.

The design itself was unsound at W4 (the W4 branches were written by analogy, never validated): the
row-swap loop lets four ranks overwrite the same destination segment in turn, and the partial-swap
pairing mixes rows of *different* pairs inside one all-gather — the subsequent world-summed
`hc_post` then sums the wrong rows together.

**How to catch this class early — the verdict battery.** Before trusting any never-executed port branch,
run a prompt-length-graded battery of *verifiable* instructions (one instruction at the end of each
prompt):

```
250 tok → OK · 782 tok → BAD · 2,660 tok → OK but leakage · 3,632 tok → BAD (noise) · 7,622 tok → BAD
```

This localizes the failure to the first chunk size that engages the path. A smoke test with a fixed short
prompt (see §5) is *necessary but not sufficient*.

**The correct W4 design** (per drowzeys' sequence-parallel glue, §10-02 of their prefill design doc):
contiguous row chunks per rank; each all-reduce becomes a **reduce-scatter over rows** (NCCL bf16
`ncclReduceScatter`, or an fp32 all-to-all of row shards + rank-ordered sum); glue ops on own rows only;
then one in-place all-gather spreads the projected tensors back. Their implementation is validated
bit-identical at 4 ranks — but the gain is only +8–12%, and it means rewriting the reduce points in
`forward.py` — that remains an orthogonal, unshipped direction; the `reduce_scatter` primitive from their
branch is ported in `cuda/comm.py` for whoever picks it up.

### 4.1 Resolution: root cause found, fixed, and HC_SPLIT is back in production

The garbage had a mundane root cause, and we owe the earlier "the design itself was unsound" reading a
correction: **`partial()` carried two W4 branches — an early one, inherited from the W2 shape, that filled
only one row-block (`fill(theirs, theirs+H)`) and returned, and the correct one we wrote later
(`_fill(fill, 0, R)` — all rows, one swap) that had become dead code.** Every rank therefore computed
fresh partials for just 1/4 of its rows; the rest were zero/stale. That also explains why every
comparison "passed": both sides of the check read the *same* quarter-filled buffer — a self-referential
verification. (The first dump even showed `T0[0]=T0[1]=T0[2]=0`; we ignored it.) Two more fixes landed
with it: a **private scratch buffer** for the split (no longer borrowing the engine's shared `b.gath`,
which collides with other users), and the non-overlap branch — the one `PREFILL_OVERLAP=0` actually runs —
being the place the fix had to apply.

**Verification (bit-exact against the unsplit engine):** three fixed prompts, `sha=` digests from the
`done req-` log lines — split and unsplit must agree exactly: `370501976a66` / `a99b19fa58c8` /
`a99b19fa58c8`. **All three match.** With the fill fixed, the W4 gather design (all-gather of partials +
rank-ordered `hc_post`) is sound after all; the earlier "four ranks overwrite each other" analysis
described the dead-code branch's behavior, not the shipped one.

**Production numbers with the split on** (on top of §7's v19 kernel):

| | unsplit (v19 kernel) | + HC_SPLIT (W4, gather) |
|---|---|---|
| 8,192-row chunk | 1,426 tok/s | **1,654 tok/s** (+16%) |
| 16k cold prefill | — | **1,575 tok/s** (10.16 s) |

Decode unchanged; the 10-token smoke sentinel is unchanged too (it never reaches the path — see §5).
Remaining known follow-up: re-enabling `TF_GLM_PREFILL_OVERLAP=2` (hiding the exchange behind the next
block's row-independent front) — it needs the parity-alternating call order on the ring (even ranks
`[right,left]`, odd ranks `[left,right]`) or adjacent ranks deadlock at warm-up with no error printed.

## 5. The bit-exact sentinel

The bring-up script's smoke test uses a fixed prompt (`Reply with OK.`), and the engine logs a
`sha=` digest per request. **Any change to numeric paths must leave that digest identical** — a changed
digest means your change altered output bits and needs re-evaluation. Baseline: `694aab7656bc`.

> ⚠ A *passing* sentinel does not clear a code path the smoke prompt never reaches (that is exactly how
> the HC-split garbage survived). The sentinel proves you didn't change bits where you touched; the
> §4 battery proves you didn't break paths you never touched.

## 6. drowzeys' PR #159 — why it is a no-op for `glm5_next`

The PR (`glm-moe-dsa-tp4` branch) is a **new family** for the 753B model; `families/glm5_next/` is
untouched. Of its shared-file changes, `glm5_next` only walks one (`cuda/exl3/experts.*`), and that change
encodes *their* checkpoint's fused-trellis packing (`ns`/`gu_stride`) — with our mcg-codebook checkpoint
those arguments are zero/no-op. Their headline kernels are not reusable as-is, partly by their own note
("not reusable: glm5_next exl3.cu is 4-bit + mcg2 + uniform stack") and partly by codebook mismatch
(mul1 vs mcg). Their `prefill.py` `_TUNED` tables are bit-identical but for dense-matmul paths
`glm5_next` doesn't walk. What *is* portable: the methodology (per-shape tile tables, UnpackCache, big
chunks, radix top-k), their `reduce_scatter`, and their GB10 field notes.

## 7. The big one: routing MoE prefill back to Mia's v19 prompt-expert kernel (**+80%**)

`TF_GLM_PROFILE=1` on a real 8,192-token chunk showed **69% of prefill inside MoE**, with the single
block `moe: routed (exl3)` at **51%**. The reason: in the v0.6.2 rebase the MoE call path got wired to
the *shared* `cuda/exl3/experts.py` — the decode-shaped kernel, tuned for small windows — while Mia's
dedicated prompt-expert kernel (`exl3_mm.py`: `routed` / `Exl3Experts` / `Scratch` with the
prompt-tuned tile plan) was left in the tree with **zero call sites**. The shared kernel on 8,192-row
chunks is a plain mismatch.

The restoration is small because `exl3_mm.py` is byte-identical in both trees; only the wiring moved:

1. `weights.py`: import back from `.exl3_mm` and rebuild the stacked container (`torch.stack` over the
   checkpoint's `trellis`/`suh`/`svh`);
2. `forward.py`: drop the shared-kernel imports; allocate `self.ey` as a **float32** buffer on the exl3
   path (it has its own scratch instead of reusing the bf16 activation buffer); construct Mia's
   `Scratch(rows, slots, D, ml, dev, prompt=prompt)`; call `grouped.route(...)` unconditionally and
   dispatch to `exl3_mm.routed(...)`;
3. `self.plan` must be built after `prompt` is known (`tile=prompt_pass() if prompt else None`) and must
   **not** be nulled on the exl3 path (`moe_block` uses `b.plan` unconditionally).

Measured on the same 8,192-token chunk:

| | shared decode kernel (before) | Mia v19 (after) |
|---|---|---|
| `moe: routed (exl3)` | 5.32 s (51%) | **0.74 s (13%)** — **7.2×** |
| MoE total | 7.15 s (69%) | 2.53 s (44%) |
| chunk rate | 793 tok/s | **1,426 tok/s** (+80%) |
| 37k cold-prefill probe | 48.2 s (767 tok/s) | **26.8 s (~1,300 tok/s)** (+70%) |
| decode | 60–68 tok/s | 63–68 (no regression) |

(Mia's own comment claims "2.3–2.5× at 4,096 rows"; we measured 7.2× because the shared kernel is even
more mismatched at 8,192.) Note the smoke sentinel (`Reply with OK.`, 10 tokens < `PROMPT_ROWS=64`) walks
the decode path, so it *cannot* validate this change — correctness needs a Q&A battery (short/medium/long
prompt + trailing instruction), same discipline as §4.

**New bottleneck after this fix:** communication — `moe: all-gather` 23% + `kda: out + all-gather` 20%
(≈43%), consistent with drowzeys' "~160 small all-gathers, latency-bound" account; their remedy is
reduce-scatter over rows + sequence-parallel glue (§4.1's "correct design" paragraph). That is real
kernel/forward-path work, the next lever if prefill matters further.
