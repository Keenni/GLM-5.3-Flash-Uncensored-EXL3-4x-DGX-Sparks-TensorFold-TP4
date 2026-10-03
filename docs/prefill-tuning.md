# Prefill: what the knobs do, what we patched, and an HC-split postmortem

**One-line conclusion:** TensorFold GLM prefill at ~700 tok/s on a 4-rank GB10 ring is the *native
baseline*, not a defect — the other TP4 effort (drowzeys) measured the same numbers on v0.6.1. With three
changes (one-shot sparse, a chunk-size kernel patch, higher concurrency) we sit at **~800 tok/s**, with
the decode side unchanged. The one big lever we tried and **removed** was the hyper-connection row-split
(HC_SPLIT): our W4 port passed every check we had and still produced deterministic garbage.

Current production knobs (see `scripts/start-glm53-flash-tf-unc-tp4-v062.sh`):

| Knob | Value | Status |
|---|---|---|
| `TF_GLM_SPARSE_ONEPASS` | 1 | ✅ verified **bit-exact** (sentinel unchanged), kept |
| `TF_GLM_PREFILL_ROWS` | 4096 | ✅ verified, kept — required the kernel patch in §2 |
| `PARALLEL` (engine `PARALLEL_MOST`) | 12 | ✅ verified, kept — policy constant, see §3 |
| `TF_GLM_HC_SPLIT` / `TF_GLM_PREFILL_OVERLAP` | 0 / 0 | ❌ removed — see §4 |

## 1. The baseline is upstream, not a regression

Cold-prefill measurements on our stack: 49,985 tok → 677 tok/s; 93,237 tok → 699 tok/s; 200,086 tok →
699 tok/s. drowzeys' independent measurements on v0.6.1: 726 / ~773 / 717 / 497 tok/s at 4K–128K. Same
league. Their optimization campaign (indexer top-k radix-select, per-shape prompt-GEMM tiles,
UnpackCache, 8192-row chunks, sequence-parallel glue) brought *their* family to ~1000–1100 tok/s — by
**writing kernels**, not by configuring. That set our expectations correctly: anything on our path is
kernel engineering too.

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
on that request; answers correct. Cost: `PREFILL_ROWS=4096` costs ~1.4 GiB of pool versus 2048
(6,998,016 → ~6.65M tokens at the final configuration).

## 3. Concurrency: a policy constant, not physics

`PARALLEL_MOST = 6` lives at one place in `engine.py` and is used by exactly one guard; the scheduler and
geometry take the runtime value (`--parallel 12` alone just errors with
`1 to 6 requests at once for GLM-5.3-Flash, not 12`). Raising the constant to 12 and restarting is the
whole change. Pool cost is nearly free: `PARALLEL 6 → 12` costs ~94k tokens (≈ 0.6 GiB), since each extra
stream adds one `slot_bytes` + ring buffer.

## 4. HC_SPLIT at world=4: the full postmortem (removed from production)

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
`forward.py`. We removed HC_SPLIT instead (`HC_SPLIT=0`); the `reduce_scatter` primitive from their
branch remains ported in `cuda/comm.py` for whoever picks this up.

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
