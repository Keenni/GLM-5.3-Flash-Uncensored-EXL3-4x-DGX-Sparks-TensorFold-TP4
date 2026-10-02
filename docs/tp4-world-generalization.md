# TP4 world-generalization of TensorFold's GLM-5.3-Flash engine

This is the engineering dossier of the overlay: what stock TensorFold v0.6.0 (as shipped in Mia's 2× Spark
image) hard-codes about its two-rank world, what we changed, and — more usefully to the next person — how
the three multi-day failures announced themselves and how to recognize them in minutes instead.

Everything below was learned on a four-node DGX Spark (GB10, sm_121, 128 GB unified memory) fiber ring with
the checkpoint of [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3).

## 1. What stock code assumes world = 2

The `glm5_next` CUDA engine is written for exactly one master (rank 0) and one follower (rank 1). The
overlay generalizes it; the full edit list:

| File | Change |
|---|---|
| `cli_args.py` | `--tp` choices `(1,2)` → `(1,2,4)` |
| `cli.py` | master validation `tp > 1`; worker branch `rank == 1` → `rank > 0`; banner prints `rank of N`; `--rank` choices `(0,1)` → `(0,1,2,3)` (argparse rejects rank 2/3 otherwise — validate by really parsing, not by grepping constants) |
| `families/glm5_next/__init__.py` | TP gate `!= 2` → `not in (1, 2, 4)`; passes `world = tp` into `GlmEngine` |
| `cuda/engine.py` | `NCCL(rank, world)`; `mla_geometry` / `dflash2_geometry` / `multi_draft_bytes` / `slot_bytes` / `split_weights` / `dflash2_weights` / `admit` all carry `world`; `_gather_ints` / `_share` / `VisionFeed` buffers `2 *` → `world *`; settings comparison N-rank aware; doorbell keys per follower (below); `PARALLEL_MOST` 4 → 6 |
| `cuda/weights.py` | `load(world=)`; divisibility guards so a checkpoint that doesn't split evenly at world 4 fails loudly instead of silently mis-splitting (heads 96/4, lin_heads 64/4, vocab 154880/4, expert tile (288×2048)/(64×4) — all exact for this checkpoint) |
| `cuda/split.py` | `split_bytes` / `split_device` / `RankReader` / `_span` row fast path fully world-parameterized (rank r owns byte ranges `[r·S/4, (r+1)·S/4)`); the numpy reassembly self-test (8 cases) must pass |
| `cuda/hcsplit.py` | explicit `raise` for world > 2 — the HC split is a 2-rank row mechanism (`peer = 1 − rank`); we run `TF_GLM_HC_SPLIT=0`. Failing loudly here beats a silent wrong split |
| `cuda/multi.py` | rank-0 async `_share` (both length and values buffers) and `_time_rows` all-gather receive buffers `2 *` → `world *` |
| `cuda/segments.py` | `MAX_SEGS` 4 → 6 (parity with the production TP2 tuning) |

Already world-safe in stock v0.6.0 (verified, do not touch): `cuda/comm.py` (NCCL all-gather/exchange/
ready/TCPStore), `forward.gather`, `glue.residual_add`, decode top-k gather, `capacity.gather_ints(world)`,
`multi.py`'s own drafter paths via `self.world`, the `.cu`/`.cpp` kernels (no recompile, caches stay valid).

## 2. Failure #1 — the request doorbell (requests hang forever, one GPU busy)

Stock: rank 0 "rings" a single TCPStore key per request; rank 1 waits on it. With three followers they all
wait on — and race to delete — the *same* key; one deletes it, the other two never see it.

Symptom (learned the hard way, spotted first by our human operator): requests issue and never respond;
**only the master's GPU shows load** (~96% util at ~10 W, spinning in the all-gather), the other ranks' GPUs
are idle — the followers are blocked in the doorbell wait, *before* entering any collective. That GPU
pattern is the fastest differential between "deadlocked" and "desynchronized".

Fix — one key per follower:

```python
# engine.py, rank 0 side:
self._bell = getattr(self, "_bell", 0) + 1
for r in range(1, self.world):
    self.store.set(f"tf_glm_request_{r}_{self._bell}", b"1")

# follower r side:
key = f"tf_glm_request_{self.rank}_{getattr(self, '_bell', 0) + 1}"
```

## 3. Failure #2 — rank 0 binding (container green, endpoint dead)

TensorFold's `serve` defaults to `127.0.0.1:8080`. Inside a container that binds fine, logs look perfect
(`serving ... at http://127.0.0.1:8080/v1`), all four containers are Up — and nothing answers on :8000.
The start script must pass `--host 0.0.0.0 --port 8000 --name <served-name>` on the rank 0 invocation
(other ranks take neither). If your endpoint is dead but containers are green, check this before anything
else.

## 4. Failure #3 — all-gather receive buffers sized `2 *` (the killer, twice)

Any remaining `torch.empty((2, ...))` / `view(2, ...)` receive buffer for a world-rank all-gather dies at
the first use of that path with exactly one clean line and no traceback:

```
tensorfold: all_gather: recv must hold world x send of the same dtype
```

(the size check lives in `cuda/comm.py`; the top-level `except` in `cli.py` prints `{exc}` and exits).

The complete checklist of buffers that must be world-sized — every one of these bit us or was one edit
away from doing so:

- `engine.py`: `_gather_ints`, `_share` (two buffers: length and values), `VisionFeed`, and **`_calibrate`
  — the startup calibration, the first thing to die and the easiest to miss**
- `multi.py`: rank-0 async `_share` (two buffers), `_time_rows`
- `hcsplit.py`'s literal `2`s are the 2-rank row mechanism, not a world size — leave them (HC_SPLIT is off
  at TP4 anyway)

Pre-boot static self-check (seconds, run it every time you touch the overlay):

```bash
cd overlay/tensorfold && grep -rn 'torch.empty((2\|view(2,' families/glm5_next/cuda/*.py | grep -v hcsplit.py
# → must print nothing
```

## 5. NCCL on a switchless four-node ring

TensorFold's comm path runs on NCCL; on a direct-cable ring (no switch) stock NCCL wedges in Tree/PAT
setup — the switchless story. Our stack:

- **Library**: patched NCCL 2.30.7 for sm_121/arm64 per
  [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) (SparkRing skip-Tree/PAT +
  OpenFaaS hardening), injected with `TF_NCCL_LIB=/opt/patched-nccl/libnccl.so.2` (TensorFold's name for
  what vLLM calls `VLLM_NCCL_SO_PATH`).
- **Environment**: the complete 23-variable set is in `scripts/start-glm53-flash-tf-unc-tp4.sh`. The first
  boot used only the 4 "obvious" ones (HCA list, socket ifnames, GID index, address range) and all four
  ranks died at the engine's first `comm.barrier()` with `NCCL error 2: unhandled system error`;
  `NCCL_DEBUG=INFO` showed `ibv_modify_qp failed with 110 Connection timed out, curr state INIT, next state
  RTR` — NCCL was pairing each local GID with the peer's *other* cable (physically absent). The three
  classes that fix it, none optional with the patched build:
  1. ring leg selection: `NCCL_IB_SUBNET_AWARE_ROUTING=1` + `NCCL_IB_SUBNET_PREFIX_LEN=24`
  2. the patch's own switches: `NCCL_SKIP_TREE_CONNECT=1` + `NCCL_SWITCHLESS_RING_ONLY=1` (the hardening
     patch gates its behavior on these; without them the patch is inert)
  3. alignment with the vLLM TP4 production set: `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_ALGO=Ring`,
     `NCCL_CROSS_NIC=1`, `NCCL_MIN_NCHANNELS=4`, `NCCL_MAX_NCHANNELS=4`, `NCCL_IB_MERGE_NICS=0`,
     `NCCL_IB_ADDR_FAMILY=AF_INET`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_CUMEM_ENABLE=0`,
     `NCCL_NVLS_ENABLE=0`, `NCCL_IGNORE_CPU_AFFINITY=1`, plus HCA/socket/GID/address-range entries.
- **Validate the fabric before any full boot**: `scripts/tf_nccl_diag.py` builds TensorFold's own
  `comm.NCCL` at world=4 and runs one all-gather — seconds, no model load. All four ranks must PASS before
  you spend a boot cycle on the real thing.
- Harmless noise seen on every boot (also present in our vLLM production): `Spectrum-X (SPCX) plugin is not
  supported ... skipping` and `ibv_query_port_speed errno 93`.

Why a ring at all: with four nodes cabled 0–1–2–3–0, NCCL's Ring algorithm needs exactly those four edges —
the topology is self-consistent, and Tree/PAT (which want non-adjacent connections a cable ring can't
offer) are skipped by the patches. `TF_GLM_COMM=roce` (TensorFold's one-shot RoCE transport) remains
untested above world=2; NCCL is the safe default.

## 6. KV pool and prefix cache (semantics differ from vLLM)

`engine.py` sizes the pool as `spare = min(TF_GLM_CACHE_GIB, budget − estimate)` with
`budget = MemAvailable_at_start − TENSORFOLD_MEMORY_RESERVE_GIB`:

- `TF_GLM_CACHE_GIB` is a **cap**. We set `1000` so it never binds and the pool auto-sizes (measured:
  budget 97.92 GiB, weights estimate 50.20 GiB → **8,077,312 tokens**; with the cap at 12.5 the same boot
  yields only 2,895,872). The env parser only accepts floats — `TF_GLM_CACHE_GIB=auto` crashes; omitting
  it falls back to 3 GiB.
- `TENSORFOLD_MEMORY_RESERVE_GIB` (14.5 upstream, keep it) is the real headroom knob for 1M-token prompt
  peaks.
- `CACHE_ENTRIES=64` costs a little pool (each entry reserves slot state) in exchange for more concurrent
  conversations keeping their prefix state.
- `--prompt-cache-gib` / `--checkpoint-slots` are **MLX-path options** and do nothing on the CUDA path for
  glm5_next — don't tune them here.
- Prefix cache (kept prompts) triggers on a **shared system prompt**, not on arbitrary common prefixes:
  a second conversation with the same system prompt hit 1,984/2,035 cached (97.5%), TTFT 1.8 s → 0.2 s;
  identical requests *without* a system message hit 0% by design. Clients that want hits must pin one
  stable system prompt. Watch `curl :8000/metrics | grep tensorfold_health:` (`cached_tokens_total`).

### 6.1 Field observation: warm-up lag and multi-agent contention (operator report)

Two behaviors from real agent traffic on this stack — self-observed over days of daily use, mechanism not
pinned down yet, reported as-is:

- **The prefix cache warms up late.** The first 2–3 turns of a conversation never hit the kept-prompt
  cache; hits start from roughly the third turn onward. This looks exactly like the vLLM symptom of a KV
  pool too small and evicting other sessions — but the 8,077,312-token pool behaves the same way, so pool
  size is not the cause. Kept-prompt checkpoints appear to lag the live conversation by a couple of turns.
- **Concurrent agents contend hard.** With several agents running at once, GPU time is dominated by
  prefill and decode stalls; sessions visibly push each other around, and per-session throughput drops
  faster than on our vLLM TP4 stack on the same ring.

Practical read: TensorFold (this overlay included) is at its best with **one user and a few sessions** —
exactly the interactive case TP2/TP4 was shaped for, and there it is excellent. For multi-tenant,
many-agent serving, vLLM or SGLang on the same hardware remain the safer choice today. If you deploy
this, benchmark it against your own concurrency pattern first.

## 7. Deployment notes that cost us time

- **Overlay mount path must keep the `tensorfold` layer**: host layout
  `/root/tf-overlay-tp4/tensorfold/...` bind-mounted over the image's `/opt/tensorfold`. Mounting the
  parent directly fails with "not a directory".
- **Verify inside the image, not on the host**: `scripts/verify_overlay_in_image.py` runs 13 assertions in
  a throwaway container with the exact mounts the start script uses; `scripts/verify_cli_tp4.py` parses a
  real argv through `cli_args` + the family gate. Both run without GPUs.
- **Weight distribution**: 92 shards ≈ 41 GB/rank; `RankReader` maps each rank's byte range straight from
  the flat shard directory — no pre-splitting. `scripts/distribute-to-0102.sh` idempotently ships image
  (`ctr -n moby images export | import` — `docker save` produced empty-layer archives on our cluster) and
  shards.
- **Container image distribution on DGX Spark**: `ctr -n moby images export` needs the *fully qualified*
  image reference; after import, verify with a real `docker create` — a green `inspect` is not evidence.
- **Overlay integrity across nodes**: `overlay/MANIFEST.md5` covers all 9 files; the start script
  preflights it on every rank and refuses to boot a mismatched fleet.
- **In-container `ssh` in loops**: a bare `ssh host "test -f ..."` inside `while read` eats the loop's
  stdin and silently ends the loop after one iteration; redirect `< /dev/null` or use rsync's own
  checksumming.

## 8. What we measured (2026-10-02)

Same fleet, warm kernel caches, single boot. Orientation numbers, not a benchmark suite:

| Item | Value |
|---|---|
| Boot to first reply | 117 s (warm kernel cache) |
| KV pool at start | 8,077,312 tokens (cap=1000 GiB, reserve=14.5, entries=64, ctx=1M, parallel=6) |
| Prefill, cold | 8.2k tok → 6.65 s; 13.7k tok → 10.8 s (≈1.25k tok/s) |
| Prefill vs our vLLM TP4 NVFP4 (same ring) | ≈35% slower (vLLM ≈2.0k tok/s) |
| Decode, c1 prose | ≈80+ tok/s, ~1.5× our vLLM TP4 baseline (self-timed) |
| Decode under concurrency | falls off faster than the vLLM stack |
| Prefix cache (shared system prompt) | 97.5% hit, TTFT 1.8 s → 0.2 s |
| Identical request, no system message | 0% cache hit (by design) |
