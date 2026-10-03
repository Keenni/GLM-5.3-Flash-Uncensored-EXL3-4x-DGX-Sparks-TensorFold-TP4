<h1 align="center">GLM-5.3-Flash Uncensored EXL3 on 4× DGX Spark — TensorFold TP4</h1>

<p align="center">
  <sub>Serving GLM-5.3-Flash from four DGX Sparks over a switchless RoCE fiber ring with
  <a href="https://github.com/ashhart/TensorFold">TensorFold</a>:
  a 4-rank generalization of the 2-rank engine, now rebased onto v0.6.2 with our cache and prefill fixes.</sub>
</p>

---

Serve [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
(EXL3 TR3 4bpw, ~176 GB, uncensored/abliterated) from **four** NVIDIA DGX Sparks (GB10, 128 GB unified memory each)
as one tensor-parallel engine — **TensorFold TP4** — through an OpenAI-compatible API with the model's full
1,048,576-token context, DFlash2 speculative decoding, tool calling and `/metrics`.

TensorFold ships a two-rank CUDA engine for GLM-5.3-Flash (see
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
for the reference 2× Spark deployment). The `glm5_next` family hard-codes `world = 2` throughout:
rank fan-out, all-gather receive buffers, the request "doorbell" keys, weight splitting, cache slots.
This repository contains two routes to four ranks:

- **The v0.6.2 line (current)** — TensorFold **v0.6.2** with all of Mia's 53 patches rebased onto it, our
  4-rank generalization ported in, plus fixes of our own (cross-session cache eviction, a prefill kernel
  patch), all built into a local image. Start here: [`docs/v062-rebase.md`](docs/v062-rebase.md).
- **The v0.6.0 overlay (historical)** — a ~40-edit bind-mount overlay over Mia's published image,
  kept in `overlay/` with its full pitfall dossier:
  [`docs/tp4-world-generalization.md`](docs/tp4-world-generalization.md).

No switch is needed: the four Sparks are cabled in a ring with their ConnectX-7 QSFP ports, and NCCL
(a patched, ring-only build) talks RoCE over the cables directly.

## Credits and provenance

Almost everything that makes this work was built by others. Please read [CREDITS.md](CREDITS.md) for the
full chain; the short version:

| Piece | Source |
|---|---|
| Model GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai |
| Uncensored FP8 base of the checkpoint | [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) |
| EXL3 4bpw checkpoint we serve | [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) (ShapleyMcg License 1.0, attribution required — see [CREDITS.md](CREDITS.md)) |
| Inference engine | [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (Apache-2.0; the unmodified v0.6.0 source tree ships in `baseline/tensorfold/` with its upstream `LICENSE` / `NOTICE` / `THIRD_PARTY_NOTICES.md`; the current line is a v0.6.2 rebase) |
| 2× Spark TP2 stack, container image, 53 patches | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) — the v0.6.0 route ran her published image with our overlay bind-mounted on top; the v0.6.2 route rebuilds her stack per her `prepare.sh` |
| Switchless NCCL 2.30.7 build | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) (Alex Ellis, OpenFaaS Ltd) on [SparkRing](https://github.com/FujitsuPolycom/sparkring) patches; prior art [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0 — non-commercial) |
| TP4 reference & sequence-parallel design | [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) by drowzeys (a separate `glm_moe_dsa` family for the 753B model — not portable as code to `glm5_next`, but its `reduce_scatter` and design docs informed ours; see [docs/prefill-tuning.md](docs/prefill-tuning.md)) |
| Development | The stack was developed with the [Hermes](https://hermes-agent.nousresearch.com) agent, and cross-checked daily against our local production GLM-5.3-Flash **NVFP4** vLLM TP4 stack serving on the same four Sparks (also our decode baseline below) |

The overlay, rebase work, scripts and documentation in this repository are ours (Apache-2.0, see
[LICENSE](LICENSE)); `baseline/` is verbatim TensorFold v0.6.0 and keeps its own licenses. We are not
affiliated with any of the parties above; bugs in the TP4 generalization, the rebase and the fixes are
ours, not theirs.

## Results

Measured on our fleet (4× DGX Spark, fiber ring, warm kernel caches; self-measured numbers, not a swept
benchmark suite — treat them as orientation, not a leaderboard):

| Item | Value |
|---|---|
| KV pool at start (current config) | **6,651,904 tokens** shared across streams (single-request cap 1M; `PARALLEL=12`, `CACHE_ENTRIES=128`, `PREFILL_ROWS=4096` — every knob's pool cost is itemized in [docs/v062-rebase.md](docs/v062-rebase.md) and the start script) |
| KV pool, v0.6.0 overlay route | 8,077,312 tokens (`CACHE_ENTRIES=64`, `PARALLEL=6`, `PREFILL_ROWS=2048`) |
| Prefill, cold, all-new content | **~800 tok/s** with our patches (vs ~700 tok/s native baseline; 7,615 tok → 806, 5,719 tok → 803) |
| Prefill reference | our vLLM TP4 NVFP4 stack on the same ring: ~2.0k tok/s — TensorFold TP4 prefill remains the weak side, decode is the strong side |
| Decode, single stream, prose | ≈ 80+ tok/s, ~**1.5×** our vLLM TP4 NVFP4 baseline (self-timed) |
| Prefix cache, shared system prompt | 2nd conversation hit 1,984/2,035 tokens (97.5%), TTFT 1.8 s → 0.2 s |
| Prefix cache, returning to a 5-turn-old conversation | **43,904 cached tokens in 7.94 s** (was: full 66 s re-prefill — see [docs/cache-eviction.md](docs/cache-eviction.md)) |

Known shape of the stack, so nobody is surprised:

- **Decode is the strong side; prefill is the weak side** (~2.5× behind vLLM TP4 NVFP4 on the same ring;
  closing it means kernel work, not configuration — see [docs/prefill-tuning.md](docs/prefill-tuning.md)).
- **Cross-session cache eviction is fixed** (this repository's fixes); multi-stream decode contention is
  physics, and heavy concurrent agents still stall each other — fine for a handful of interactive streams.
- **Prefix cache hits require a stable, shared system prompt** (kept-prompt semantics), and warm up over
  the first turns; no system message → 0% hits by design.
- The checkpoint is uncensored/abliterated; behavior is the checkpoint's, not the engine's.

## The v0.6.2 line (current)

`docs/v062-rebase.md` is the full dossier; the short version:

1. **Rebase**: TensorFold v0.6.1/0.6.2 overhauled the engine; all of Mia's 53 patches were re-applied to a
   clean v0.6.2 tree (51 applied, 2 superseded/dropped with evidence), plus the 4-rank generalization
   ported in. `git merge-file --diff3` against the production tree as the systematic port check; four
   static verification gates + a real boot as the final arbiter (`py_compile` passing means nothing).
2. **Cache eviction fixes**: three kept-prompt eviction paths destroyed *other* sessions' cached state —
   an idle few minutes cost a 26–66 s full re-prefill even with nobody competing for memory. Fixed at the
   engine level and in the extent manager; verified with a controlled A→B→A replay (66 s → 7.94 s).
   [`docs/cache-eviction.md`](docs/cache-eviction.md)
3. **Prefill**: one-shot sparse (bit-exact), a small kernel patch lifting the GB10 shared-memory row
   ceiling (2,812 → 4,096+ rows per chunk via an L2 fallback), and the concurrency policy constant raised
   6 → 12. The hyper-connection row-split we *removed* after our W4 port produced deterministic garbage
   on long prompts despite passing every check we had — full postmortem in
   [`docs/prefill-tuning.md`](docs/prefill-tuning.md).

## What the overlay changes (v0.6.0 route)

`overlay/tensorfold/` binds over the image's `/opt/tensorfold` (9 files, MANIFEST-checked):

| File | Change |
|---|---|
| `cli_args.py` | `--tp` choices (1,2) → (1,2,4) |
| `cli.py` | master validation `tp > 1`; worker branch `rank == 1` → `rank > 0`; banner `rank of N`; `--rank` choices (0,1,2,3) |
| `families/glm5_next/__init__.py` | TP gate `!= 2` → `not in (1,2,4)`; pass `world = tp` into the engine |
| `cuda/engine.py` | `NCCL(rank, world)`; every geometry/split/admit call carries `world`; per-follower doorbell keys; `PARALLEL_MOST` 4 → 6 |
| `cuda/weights.py` | `load(world=)` + divisibility guards for heads / lin_heads / vocab / expert tiles at world 4 |
| `cuda/split.py` | row-span fast path and byte/device splitting fully world-parameterized |
| `cuda/hcsplit.py` | explicit error for world > 2 (the HC split is a 2-rank mechanism; we disable it at TP4) |
| `cuda/multi.py` | rank-0 async share / all-gather receive buffers `2 *` → `world *` |
| `cuda/segments.py` | `MAX_SEGS` 4 → 6 |

Note on parallelism: stock TensorFold v0.6.0 serves GLM-5.3-Flash **one request at a time** (its README:
"GLM and Nemotron CUDA serve one request at a time"). Multi-stream concurrency for GLM is Mia's patch
family (0030 multi-stream engine, 0040 deadlock fixes, 0041 ring base, 0049 multi-prefill, plus the
multi-KDA/DFlash2/DSA kernel paths), capped at `PARALLEL_MOST = 4`. Our overlay only raises that cap to 6.

Three failures cost us two boots each before the overlay stack came up; they are documented in detail with
the exact symptoms in [docs/tp4-world-generalization.md](docs/tp4-world-generalization.md):

1. **The request doorbell must be per-follower** (`tf_glm_request_<rank>_<n>`): the stock engine has exactly
   one follower, so one shared key worked; with three followers they raced to delete the same key and the
   request hung forever with only the master's GPU busy (stuck in the all-gather, followers still waiting
   on the doorbell — that's the diagnostic signature).
2. **Rank 0 must be started with `--host 0.0.0.0 --port 8000 --name`**: TensorFold's `serve` default is
   `127.0.0.1:8080`, which binds fine inside the container and serves nothing.
3. **Every all-gather receive buffer must be sized by `world`, not hard-coded `2`** — including the one in
   the startup calibration path, which dies first with a one-line `all_gather: recv must hold world x send`
   and no traceback. A static self-check for stragglers is in the docs.

## Repository layout

```
baseline/tensorfold/     TensorFold v0.6.0 source tree, verbatim, with its upstream LICENSE/NOTICE/THIRD_PARTY_NOTICES.md
overlay/tensorfold/      the v0.6.0-route TP4 world-generalization (bind-mount over the image's /opt/tensorfold) + MANIFEST.md5
scripts/
  start-glm53-flash-tf-unc-tp4-v062.sh   current line: worker-first bring-up of the v0.6.2 image (port check, memory gate, smoke test, full knob set in comments)
  stop-glm53-flash-tf-unc-tp4-v062.sh    master-first stop for the current line
  start-glm53-flash-tf-unc-tp4.sh        v0.6.0 overlay route: overlay md5 preflight + the same safety rails
  stop-glm53-flash-tf-unc-tp4.sh         master-first stop (overlay route)
  distribute-to-0102.sh                  idempotent image (ctr export/import) + shard distribution to fresh nodes
  smoke-tp4.sh                           one-shot chat smoke against the live endpoint
  tf_nccl_diag.py                        NCCL-only world=4 all-gather probe (run this before a full boot)
  verify_overlay_in_image.py             13 assertions that the overlay + mounts are coherent inside the image
  verify_cli_tp4.py                      argparse + family-gate check (must really parse, not just grep)
docs/
  tp4-world-generalization.md   overlay-route dossier: doorbell/host/recv-buffer/NCCL env/cache tuning
  v062-rebase.md                current line: the v0.6.2 rebase, verification gates, porting method
  cache-eviction.md             cross-session kept-prompt eviction: symptoms, root causes, fixes, A/B proof
  prefill-tuning.md             prefill knobs, the chunk-size kernel patch, the HC-split postmortem
```

## Requirements

- **Four DGX Sparks** (GB10, 128 GB unified memory each), cabled **ring-style** with their ConnectX-7 QSFP
  ports (each node ↔ both neighbors; 4 cables total), each link on its own /24 RoCE subnet, MTU 9000.
  Every rank needs ~100 GiB free unified memory at start.
- **Patched switchless NCCL 2.30.7** for sm_121/arm64, built per
  [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl); the library is injected with
  `TF_NCCL_LIB=/opt/patched-nccl/libnccl.so.2` plus the ring-only NCCL environment (the full set we use is
  in `scripts/start-glm53-flash-tf-unc-tp4-v062.sh`; validate any fabric change with
  `scripts/tf_nccl_diag.py` first — it runs one world=4 all-gather without loading the model).
- **The engine image**:
  - v0.6.0 overlay route: Mia's public image
    `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` (TensorFold v0.6.0 + her 53 patches
    prebuilt) with `overlay/tensorfold/` bind-mounted on top. This repository does not fork or re-tag it.
  - v0.6.2 line: built locally from a clean v0.6.2 tree per her `prepare.sh` recipe, with the rebased
    patches and our fixes applied — the start script's header documents the knob set and the image chain.
- **Weights on every node**, same path: the
  [checkpoint](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) (~176 GB, 92 shards; each
  rank memory-maps its byte range directly, no pre-splitting) and the
  [DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Quick start

Edit the node list at the top of the start script (it ships with our internal addresses as working
defaults — four lines, rank 0 first), then:

```bash
# on rank 0 (the API host), after the image + weights + patched NCCL are on all four nodes:
bash scripts/start-glm53-flash-tf-unc-tp4-v062.sh   # current line — workers first (rank 3 → 2 → 1), then rank 0
# or, for the v0.6.0 overlay route:
bash scripts/start-glm53-flash-tf-unc-tp4.sh
```

The script refuses to start if :8000 is taken (it will not stop whatever is already serving — *you* pick
the maintenance window), gates on free memory per node, brings ranks up worker-first, and runs a chat
smoke test before declaring victory. The stop scripts run master-first.

Served model id: `glm53-flash-tf-unc-tp4-v062` (current line) or `glm53-flash-tf-unc-tp4` (overlay route)
on `http://<rank0>:8000/v1`.

To fold the world-generalization into your own fork instead of the quick-start scripts: bind-mount
`overlay/tensorfold` over the image's `/opt/tensorfold` (keep the `tensorfold` path layer — mounting the
parent directory directly fails with "not a directory"), set the NCCL environment, and start rank 0 with
`--host 0.0.0.0 --port 8000`. For the v0.6.2 line the changes live in the image; the start script's
header is the authoritative knob list.

## Cache and memory tuning (differs from vLLM intuition)

- `TF_GLM_CACHE_GIB` is a **cap, not an allocation**. To let the pool auto-size from free memory, set it
  large (we use `1000`): the engine computes `min(cap, MemAvailable − RESERVE − weights)`. Setting it small
  silently strangles the pool.
- `TENSORFOLD_MEMORY_RESERVE_GIB` is the only real memory knob (upstream default 14.5 GiB, sized for
  1M-token prompt peaks).
- `TF_GLM_CACHE_ENTRIES` trades pool tokens for more retained conversations: 64 → 3.5 GiB reserved,
  128 → 7.1 GiB, 256 → 14.2 GiB (at `slot_bytes ≈ 69 MiB`, ~57 MiB per entry). We run 128.
- `PARALLEL` is nearly free: 6 → 12 costs ~0.6 GiB of pool.
- **Prefix cache hits require a stable, shared system prompt** (kept-prompt semantics), not arbitrary
  common prefixes à la vLLM. No system message → 0% hits by design. Monitor with
  `curl :8000/metrics | grep tensorfold_health:`. Cross-session eviction behavior is documented (and
  fixed) in [docs/cache-eviction.md](docs/cache-eviction.md).

## License

- This repository's overlay, rebase work, scripts and docs: **Apache-2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)).
- `baseline/tensorfold/`: TensorFold v0.6.0 under its own Apache-2.0 (and MIT for pre-v0.6.0 code), kept
  verbatim with its upstream license files.
- Checkpoint: ShapleyMcg License 1.0 (attribution required — carried in [CREDITS.md](CREDITS.md)).
- DFlash2 drafter: CC BY-NC-ND 4.0 (non-commercial, no derivatives) — download from its source, we do not
  redistribute it.
- Model weights are never stored in this repository; scripts download/reference them from Hugging Face.
