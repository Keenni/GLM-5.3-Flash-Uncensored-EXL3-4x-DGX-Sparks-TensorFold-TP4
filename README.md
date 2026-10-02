<h1 align="center">GLM-5.3-Flash Uncensored EXL3 on 4× DGX Spark — TensorFold TP4</h1>

<p align="center">
  <sub>A 4-rank overlay for <a href="https://github.com/ashhart/TensorFold">TensorFold</a> v0.6.0:
  generalizing its GLM-5.3-Flash engine from 2 to 4 DGX Sparks over a switchless RoCE fiber ring.</sub>
</p>

---

Serve [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
(EXL3 TR3 4bpw, ~176 GB, uncensored/abliterated) from **four** NVIDIA DGX Sparks (GB10, 128 GB unified memory each)
as one tensor-parallel engine — **TensorFold TP4** — through an OpenAI-compatible API with the model's full
1,048,576-token context, DFlash2 speculative decoding, tool calling and `/metrics`.

TensorFold v0.6.0 ships a two-rank CUDA engine for GLM-5.3-Flash (see
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
for the reference 2× Spark deployment). The `glm5_next` family hard-codes `world = 2` throughout:
rank fan-out, all-gather receive buffers, the request "doorbell" keys, weight splitting, cache slots.
This repository is a small overlay (~40 edits across 9 files) that generalizes that engine to
`world ∈ {1, 2, 4}`, plus the four-node start/stop/distribute tooling and the NCCL environment that makes
TensorFold run over a **switchless** (no-switch, direct-cable fiber ring) RoCE fabric.

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
| Inference engine | [TensorFold](https://github.com/ashhart/TensorFold) v0.6.0 by Ash Hart (Apache-2.0; the unmodified source tree ships in `baseline/tensorfold/` with its upstream `LICENSE` / `NOTICE` / `THIRD_PARTY_NOTICES.md`) |
| 2× Spark TP2 stack, container image, 53 patches | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) — our deployment runs her published image `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` (image id `22789f0cb3dc` at the time of writing) with this overlay bind-mounted on top |
| Switchless NCCL 2.30.7 build | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) (Alex Ellis, OpenFaaS Ltd) on [SparkRing](https://github.com/FujitsuPolycom/sparkring) patches; prior art [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0 — non-commercial) |
| TP4 idea cross-check | [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) by drowzeys (a separate `glm_moe_dsa` family; not used here, only the `--tp4` CLI/comm approach was read) |
| Development | The overlay was developed with the [Hermes](https://hermes-agent.nousresearch.com) agent, and cross-checked daily against our local production GLM-5.3-Flash **NVFP4** vLLM TP4 stack serving on the same four Sparks (also our decode baseline below) |

The overlay, scripts and documentation in this repository are ours (Apache-2.0, see [LICENSE](LICENSE));
`baseline/` is verbatim TensorFold v0.6.0 and keeps its own licenses. We are not affiliated with any of the
parties above; bugs in the TP4 generalization are ours, not theirs.

## Results

Measured on our fleet (4× DGX Spark, fiber ring, warm kernel caches, single boot 2026-10-02; self-measured
numbers, not a swept benchmark suite — treat them as orientation, not a leaderboard):

| Item | Value |
|---|---|
| KV pool at start | **8,077,312 tokens** shared across streams (`CACHE_GIB=1000` auto-size, `CACHE_ENTRIES=64`, `RESERVE=14.5`, ctx 1M, parallel 6) |
| Cold boot to first reply | 117 s (warm kernel cache) |
| Prefill, cold, all-new content | 8.2k tok → 6.65 s; 13.7k tok → 10.8 s (~1.25k tok/s) |
| Prefill reference | our vLLM TP4 NVFP4 stack on the same ring: ~2.0k tok/s → **TensorFold TP4 prefill ≈ 35% slower** |
| Decode, single stream, prose | ≈ 80+ tok/s, ~**1.5×** our vLLM TP4 NVFP4 baseline (self-timed) |
| Prefix cache, shared system prompt | 2nd conversation hit 1,984/2,035 tokens (97.5%), TTFT 1.8 s → 0.2 s |

Known shape of the stack, so nobody is surprised:

- **Decode is the strong side; prefill is the weak side** (~35% behind vLLM TP4 NVFP4 on the same ring).
- **Decode throughput under concurrency falls off faster than our vLLM stack** — fine for 1–2 interactive
  streams, not the right tool for throughput farming.
- **Prefix cache warms up late in real agent traffic**: the first 2–3 turns of a conversation don't hit it;
  hits start around the third turn. Not a pool-size issue — an 8M-token pool behaves the same (details in
  the [dossier](docs/tp4-world-generalization.md)).
- **Concurrent agents contend hard**: with several agents at once, GPU time goes to prefill and decode
  stalls. Single-user / low-concurrency is the sweet spot today; for multi-tenant serving, vLLM or SGLang
  on the same hardware remain the safer choice.
- The checkpoint is uncensored/abliterated; behavior is the checkpoint's, not the engine's.

## What the overlay changes

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

Three failures cost us two boots each before the stack came up; they are documented in detail with the
exact symptoms in [docs/tp4-world-generalization.md](docs/tp4-world-generalization.md):

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
overlay/tensorfold/      our TP4 world-generalization (bind-mount over the image's /opt/tensorfold) + MANIFEST.md5
scripts/
  start-glm53-flash-tf-unc-tp4.sh   worker-first bring-up: overlay md5 preflight, port check, memory gate, smoke test
  stop-glm53-flash-tf-unc-tp4.sh    master-first stop
  distribute-to-0102.sh             idempotent image (ctr export/import) + shard distribution to fresh nodes
  smoke-tp4.sh                      one-shot chat smoke against the live endpoint
  tf_nccl_diag.py                   NCCL-only world=4 all-gather probe (run this before a full boot)
  verify_overlay_in_image.py        13 assertions that the overlay + mounts are coherent inside the image
  verify_cli_tp4.py                 argparse + family-gate check (must really parse, not just grep)
docs/tp4-world-generalization.md   full pitfall dossier: doorbell/host/recv-buffer/NCCL env/cache tuning
```

## Requirements

- **Four DGX Sparks** (GB10, 128 GB unified memory each), cabled **ring-style** with their ConnectX-7 QSFP
  ports (each node ↔ both neighbors; 4 cables total), each link on its own /24 RoCE subnet, MTU 9000.
  Every rank needs ~100 GiB free unified memory at start.
- **Patched switchless NCCL 2.30.7** for sm_121/arm64, built per
  [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl); the library is injected with
  `TF_NCCL_LIB=/opt/patched-nccl/libnccl.so.2` plus the ring-only NCCL environment (the full set we use is
  in `scripts/start-glm53-flash-tf-unc-tp4.sh`; validate any fabric change with `scripts/tf_nccl_diag.py`
  first — it runs one world=4 all-gather without loading the model).
- **The container image**: `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` (Mia's public
  image, TensorFold v0.6.0 + her 53 patches prebuilt). You can also rebuild it yourself with her
  `scripts/prepare.sh` — this repository does not fork or re-tag it.
- **Weights on every node**, same path: the
  [checkpoint](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) (~176 GB, 92 shards; each
  rank memory-maps its byte range directly, no pre-splitting) and the
  [DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Quick start

Edit the node list at the top of `scripts/start-glm53-flash-tf-unc-tp4.sh` (it ships with our internal
addresses as working defaults — four lines, rank 0 first), then:

```bash
# on rank 0 (the API host), after the image + weights + patched NCCL are on all four nodes:
bash scripts/start-glm53-flash-tf-unc-tp4.sh        # workers first (rank 3 → 2 → 1), then rank 0
```

The script preflights overlay md5s on all four nodes, refuses to start if :8000 is taken (it will not stop
whatever is already serving), gates on free memory, brings ranks up worker-first, and runs a chat smoke
test before declaring victory. `scripts/stop-glm53-flash-tf-unc-tp4.sh` stops master-first.

Served model id: `glm53-flash-tf-unc-tp4` on `http://<rank0>:8000/v1`.

To fold this overlay into your own fork of Mia's stack instead of the quick-start scripts: bind-mount
`overlay/tensorfold` over the image's `/opt/tensorfold` (keep the `tensorfold` path layer — mounting the
parent directory directly fails with "not a directory"), set the NCCL environment, and start rank 0 with
`--host 0.0.0.0 --port 8000`.

## Cache and memory tuning (differs from vLLM intuition)

- `TF_GLM_CACHE_GIB` is a **cap, not an allocation**. To let the pool auto-size from free memory, set it
  large (we use `1000`): the engine computes `min(cap, MemAvailable − RESERVE − weights)`. Setting it small
  silently strangles the pool.
- `TENSORFOLD_MEMORY_RESERVE_GIB` is the only real knob (upstream default 14.5 GiB, sized for 1M-token
  prompt peaks).
- **Prefix cache hits require a stable, shared system prompt** (kept-prompt semantics), not arbitrary
  common prefixes à la vLLM. No system message → 0% hits by design. Monitor with
  `curl :8000/metrics | grep tensorfold_health:`.

## License

- This repository's overlay, scripts and docs: **Apache-2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)).
- `baseline/tensorfold/`: TensorFold v0.6.0 under its own Apache-2.0 (and MIT for pre-v0.6.0 code), kept
  verbatim with its upstream license files.
- Checkpoint: ShapleyMcg License 1.0 (attribution required — carried in [CREDITS.md](CREDITS.md)).
- DFlash2 drafter: CC BY-NC-ND 4.0 (non-commercial, no derivatives) — download from its source, we do not
  redistribute it.
- Model weights are never stored in this repository; scripts download/reference them from Hugging Face.
