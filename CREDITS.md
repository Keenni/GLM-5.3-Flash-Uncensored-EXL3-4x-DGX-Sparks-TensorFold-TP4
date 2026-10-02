# Credits

This repository is a thin layer: a world-generalization overlay for TensorFold, four-node tooling, and
documentation. Almost everything that makes it work was built by others.

Our own work (the `overlay/` TP4 world-generalization, the `scripts/` four-node tooling, and the docs) is
licensed under the Apache License 2.0 ([LICENSE](LICENSE)); [NOTICE](NOTICE) carries the third-party notices
that go with it.

## Model

- **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** by [Z.ai](https://z.ai): the model's
  design, training and evaluations. Its license (on the model card) governs any use of the weights. The
  weights are not part of this repository.
- **[orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8)**:
  the uncensored/abliterated FP8 base of the checkpoint served here.
- **[neko-legends](https://huggingface.co/neko-legends)**: the EXL3 TR3 4bpw quantization of it,
  [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
  (~176 GB), made with **ShapleyMcg** by Brandon M. Music under the **ShapleyMcg License 1.0**
  (attribution required; the checkpoint's `LICENSE` file has the terms). Its attribution notice:

  > This work includes or was produced using ShapleyMcg, created by Brandon M. Music
  > (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an
  > attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without
  > this attribution is unlicensed.
- **[IncoAI](https://huggingface.co/incoai)**: the DFlash2 drafter,
  [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (**CC BY-NC-ND 4.0**: non-commercial use, no derivatives). Downloaded from its source, never
  redistributed here.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart))
  and the TensorFold contributors (Apache 2.0 from v0.6.0; releases up to v0.5.0 were MIT, and code written
  before v0.6.0 keeps its MIT notice): the engine that serves the model — the multi-rank CUDA engine for
  GLM-5.3-Flash, its EXL3 expert kernels, DFlash2/MTP drafting with exact verification, and the
  OpenAI-compatible server. The **unmodified v0.6.0 source tree ships verbatim in
  `baseline/tensorfold/`**, with its upstream `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` files. Our
  overlay in `overlay/tensorfold/` modifies 9 files of that tree; those modified files remain derivatives of
  TensorFold and keep its license boundary — our added changes are Apache-2.0.
- TensorFold itself builds on, and credits in its
  [third-party notices](baseline/tensorfold/THIRD_PARTY_NOTICES.md):
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), whose EXL3 format the routed
  experts are stored in; the GLM-5.3-Flash modeling code in Hugging Face
  [transformers](https://github.com/huggingface/transformers) (Apache 2.0); and
  [z-lab/dflash](https://github.com/z-lab/dflash) (Z Lab, MIT), whose DFlash2 architecture its drafter ports.

## The 2× Spark reference stack

- **[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
  by [Mia's AI Lab](https://x.com/MiaAI_lab): the working, production-grade 2× Spark deployment this project
  extends — TensorFold v0.6.0 plus her 53 patches (DFlash2 and copy drafts, 4-bit dense weights, FP8 KV
  cache, faster prompt kernels, one-shot RoCE all-gather, vision, tool calling, `/tokenize`, `/metrics`,
  and more), each with her own meticulous credit chain in her `CREDITS.md`.
- **Her published container image is what our deployment runs**:
  `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold` (TensorFold v0.6.0 + her patches
  prebuilt). This repository does not fork, rebuild or re-tag the image; it bind-mounts `overlay/tensorfold/`
  over the image's `/opt/tensorfold` at start.
- Our TP4 overlay touches 9 files, all of them Mia-patched files or TensorFold v0.6.0 files; her patches'
  logic is carried through unmodified except where the TP4 generalization requires it (doorbell keys,
  buffer sizing, rank gating — detailed in
  [docs/tp4-world-generalization.md](docs/tp4-world-generalization.md)).
- The 2-rank TP2 variant of this deployment ran on our fleet for a week before the TP4 work, and her
  choice of `PARALLEL_MOST` and segment counts was kept where it made sense at TP4.

## Switchless NCCL

Our deployments use the prebuilt library and recipes of
**[alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)** by Alex Ellis (OpenFaaS Ltd,
Apache-2.0). Its own provenance chain, from its `PROVENANCE.md`:

- **[NVIDIA NCCL](https://github.com/NVIDIA/nccl)** v2.30.7-1 (commit `73cf1122…`), Apache-2.0 with
  BSD-licensed portions.
- **[SparkRing](https://github.com/FujitsuPolycom/sparkring)** (FujitsuPolycom, Apache-2.0): the two
  switchless-cycle patches — skip Tree/PAT transport connect under `NCCL_SKIP_TREE_CONNECT`, and advertise
  all eligible listener GIDs so subnet-aware routing can pick the physically-cabled NIC.
- **OpenFaaS hardening patch**: strict opt-in for the switchless behaviour, diagnostics, and listener-GID
  validation, owned by Alex Ellis.
- **Prior art**: Joseph Rose's switchless skip-Tree/skip-PAT approach
  ([josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless)) — credited
  conceptually; no source from that repository is included anywhere here.
- The listener-GID patch extends NVIDIA's DGX Spark subnet-aware routing work (NCCL commit
  `5c1c4288…` by Zifu Yang).

We contributed **no NCCL code**; our contribution on this layer is the NCCL *environment* (the 23-variable
set that makes TensorFold's comm path work on a four-node switchless ring, validated by
`scripts/tf_nccl_diag.py`) and the diagnosis of why the naive subset of variables fails — documented in
[docs/tp4-world-generalization.md](docs/tp4-world-generalization.md).

## Ideas

- **[TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159)** by drowzeys: an independent
  TP4 route (a separate `glm_moe_dsa` family for a 753B-class model). We read its `--tp4` CLI/comm approach
  for orientation but did **not** port its code; our overlay generalizes the existing `glm5_next` family
  instead.
- The KV-pool sizing approach (cap-vs-reserve semantics) follows the upstream
  `TENSORFOLD_MEMORY_RESERVE_GIB` design; the pool numbers in our README are our measurements.

## Development

- The overlay, scripts and documentation were developed with the **[Hermes](https://hermes-agent.nousresearch.com)**
  agent (Nous Research), driven by a human operator.
- Development was cross-checked daily against a local production **GLM-5.3-Flash NVFP4 vLLM TP4** stack
  serving on the same four DGX Sparks (also the decode/prefill baseline quoted in the README). The vLLM
  stack is not part of this repository.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)** (the base
  of Mia's image), with NVIDIA's CUDA, cuDNN, cuBLAS, NCCL and related libraries. Governed by the NVIDIA
  Software License Agreement and the Product-Specific Terms for NVIDIA AI Products.
- **[PyTorch](https://pytorch.org/)** (BSD-3-Clause) and
  **[Triton](https://github.com/triton-lang/triton)** (MIT): the tensor/runtime and kernel language layers
  TensorFold's CUDA kernels compile and run on.

## Corrections

If we have misattributed anything above, please open an issue — attribution accuracy matters more to us
than vanity.
