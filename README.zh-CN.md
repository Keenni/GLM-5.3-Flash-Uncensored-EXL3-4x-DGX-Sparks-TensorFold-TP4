<h1 align="center">GLM-5.3-Flash Uncensored EXL3 跑在 4× DGX Spark 上 — TensorFold TP4</h1>

<p align="center">
  <sub><a href="README.md">English</a> | 简体中文<br><br>
  用四台 DGX Spark 经无交换机 RoCE 光纤环，以
  <a href="https://github.com/ashhart/TensorFold">TensorFold</a> 服务 GLM-5.3-Flash：
  2-rank 引擎的 4-rank 推广，现已 rebase 到 v0.6.2 并带上我们的缓存与 prefill 修复。</sub>
</p>

---

用**四台** NVIDIA DGX Spark（GB10，各 128 GB 统一内存）把
[neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
（EXL3 TR3 4bpw，约 176 GB，去审查/abliterated 版）作为单个张量并行引擎跑起来——**TensorFold TP4**——
走 OpenAI 兼容 API，带模型完整的 1,048,576 token 上下文、DFlash2 投机解码、工具调用与 `/metrics`。

TensorFold 自带 GLM-5.3-Flash 的双 rank CUDA 引擎（2× Spark 参考部署见
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)）。
`glm5_next` family 通篇写死 `world = 2`：rank 扇出、all-gather 接收缓冲、请求「门铃」key、权重切分、缓存槽。
本仓库包含通向 4 rank 的两条路线：

- **v0.6.2 线（现役）**——TensorFold **v0.6.2** + Mia 全部 53 个补丁 rebase 其上 + 我们的 4-rank
  推广移植 + 我们自己的修复（跨会话缓存淘汰、prefill kernel 补丁），全部烘进本地镜像。入口：
  [`docs/v062-rebase.md`](docs/v062-rebase.md)。
- **v0.6.0 overlay（历史路线）**——在 Mia 发布的镜像上叠约 40 处修改的 bind-mount overlay，
  连同完整踩坑档案存于 `overlay/`：
  [`docs/tp4-world-generalization.md`](docs/tp4-world-generalization.md)。
- **v0.6.7 线的 spill 层，为 `--parallel` 接好线**——上游 #423 只在单流路径落盘/回读；
  `--parallel 12` 下淘汰会整只丢掉已保留 prompt（30 小时 0 字节）。两文件 overlay
  （`overlay-v067/`），两条栈通用：
  [`docs/spill-parallel-wiring.md`](docs/spill-parallel-wiring.md)。

不需要交换机：四台 Spark 用 ConnectX-7 QSFP 口接成环，NCCL（打了补丁的 ring-only 构建）
直接在线上跑 RoCE。

## 致谢与出处

让这套东西能跑起来的几乎所有工作都是别人做的。完整链条见 [CREDITS.md](CREDITS.md)；简版：

| 部件 | 来源 |
|---|---|
| 模型 GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)（Z.ai） |
| checkpoint 的去审查 FP8 基座 | [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) |
| 我们服务的 EXL3 4bpw checkpoint | [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)（ShapleyMcg License 1.0，需署名——见 [CREDITS.md](CREDITS.md)） |
| 推理引擎 | [TensorFold](https://github.com/ashhart/TensorFold)，作者 Ash Hart（Apache-2.0；未改动的 v0.6.0 源码树随 `baseline/tensorfold/` 原样附带，含上游 `LICENSE` / `NOTICE` / `THIRD_PARTY_NOTICES.md`；现役线为 v0.6.2 rebase） |
| 2× Spark TP2 栈、容器镜像、53 个补丁 | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)——v0.6.0 路线跑她发布的镜像加我们的 overlay；v0.6.2 路线按她的 `prepare.sh` 重建 |
| Switchless NCCL 2.30.7 构建 | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)（Alex Ellis，OpenFaaS Ltd），基于 [SparkRing](https://github.com/FujitsuPolycom/sparkring) 补丁；先例 [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC BY-NC-ND 4.0——非商业） |
| TP4 参考与序列并行设计 | drowzeys 的 [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159)（为 753B 模型单开 `glm_moe_dsa` family——代码不可直接移植到 `glm5_next`，但其 `reduce_scatter` 与设计文档给了我们参照；见 [docs/prefill-tuning.md](docs/prefill-tuning.md)） |
| 开发过程 | 本栈由 [Hermes](https://hermes-agent.nousresearch.com) agent 开发，每天与同一批 Spark 上服役的生产 GLM-5.3-Flash **NVFP4** vLLM TP4 栈交叉对拍（也是下文 decode 基线） |

本仓库的 overlay、rebase 工作、脚本与文档归我们所有（Apache-2.0，见
[LICENSE](LICENSE)）；`baseline/` 是原样 TensorFold v0.6.0，保留其自身许可。我们与上述各方均无隶属
关系；TP4 推广、rebase 与各处修复里的 bug 归我们，不归他们。

## 结果

我们机群实测（4× DGX Spark 光纤环，kernel 缓存已热；自测数字，非扫参数的基准套件——当参照看，
别当排行榜）：

| 项目 | 数值 |
|---|---|
| 启动时 KV 池（现役配置） | **6,238,208 token**，跨流共享（单请求上限 1M；`PARALLEL=12`、`CACHE_ENTRIES=128`、`PREFILL_ROWS=8192`——每个旋钮的池代价在 [docs/v062-rebase.md](docs/v062-rebase.md) 与启动脚本里逐项列明） |
| KV 池，v0.6.0 overlay 路线 | 8,077,312 token（`CACHE_ENTRIES=64`、`PARALLEL=6`、`PREFILL_ROWS=2048`） |
| Prefill，冷启动全新材料 | **约 1,600 tok/s**（带我们的补丁；原生基线约 700 tok/s；16k prompt → 1,575 tok/s / 10.16 秒；8,192 行 chunk → 1,654 tok/s） |
| Prefill 参照 | 同环上我们 vLLM TP4 NVFP4 栈：约 2.0k tok/s——经 v19 prompt-kernel 恢复与修好的 HC row-split 后，差距收进约 1.2×（[docs/prefill-tuning.md](docs/prefill-tuning.md)） |
| Decode，单流 prose | 约 80+ tok/s，约为我们 vLLM TP4 NVFP4 基线的 **1.5×**（自测） |
| Prefix cache，共享 system prompt | 第二次会话命中 1,984/2,035 token（97.5%），TTFT 1.8 秒 → 0.2 秒 |
| Prefix cache，回到 5 轮前的会话 | **43,904 cached token 7.94 秒**（此前：66 秒全量重 prefill——见 [docs/cache-eviction.md](docs/cache-eviction.md)） |

先把栈的已知形态说清楚，免得有人意外：

- **decode 仍是强项；prefill 已追平大半**（同环约 1.6k vs 约 2.0k tok/s）。两个关键胜利都在 kernel 层：
  恢复 Mia 的 v19 prompt-expert kernel 用于 MoE prefill（主导 block 上 7.2×），以及根因修毕、
  逐位一致的 W4 超连接 row-split——见 [docs/prefill-tuning.md](docs/prefill-tuning.md)。
- **跨会话缓存淘汰已修**（本仓库的修复）；多流 decode 争抢是物理规律，高并发 agent 仍会互相拖慢——
  少量交互流没问题。
- **Prefix cache 命中需要稳定且共享的 system prompt**（kept-prompt 语义），且要几轮才热起来；
  无 system 消息 → 按设计 0%。
- checkpoint 是去审查/abliterated 版；行为差异来自 checkpoint，不来自引擎。

## v0.6.2 线（现役）

完整档案在 `docs/v062-rebase.md`；简版：

1. **Rebase**：TensorFold v0.6.1/0.6.2 重写了引擎；Mia 全部 53 个补丁重新落到干净 v0.6.2 树
   （51 个套用，2 个被取代/舍弃、附证据），再加 4-rank 推广移植。以 `git merge-file --diff3` 对生产树
   做系统性移植核对；四道静态验证门 + 一次真实启动做最终仲裁（`py_compile` 通过说明不了什么）。
2. **缓存淘汰修复**：四条 kept-prompt 淘汰路径会毁掉别的会话（或本会话自己）的缓存状态——
   闲置几分钟就吃 26–85 秒全量重 prefill，即使无人抢内存。在引擎层与 extent manager 修掉
   （含碎片池的 compact 前先淘汰）；用受控 A→B→A 回放验证（66 秒 → 7.94 秒；一个 136k 会话在
   交错负载下零淘汰存活）。
   [`docs/cache-eviction.md`](docs/cache-eviction.md)
3. **Prefill 约 700 → 约 1,600 tok/s**，按收益排序：把 MoE prefill 路由回 Mia 的 v19
   prompt-expert kernel（rebase 时接到了共享的 decode 形 kernel——占 prefill 51% 的那个 block 上 7.2×）、
   根因修毕并重新启用的 W4 超连接 row-split（逐位一致，+16%）、one-shot sparse（逐位一致）、
   GB10 共享内存行上限抬升（经 L2 fallback 从 2,812 → 8,192 行）、并发策略常数 6 → 12。row-split
   第一轮**确实**在长 prompt 上产出过确定性乱码，尽管通过了当时所有检查——根因与复盘见
   [`docs/prefill-tuning.md`](docs/prefill-tuning.md)。

## overlay 改了什么（v0.6.0 路线）

`overlay/tensorfold/` bind 到镜像的 `/opt/tensorfold` 上（9 个文件，MANIFEST 校验）：

| 文件 | 改动 |
|---|---|
| `cli_args.py` | `--tp` choices (1,2) → (1,2,4) |
| `cli.py` | master 校验 `tp > 1`；worker 分支 `rank == 1` → `rank > 0`；banner `rank of N`；`--rank` choices (0,1,2,3) |
| `families/glm5_next/__init__.py` | TP 门 `!= 2` → `not in (1,2,4)`；向引擎传 `world = tp` |
| `cuda/engine.py` | `NCCL(rank, world)`；每处 geometry/split/admit 调用带上 `world`；per-follower 门铃 key；`PARALLEL_MOST` 4 → 6 |
| `cuda/weights.py` | `load(world=)` + world 4 下 heads / lin_heads / vocab / expert tiles 的整除守卫 |
| `cuda/split.py` | row-span 快路径与字节/设备切分完全 world 参数化 |
| `cuda/hcsplit.py` | world > 2 显式报错（HC split 是 2-rank 机制；TP4 下我们关掉它） |
| `cuda/multi.py` | rank-0 异步 share / all-gather 接收缓冲 `2 *` → `world *` |
| `cuda/segments.py` | `MAX_SEGS` 4 → 6 |

并发能力说明：原生 TensorFold v0.6.0 服务 GLM-5.3-Flash **一次一个请求**（其 README：
"GLM and Nemotron CUDA serve one request at a time"）。GLM 的多流并发是 Mia 的补丁家族
（0030 多流引擎、0040 死锁修复、0041 ring 基座、0049 multi-prefill，以及 multi-KDA/DFlash2/DSA
kernel 路径），上限 `PARALLEL_MOST = 4`。我们的 overlay 只把上限抬到 6。

overlay 栈起来之前，三个 failure 各坑了我们两次启动；症状细节全部记录在
[docs/tp4-world-generalization.md](docs/tp4-world-generalization.md)：

1. **请求门铃必须 per-follower**（`tf_glm_request_<rank>_<n>`）：原生引擎只有一个 follower，共享
   一个 key 没问题；三个 follower 抢删同一个 key，请求永久挂起且只有 master 的 GPU 在忙
   （卡在 all-gather，follower 还在等门铃——这就是诊断特征）。
2. **rank 0 必须以 `--host 0.0.0.0 --port 8000 --name` 启动**：TensorFold `serve` 默认
   `127.0.0.1:8080`，容器内绑定正常但外面什么都服务不到。
3. **每处 all-gather 接收缓冲都必须按 `world` 取尺寸，不能写死 `2`**——包括启动标定路径里的那处，
   它最先死，一行 `all_gather: recv must hold world x send`，无 traceback。静态自查脚本见文档。

## v0.6.7 统一镜像（现役生产）

两条线现在跑同一个自打补丁的镜像 `tensorfold-glm53:v0.6.7`（上述 v0.6.2 rebase 谱系，外加：
内存 janitor 守护线程、被淘汰 kept-prompt 的 spill 层、system-block checkpoint、烘进去的
vision prefix-reuse 修复、前后台分流的 `--alias`/`--name-priority`）——**无 bind-mount overlay**。
启停脚本在 `scripts/`：

- `start-glm53-flash-tf-unc-tp4-v067.sh` / `stop-...`——TP4 生产对（master 上跑一次；脚本按
  worker-first 起全部 4 个 rank，不碰任何其他容器）。
- 同镜像上的 TP2 生产对在姊妹仓库
  [GLM-5.3-Flash-Uncensored-EXL3-2x-DGX-Sparks-TensorFold-TP2](https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-2x-DGX-Sparks-TensorFold-TP2)
  ——连同 world=2 row-split 根因、TP2 调优 A/B 表、直连 CX7 的 2-rank NCCL 配置。

⚠️ v0.6.7 同 tag 在多机改过一台不改另一台就会内容分叉：逐文件用
`docker run --rm --entrypoint md5sum <image> <path>` 核，别信 tag。

## 仓库结构

```
baseline/tensorfold/     TensorFold v0.6.0 源码树，原样，含上游 LICENSE/NOTICE/THIRD_PARTY_NOTICES.md
overlay/tensorfold/      v0.6.0 路线的 TP4 world 推广（bind 到镜像 /opt/tensorfold）+ MANIFEST.md5
scripts/
  start-glm53-flash-tf-unc-tp4-v062.sh   现役线：v0.6.2 镜像 worker-first 启动（端口检查、内存门、冒烟测试、全量旋钮在注释里）
  stop-glm53-flash-tf-unc-tp4-v062.sh    现役线 master-first 停止
  start-glm53-flash-tf-unc-tp4.sh        v0.6.0 overlay 路线：overlay md5 preflight + 同一套安全栏
  stop-glm53-flash-tf-unc-tp4.sh         master-first 停止（overlay 路线）
  distribute-to-0102.sh                  幂等镜像（ctr export/import）+ 分片向新节点分发
  smoke-tp4.sh                           对存活端点的一次性 chat 冒烟
  tf_nccl_diag.py                        仅 NCCL 的 world=4 all-gather 探针（完整启动前先跑这个）
  verify_overlay_in_image.py             13 条断言：overlay + 挂载在镜像内自洽
  verify_cli_tp4.py                      argparse + family 门检查（要真的能 parse，不是 grep 到就算）
docs/
  tp4-world-generalization.md   overlay 路线档案：门铃/host/接收缓冲/NCCL env/缓存调优
  v062-rebase.md                现役线：v0.6.2 rebase、验证门、移植方法
  cache-eviction.md             跨会话 kept-prompt 淘汰：症状、根因、修法、A/B 实证
  prefill-tuning.md             prefill 旋钮、chunk-size kernel 补丁、HC-split 复盘
```

## 硬件与依赖

- **四台 DGX Spark**（GB10，各 128 GB 统一内存），ConnectX-7 QSFP 口接成**环**
  （每节点 ↔ 两邻居；共 4 根线），每条链路独立 /24 RoCE 网段，MTU 9000。
  每个 rank 启动时需约 100 GiB 空闲统一内存。
- **打了补丁的 switchless NCCL 2.30.7**（sm_121/arm64），按
  [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) 构建；以
  `TF_NCCL_LIB=/opt/patched-nccl/libnccl.so.2` 注入，外加 ring-only NCCL 环境变量
  （我们用的全集在 `scripts/start-glm53-flash-tf-unc-tp4-v062.sh`；任何 fabric 变更先用
  `scripts/tf_nccl_diag.py` 验证——它不加载模型就跑一次 world=4 all-gather）。
- **引擎镜像**：
  - v0.6.0 overlay 路线：Mia 的公开镜像
    `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold`（TensorFold v0.6.0 + 她
    预编译的 53 个补丁），上叠 `overlay/tensorfold/`。本仓库不 fork、不重打 tag。
  - v0.6.2 线：按她 `prepare.sh` 的配方从干净 v0.6.2 树本地构建，套上 rebase 补丁与我们的修复——
    启动脚本头部记录了旋钮集与镜像链。
- **每台节点上都有权重**，同路径：[checkpoint](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
  （约 176 GB，92 个分片；每个 rank 直接 mmap 自己的字节区间，无需预切分）与
  [DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)。

## 快速开始

改启动脚本顶部的节点表（出厂带我们的内网地址当可用默认——四行，rank 0 在前），然后：

```bash
# 在 rank 0（API 宿主）上，镜像 + 权重 + 补丁 NCCL 四台就位后：
bash scripts/start-glm53-flash-tf-unc-tp4-v062.sh   # 现役线——先 worker（rank 3 → 2 → 1），再 rank 0
# 或走 v0.6.0 overlay 路线：
bash scripts/start-glm53-flash-tf-unc-tp4.sh
```

:8000 被占时脚本拒绝启动（它不会停掉已在服务的进程——维护窗口**你**来定），按节点空闲内存设门，
worker-first 起全部 rank，宣布成功前先跑 chat 冒烟。stop 脚本 master-first。

服务模型名：`glm53-flash-tf-unc-tp4-v062`（现役线）或 `glm53-flash-tf-unc-tp4`（overlay 路线），
端点 `http://<rank0>:8000/v1`。

想把 world 推广折进你自己的 fork 而不用快速开始脚本：把 `overlay/tensorfold` bind 到镜像的
`/opt/tensorfold` 上（保留 `tensorfold` 路径层——直接挂父目录会报 "not a directory"），设好 NCCL
环境变量，rank 0 以 `--host 0.0.0.0 --port 8000` 启动。v0.6.2 线的改动在镜像里；启动脚本头部是
权威旋钮清单。

## 缓存与内存调优（与 vLLM 直觉不同）

- `TF_GLM_CACHE_GIB` 是**上限，不是分配**。要让池按空闲内存自动取尺寸，把它设大（我们用 `1000`）：
  引擎算 `min(cap, MemAvailable − RESERVE − weights)`。设小了会静默掐死池。
- `TENSORFOLD_MEMORY_RESERVE_GIB` 是唯一真正的内存旋钮（上游默认 14.5 GiB，按 1M-token prompt
  峰值定标）。
- `TF_GLM_CACHE_ENTRIES` 用池 token 换更多保留会话：64 → 预留 3.5 GiB、128 → 7.1 GiB、256 →
  14.2 GiB（`slot_bytes ≈ 69 MiB`，每 entry 约 57 MiB）。我们跑 128。
- `PARALLEL` 几乎免费：6 → 12 只花约 0.6 GiB 池。
- **Prefix cache 命中需要稳定且共享的 system prompt**（kept-prompt 语义），不是 vLLM 那种任意
  公共前缀。无 system 消息 → 按设计 0%。监控：
  `curl :8000/metrics | grep tensorfold_health:`。跨会话淘汰行为记录（并已修）在
  [docs/cache-eviction.md](docs/cache-eviction.md)。

## 许可证

- 本仓库的 overlay、rebase 工作、脚本与文档：**Apache-2.0**（[LICENSE](LICENSE)、[NOTICE](NOTICE)）。
- `baseline/tensorfold/`：TensorFold v0.6.0 沿用其自身 Apache-2.0（v0.6.0 前代码为 MIT），原样保留
  其上游许可文件。
- checkpoint：ShapleyMcg License 1.0（需署名——载于 [CREDITS.md](CREDITS.md)）。
- DFlash2 drafter：CC BY-NC-ND 4.0（非商业、无演绎）——从源头下载，我们不转发。
- 模型权重不进本仓库；脚本从 Hugging Face 下载/引用。
