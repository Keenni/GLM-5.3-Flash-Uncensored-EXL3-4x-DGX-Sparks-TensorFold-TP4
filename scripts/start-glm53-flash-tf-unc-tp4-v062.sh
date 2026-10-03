#!/usr/bin/env bash
# =====================================================================
# start-glm53-flash-tf-unc-tp4-v062.sh   ← TP4 生产权威入口(四台同文, md5 一致)
# GLM-5.3-Flash (neko-legends 去审查 EXL3, mcg codebook) · TensorFold v0.6.2 + Mia 53 rebase
#   + TP4 W4 泛化 + drowzeys EXL3 共享层移植 + 本仓两处 prefill 修复(全部已烧进镜像, 无 overlay 挂载)
# 基于现役 TP2 脚本(start-glm53-flash-tf-unc-tp2.sh)改编；NCCL 数据面身份注入逐字复用
# vLLM TP4 生产脚本(start-glm53-flash-nvfp4-tp4.sh, 2026-09-15 宪法)的实配 helpers：
#   权重平铺 /models 直读 / 端口 8000 / master 恒 dgx03 / worker-first 3→2→1→0 /
#   内存 gate / restart=no / refit 降窗重试 / 烟测验收 / 只清理自己的容器
#
# ── 当前档(2026-10-04 修订) ──
#   ★ prefill 实测: 37k 冷预填 48.2s → 26.8s (~1300 tok/s); 8192行chunk 793 → 1426 tok/s (+80%)
#     来源: 把 MoE 的 EXL3 路径接回 Mia 的 v19 prompt 内核(exl3_mm.routed) —— rebase 时被接到共享
#     decode 形状内核上了, 那块占 prefill 51%、实测换回后快 7.2x(见 skill §10.10)
#     新瓶颈 = 通信: moe: all-gather 23% + kda: out + all-gather 20% (≈43%)

#   par=12 · ENTRIES=128 · PREFILL_ROWS=8192 · SPARSE_ONEPASS=1 · FILL_SHARE=0.3
#   ★ HC_SPLIT=1 + HC_EXCHANGE=gather：W4 行切分已修好并验收(bit-exact + 提速, 见下)
#   → 池子 ~6.65M tok (单请求上限 1M; 随启动时 MemAvailable 抖动几千)
#   其余生效点: ① PREFILL_ROWS>2812 的行数上限解除(内核装不下时改读 L2, skill §10.5)
#              ② SPARSE_ONEPASS=1 (bit-exact)   ③ PARALLEL_MOST 6→12(§10.6, 代码里的政策上限)
#              ④ FILL_SHARE=0.3 (有解码在跑时预填让位; 空载不参与, 见 §10.7 附近说明)
#              ⑤ PREFILL_ROWS=8192 (池子因此降到 ~6.2M tok)
#   ✅ 缓存修复(tp4k, skill §9.5): multi.py 的 _room/_grow 改为「先无损 compact, 再淘汰 kept」
#      —— 原代码在池子碎片化时会先杀掉 kept(实测能白干 84.71s 重预填), 整理只做 MOVE 内容不动。
#      验证: 会话养到 13.6 万 tok → 插 3 个小请求 + 并行会话 → 回会话 cached=136832/136885 prefill=0.16s, EVICT=0。
#   HC_SPLIT(W4 行切分) 现役: 2026-10-04 修复根因(partial() 旧分支只填 1/4 行就 return, 正确版成死代码)
#     修后三条固定 prompt 的 sha 与不分片基准逐位一致(370501976a66 / a99b19fa58c8 / a99b19fa58c8)
#     (长 prompt 不听指令/续写噪声; 不报错、烟测也查不出)。四 rank 正确做法 = drowzeys
#     docs/design/glm-moe-dsa-prefill.md §10-02 的 reduce-scatter over rows + 自身行 glue
#     (仅 +8~12%), 要在 forward.py 的 reduce 点重写, 不是一个 flag。
#   🔍 TF_GLM_TRACE_KEEP=1 目前开着(临时诊断, 见下方 TF_ENVS): 每请求打 RESUME HIT/MISS、
#      ROOM placed/compacting、EVICT why=... 便宜, 用来抓「缓存再被扫」的确切原因;
#      长会话稳定温热几天后摘掉即可(删那一行 + 重启)。
#   验证纪律: 改完必须跑「可验证问答基准电池」(短/中/长 prompt 各一 + 结尾指令), 光看烟测 sha 不够。
#
# 依赖(四台都要): 镜像 tensorfold-glm53:v0.6.2-tp4 (本地构建, 四台 ID 各自不同内容相同)
#   /models/GLM-5.3-Flash-Uncensored-EXL3 (92 shard) + /models/GLM-5.3-Flash-DFlash2
#   /root/nccl-switchless/libnccl.so.2 (switchless 补丁 NCCL 2.30.7)
#   内核缓存 /root/tf-kernel-cache/glm53-tf-unc-tp4-v062 (含 exl3 linear_v5/experts_v2/prompt_experts_v1)
#
# 用法: 只在 dgx03 执行:  bash /root/start-glm53-flash-tf-unc-tp4-v062.sh
#       停:  bash /root/stop-glm53-flash-tf-unc-tp4-v062.sh   (master-first)
# 回滚:  镜像链 v0.6.0(旧生产) → v0.6.2-tp4(基座) → v0.6.2-tp4f(drowzeys 移植层)
#   实测: 8192 行 chunk 1426 → 1654 tok/s(+16%); 16k 冷预填 1575 tok/s(prefill 10.16s)
#   细节与下一步(重开 PREFILL_OVERLAP=2)见 skill glm53-flash-tf-tp4 / references/hc-split-w4-*.md
#        脚本改动前备份  start-glm53-flash-tf-unc-tp4-v062.sh.bak-prefill-20261004
# 红线: 本脚本不替用户停任何在跑生产容器；:8000 被占即 die 并提示先跑 TP2 的 stop 脚本。
# =====================================================================
set -euo pipefail

# ---------------- 配置区 ----------------
CONTAINER_NAME="glm53-flash-tf-unc-tp4-v062"
MODEL_DIR="/models/GLM-5.3-Flash-Uncensored-EXL3"
DRAFTER_DIR="/models/GLM-5.3-Flash-DFlash2"
PORT=8000                       # 集群铁律
MASTER_PORT=29551
MASTER_ADDR="192.168.89.103"    # rank0(dgx03) 管理网地址: TCPStore rendezvous + NCCL bootstrap
SERVED_NAME="glm53-flash-tf-unc-tp4-v062"
H1="root@192.168.89.103"        # dgx03 (rank0 + API, master)
W1="root@192.168.89.104"        # dgx04 (rank1)
W2="root@192.168.89.101"        # dgx01 (rank2)
W3="root@192.168.89.102"        # dgx02 (rank3)
IMAGE="tensorfold-glm53:v0.6.2-tp4"
KCACHE="/root/tf-kernel-cache/glm53-tf-unc-tp4-v062"   # 每机各自编译缓存(与 TP2 的目录分开)
NCCL_PATCH="/root/nccl-switchless"
NCCL_IN_CONTAINER="/opt/patched-nccl"
TF_PKG="/usr/local/lib/python3.12/dist-packages/tensorfold"   # 容器内 TF 包根
# Mia TP2 现役档全量透传；TP4 差异: COMM=nccl(补丁栈)。HC_SPLIT 已泛化到 W4 并验收(bit-exact)
CONTEXT=1048576
PARALLEL=12
KV=fp8
DENSE=q4
MAX_TOKENS=32768
DRAFT_POLICY="fnc7:0.3"
WAIT_TIMEOUT=3000               # 首启含四机并行 CUDA kernel 编译
MEM_GATE_GIB=85                 # 起栈前每机 MemFree 门槛 (权重 ~41GiB/rank + NCCL/图/池余量)
TF_ENVS=(
  HF_HUB_OFFLINE=1
  TF_GLM_KV=$KV
  TF_GLM_DENSE=$DENSE
  TF_GLM_COMM=nccl
  TF_GLM_COPY_DRAFTS=1
  TF_GLM_COPY_MAX=15
  TF_GLM_DFLASH_POLICY=$DRAFT_POLICY
  TF_GLM_HC_SPLIT=1
  TF_GLM_HC_EXCHANGE=gather
  TF_GLM_PREFILL_OVERLAP=0
  TF_GLM_KDA_CHUNKED=1
  TF_GLM_SPARSE_ONEPASS=1         # ✓已验证 bit-exact: 单遍 sparse MLA prompt(省 chunk 程序与合并)
  TF_GLM_PREFILL_ROWS=8192         # item 2 ✓: 曾爆分组内核 smem(4096行=147456B > GB10 101248B); 内核已改为
                                   # 装不下读 L2, 上限解除, 故从 4096 推到 8192(代价: 池子会缩, 看启动 estimate)
  TF_GLM_WIDE_GRAPHS=16
  TF_GLM_COPY_REPLY_MATCH=16
  TF_GLM_MULTI_LONE=1
  TF_GLM_MULTI_PREFILL=1
  TF_GLM_PROFILE=1                  # 逐 block 计时(带 device sync)+内存, 每个 prompt chunk 打一次:
                                    # 看 prefill profile / prefill memory 两行; 会拖慢计时, 优化完可摘
  TF_GLM_FILL_SHARE=0.3            # 有解码在跑时预填让位的比例(默认 0.5): 0.3 = 更保解码平滑;
                                   # 空载无解码时该值不参与(每轮都直接发预填, 不拖慢)
  TF_GLM_L2PF=1
  TF_GLM_EXL3_LOADS=nc
  TF_GLM_SHARED_PREFIX=1
  TF_GLM_CACHE_ENTRIES=128
  TF_GLM_TRACE_KEEP=1                  # 临时诊断: 打印 ROOM/EVICT/GROW/HOME-MOVE/TAKEOVER-SKIP 轨迹
  TF_GLM_CACHE_GIB=1000                # 不封顶: 池由 MemAvailable(启动时)-RESERVE 自动算, engine 夹到 (budget-estimate)
  TENSORFOLD_MEMORY_RESERVE_GIB=14.5   # 上游值: 1M prompt 峰值(比 estimate 多 ~10GiB)留 ~4.5GiB 余量
  TF_GLM_MTP=auto
  TENSORFOLD_NO_UPDATE_CHECK=1
)
# ---------------- 工具 ----------------
log()  { printf '[start-tf-tp4] %s\n' "$*"; }
warn() { printf '[start-tf-tp4] WARN: %s\n' "$*" >&2; }
die()  { printf '[start-tf-tp4] ERROR: %s\n' "$*" >&2; exit 1; }
h()    { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$1" "${@:2}"; }
free_gib() { h "$1" "LC_ALL=C free -g | awk '/^Mem:/{print \$4}'" 2>/dev/null || echo 0; }
port_busy() { h "$H1" "curl -sf --max-time 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1" && return 0 || return 1; }
dump_logs() {
  local host=$1
  log "── $host 最后日志 ──"
  h "$host" "docker logs --tail 25 $CONTAINER_NAME 2>&1" 2>/dev/null | sed 's/^/  │ /' || true
}
# ---------------- 前置校验 ----------------
log "── 前置校验 ──"
ENV_ARGS=(); for e in "${TF_ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864)
SERVE_ARGS=(--tp 4 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
            --drafter "$DRAFTER_DIR" --context "$CONTEXT" --parallel "$PARALLEL"
            --max-tokens "$MAX_TOKENS" --thinking --vision)

for host in "$H1" "$W1" "$W2" "$W3"; do
  h "$host" "test -f $MODEL_DIR/config.json" || die "$host 缺 $MODEL_DIR/config.json"
  n=$(h "$host" "ls $MODEL_DIR/*.safetensors 2>/dev/null | wc -l")
  [[ "$n" == "92" ]] || die "$host 权重 shard 数 $n != 92"
  h "$host" "test -f $DRAFTER_DIR/model.safetensors" || die "$host 缺 DFlash2 drafter"
  h "$host" "test -f $NCCL_PATCH/libnccl.so.2" || die "$host 缺补丁 NCCL $NCCL_PATCH/libnccl.so.2"
  h "$host" "docker image inspect $IMAGE >/dev/null 2>&1" || die "$host 缺镜像 $IMAGE"
  h "$host" "mkdir -p $KCACHE"
  free=$(free_gib "$host")
  if (( free < MEM_GATE_GIB )); then
    die "$host MemFree ${free}GiB < ${MEM_GATE_GIB}GiB：先停旧栈/清理再起 (docker ps 看在跑 LLM)"
  fi
  log "$host OK (shard=92, MemFree=${free}GiB)"
done
if port_busy; then
  die "dgx03 :$PORT 已被占用(现役栈还在跑, 2026-10-02 起为 vLLM TP4 容器 glm53-flash-tp4-vllm-glm53-1)：先四机 master-first 停它(dgx03 先 docker rm -f，再 04/01/02)再起 TP4"
fi
# 只清自己名字的残留容器（绝不碰 TP2/vLLM 生产容器）
for host in "$H1" "$W1" "$W2" "$W3"; do
  h "$host" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
done
log "preflight OK (四机文件/md5/内存 gate 通过, :8000 空闲)"

# ---------------- 每机 GID index 动态查 (tonyd2wild: 钉错=ibv_modify_qp 失败或 collective hang) ----------------
gid_of() {  # 在目标机上查第一个 RoCEv2/IPv4 GID 的 index, 查不到退 3
  h "$1" 'for dev in /sys/class/infiniband/*/ports/1; do
    [ -e "$dev/gids" ] || continue
    for idx in 0 1 2 3 4 5 6 7; do
      gid=$(cat "$dev/gids/$idx" 2>/dev/null) || continue
      case "$gid" in ""|0000:0000:0000:0000:0000:0000:0000:0000) continue;; esac
      t=$(cat "$dev/gid_attrs/types/$idx" 2>/dev/null) || continue
      case "$t" in *"v2"*) ;; *) continue;; esac
      case "$gid" in *ffff:*) echo "$idx"; exit 0;; esac
    done
  done; echo 3'
}
GID_H=$(gid_of "$H1"); GID_W1=$(gid_of "$W1"); GID_W2=$(gid_of "$W2"); GID_W3=$(gid_of "$W3")
log "GID index: dgx03=$GID_H dgx04=$GID_W1 dgx01=$GID_W2 dgx02=$GID_W3"

# ---------------- 启动 (worker-first 3→2→1→0) ----------------
launch() {  # $1=rank
  local rank=$1 host gid
  case $rank in
    0) host=$H1; gid=$GID_H ;;
    1) host=$W1; gid=$GID_W1 ;;
    2) host=$W2; gid=$GID_W2 ;;
    3) host=$W3; gid=$GID_W3 ;;
    *) die "bad rank $rank" ;;
  esac
  # NCCL 身份(逐字 vLLM TP4 宪法): bootstrap 走管理网 enP7s7, 数据面 2 个主链 HCA(只列 2 个!
  # 补丁 NCCL 硬校验恰好 2 个 listener GID), ADDR_RANGE 覆盖 edge1..4; TF_NCCL_LIB 指补丁栈
  # NCCL 段逐字对齐 vLLM TP4 生产 compose(2026-09-15 实配+8900B jumbo 零丢包)：
  # 环拓扑三件套缺一不可 —— NCCL_IB_SUBNET_AWARE_ROUTING=1 + _SUBNET_PREFIX_LEN=24
  # (默认按 NIC index 配对会连对端另一个口, QP INIT→RTR 卡 110 timeout)，
  # 及补丁 NCCL 自身开关 NCCL_SKIP_TREE_CONNECT=1 / NCCL_SWITCHLESS_RING_ONLY=1
  # (挂了补丁不发这两个 env = 补丁不生效, 等于 stock tree-connect)。
  local nccl=(-e NCCL_NET=IB -e NCCL_IB_DISABLE=0
              -e NCCL_SOCKET_IFNAME=enP7s7
              -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
              -e NCCL_IB_GID_INDEX=$gid
              -e NCCL_IB_ADDR_RANGE=10.0.0.0/22
              -e NCCL_IB_SUBNET_PREFIX_LEN=24
              -e NCCL_IB_SUBNET_AWARE_ROUTING=1
              -e NCCL_IB_MERGE_NICS=0
              -e NCCL_ALGO=Ring
              -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
              -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_SWITCHLESS_RING_ONLY=1
              -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2
              -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0
              -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN
              -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
              -e TF_NCCL_LIB=$NCCL_IN_CONTAINER/libnccl.so.2)
  h "$host" "docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} ${ENV_ARGS[*]} ${nccl[*]} \
     -v $NCCL_PATCH:$NCCL_IN_CONTAINER:ro -v /models:/models:ro -v $KCACHE:/cache \
     $IMAGE tensorfold serve $MODEL_DIR --rank $rank \
       $( [ "$rank" = 0 ] && echo "--name $SERVED_NAME --host 0.0.0.0 --port $PORT" ) ${SERVE_ARGS[*]}" >/dev/null \
    || die "rank$rank ($host) 启动失败"
  log "rank$rank ($host) 已启动"
}
launch 3; launch 2; launch 1; launch 0
log "四 rank 已启动 (3→2→1→0), 等待就绪 (首次含 CUDA kernel 编译, 最长 ${WAIT_TIMEOUT}s)"

# ---------------- 就绪等待 + refit 降窗重试 ----------------
refit=""
for attempt in 1 2; do
  start=$SECONDS
  until h "$H1" "curl -sf --max-time 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; do
    for spec in "0:$H1" "1:$W1" "2:$W2" "3:$W3"; do
      rank=${spec%%:*}; host=${spec#*:}
      alive=$(h "$host" "docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null || echo missing")
      if [[ "$alive" != "true" ]]; then
        refit=$(h "$H1" "docker logs $CONTAINER_NAME 2>&1 | sed -n 's/.*largest fitting prompt-plus-reply window: \\([0-9]*\\) tokens.*/\\1/p' | tail -1" || true)
        if [[ -n "$refit" && $attempt == 1 ]]; then break 2; fi
        dump_logs "$H1"; dump_logs "$W1"; dump_logs "$W2"; dump_logs "$W3"
        die "rank$rank ($host) 在就绪前退出"
      fi
    done
    (( SECONDS - start < WAIT_TIMEOUT )) || die "${WAIT_TIMEOUT}s 未就绪 (容器仍跑, 手工查 docker logs -f $CONTAINER_NAME)"
    (( SECONDS - start >= 120 && (SECONDS - start) % 120 < 6 )) && \
      log "⋯ $((SECONDS - start))s: master GPU 已载 $(h "$H1" "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1" || echo '?')"
    sleep 5
  done
  [[ -n "$refit" ]] || break
  warn "内存预算只装得下 ${refit}-token 窗口 (非 $CONTEXT): 降窗重启"
  CONTEXT=$refit
  SERVE_ARGS=(--tp 4 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
              --drafter "$DRAFTER_DIR" --context "$CONTEXT" --parallel "$PARALLEL"
              --max-tokens "$MAX_TOKENS" --thinking --vision)
  for host in "$H1" "$W1" "$W2" "$W3"; do
    h "$host" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
  done
  launch 3; launch 2; launch 1; launch 0
done
log "API 就绪 (耗时 $((SECONDS - start))s)"

# ---------------- 烟测 ----------------
smoke=$(h "$H1" "curl -s --max-time 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$SERVED_NAME\",\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}]}'" | \
  python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"].get("content") or ""; assert c.strip(), r; print(repr(c.strip()[:40]), r["usage"]["completion_tokens"], "tok")' 2>/dev/null) \
  || die "烟测失败 (查四机 docker logs)"
log "烟测 OK: $smoke"
log "✔ glm53-flash-tf-unc-tp4-v062 LIVE: http://192.168.90.103:$PORT/v1 (mesh) · model=$SERVED_NAME · ctx=$CONTEXT · par=$PARALLEL · kv=$KV · tp=4"
log "日志: ssh dgx03 docker logs -f $CONTAINER_NAME (rank0) | dgx04/dgx01/dgx02 同名容器 = rank1/2/3"
log "停止: bash /root/stop-glm53-flash-tf-unc-tp4-v062.sh (master-first)"
