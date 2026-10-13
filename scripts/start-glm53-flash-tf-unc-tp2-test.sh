#!/usr/bin/env bash
# =====================================================================
# start-glm53-flash-tf-unc-tp2-test.sh   ← TP2 版, 由现役 TP4 脚本(start-glm53-flash-tf-unc-tp4-v067.sh)新建改编
# GLM-5.3-Flash (neko-legends 去审查 EXL3, mcg codebook) · TensorFold v0.6.7 统一镜像(本集群自 patch, 全部烘焙)
#   ★ 不用 Mia 那套 TP2 脚本(start-glm53-flash-tf-unc-tp2.sh = v0.6.0 + Mia 53 patches + COMM=roce):
#     我们镜像里的补丁栈(TP4 W4 泛化/prefill 修复/roce+hcsplit+kv8+l2pf/视觉前缀复用/janitor)全在,
#     数据面走同一套 switchless 补丁 NCCL, 除 TP 外所有参数与 TP4 现役逐字一致(对比实验纪律)。
#   ★ 与 TP4 的全部差异(仅此 7 项):
#     ① --tp 2, rank0=dgx03 rank1=dgx04 (环上相邻, 光纤直连)
#     ② NCCL_IB_HCA/ADDR_RANGE 按 dgx03↔dgx04 直连链路改写(见 launch() 内注释)
#     ③ MEM_GATE_GIB 85→100 (权重/rank 41→82GiB) + MASTER_PORT 29551→29552
#     ④ KCACHE / SPILL_DIR 独立目录 (TP2 内核形状与 TP4 不同, 缓存不混)
#     ⑤ 摘掉 TF_OVERLAY multi.py 挂载 (v0.6.7 已烘焙同内容, TP4 那个挂载是历史残留)
#     ⑥ 就绪等待/烟测/refit 只盯 2 个 rank
#     ⑦ 停止脚本 = stop-glm53-flash-tf-unc-tp2-test.sh (master-first: dgx03→dgx04)
#
# ── 内存账(起栈前必读) ──
#   权重 164G/2机 = 82 GiB/rank + DFlash2 drafter ~1.1 GiB/rank + NCCL/图开销 ~7 GiB
#   池子 = 启动时 MemAvailable − RESERVE(7) − estimate(engine 自动算, 夹到 budget)
#   KV/token: TP4 实测 6.9 KiB/token(全 rank 合计) → TP2 每 rank 3.45 KiB/token
#   → 池子(tokens) ≈ (MemAvailable − 82 − 7 − 7) / 3.45KiB   [用户拍板 2026-10-13: TP2 RESERVE 只留 7GiB]
#     停 TP4 后直接起: 空闲 ~85GiB → 负数, gate 直接 die
#     trim 后 ~105GiB → 池 ~9GiB ≈ 2.7M tok;  ~110GiB → ~14GiB ≈ 4.2M tok;  ~115GiB → ~19GiB ≈ 5.7M tok
#   (对照: TP4 现役池 6.42M tok)
#   单请求 1M 窗口需空闲 ≥ ~100GiB, 装不下由内置 refit 自动降窗 —— 机制内建, 不是错。
#   ⚠ glm5_next 无 MemoryGate; 上游 1M prompt 峰值 scratch 超 estimate ~10GiB 且 TP2 每 rank 更大,
#     RESERVE=7 是用户拍板的换池子交易: 若 1M 长会话 prefill OOM, 先降 CONTEXT/MAX_TOKENS, 不回抬 RESERVE。
#
# ── 未实测点(起栈时重点盯) ──
#   switchless 补丁 NCCL 只在 TP4 环验收过; TP2 双 HCA 同段(10.0.2/24)组合未跑过。
#   若 rank0/rank1 QP INIT→RTR 卡死: 先查 docker logs; 备选路径 = 每机只列 1 个 HCA(补丁若拒 2-listener 校验则退回)
#   或 TF_GLM_COMM=roce(Mia 路径, 需先验 v0.6.7 是否仍保留该传输)。
#
# 依赖(dgx03+dgx04 都要): 镜像 tensorfold-glm53:v0.6.7 · /models/GLM-5.3-Flash-Uncensored-EXL3 (92 shard)
#   /models/GLM-5.3-Flash-DFlash2 · /root/nccl-switchless/libnccl.so.2 · 内核缓存目录(脚本自建)
# 用法: 只在 dgx03 执行:  bash /root/start-glm53-flash-tf-unc-tp2-test.sh
#       前置: 先四机停 TP4: bash /root/stop-glm53-flash-tf-unc-tp4-v067.sh (master-first)
#             再做内存 trim(dgx-vllm-memory-trim 流程)把两机 MemFree 挤到 100+
#       停:  bash /root/stop-glm53-flash-tf-unc-tp2-test.sh   (master-first)
# 红线: 本脚本不替用户停任何在跑生产容器; :8000 被占即 die 并提示先停 TP4。
# =====================================================================
set -euo pipefail

# ---------------- 配置区 ----------------
CONTAINER_NAME="glm53-flash-tf-unc-tp2-test"
MODEL_DIR="/models/GLM-5.3-Flash-Uncensored-EXL3"
DRAFTER_DIR="/models/GLM-5.3-Flash-DFlash2"
PORT=8000                       # 集群铁律
MASTER_PORT=29553               # 与 TP4 的 29551 错开
MASTER_ADDR="192.168.89.101"    # rank0(dgx03) 管理网地址: TCPStore rendezvous + NCCL bootstrap
SERVED_NAME="glm53-flash-tf-unc-tp2-test"
H1="root@192.168.89.101"        # dgx01 (rank0 + API, master; 测试栈)
W1="root@192.168.89.102"        # dgx02 (rank1; 测试栈)
IMAGE="tensorfold-glm53:v0.6.7"
KCACHE="/root/tf-kernel-cache/glm53-tf-unc-tp2-test"   # 每机各自编译缓存(TP2 形状, 与 TP4 目录分开)
NCCL_PATCH="/root/nccl-switchless"
NCCL_IN_CONTAINER="/opt/patched-nccl"
TF_PKG="/usr/local/lib/python3.12/dist-packages/tensorfold"   # 容器内 TF 包根
SPILL_DIR="/root/glm53-spill-tp2-test"   # 与 TP4 的 spill 目录分开, 避免跨 TP 混快照
SPILL_GIB=240
# 除 TP/拓扑外与 TP4 现役逐字一致(对比实验纪律); TP2 差异仅 MEM_GATE_GIB(见下)
CONTEXT=1048576
PARALLEL=12
KV=fp8
DENSE=q4
MAX_TOKENS=65536
DRAFT_POLICY="fnc7:0.3"
WAIT_TIMEOUT=3000               # 首启含双机 CUDA kernel 编译(新 KCACHE 目录, 首次全编)
MEM_GATE_GIB=100                # 起栈前每机 MemFree 门槛 (权重 ~82GiB/rank + NCCL/图/池余量)
                                #   停 TP4 后裸空闲 ~85 不够 → die 提示先 trim; trim 到 105-115 才有像样的池子
TF_ENVS=(
  HF_HUB_OFFLINE=1
  TF_GLM_KV=$KV
  TF_GLM_DENSE=$DENSE
  TF_GLM_COMM=roce
  TF_GLM_COPY_DRAFTS=1
  TF_GLM_COPY_MAX=15
  TF_GLM_DFLASH_POLICY=fnc7:0.3
  TF_GLM_DRAFT_QUANT=mse
  TF_GLM_MULTI_PROFILE=1
  TF_GLM_HC_SPLIT=1                # 测试栈: 保留 world=2+gather 带 bug 组合供排查(生产 TP2=dgx03+dgx04 已关)
  TF_GLM_HC_EXCHANGE=p2p           # 单变量实验 2026-10-13: gather(world=2) 确定性算错; 试 Mia 原生 p2p
  TF_GLM_PREFILL_OVERLAP=0
  TF_GLM_KDA_CHUNKED=1
  TF_GLM_SPARSE_ONEPASS=1
  TF_GLM_SCORE_RB=8
  TF_GLM_PREFILL_ROWS=8192
  TF_GLM_WIDE_GRAPHS=16
  TF_GLM_COPY_REPLY_MATCH=16
  TF_GLM_MULTI_LONE=1
  TF_GLM_MULTI_PREFILL=1
  TF_GLM_PROFILE=1
  TF_GLM_FILL_SHARE=0.5
  TF_GLM_L2PF=1
  TF_GLM_EXL3_LOADS=nc
  TF_GLM_SHARED_PREFIX=1
  TF_GLM_CACHE_ENTRIES=128
  TF_GLM_CACHE_GIB=1000                # 不封顶: 池由 MemAvailable(启动时)-RESERVE 自动算, engine 夹到 (budget-estimate)
  TENSORFOLD_MEMORY_RESERVE_GIB=7      # 用户拍板 2026-10-13: TP2 只留 7GiB (TP4 档 14.5); 1M 长会话 prefill OOM 先降窗, 不回抬
  TF_GLM_MTP=auto
  TENSORFOLD_NO_UPDATE_CHECK=1
)
# ---------------- 工具 ----------------
log()  { printf '[start-tf-tp2-test] %s\n' "$*"; }
warn() { printf '[start-tf-tp2-test] WARN: %s\n' "$*" >&2; }
die()  { printf '[start-tf-tp2-test] ERROR: %s\n' "$*" >&2; exit 1; }
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
SERVE_ARGS=(--tp 2 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
            --drafter "$DRAFTER_DIR" --context "$CONTEXT" --parallel "$PARALLEL"
            --max-tokens "$MAX_TOKENS" --thinking --vision
            --spill-gib "$SPILL_GIB" --snapshot-dir "$SPILL_DIR"
            --alias llm-proxy-bg --name-priority llm-proxy-bg=background)

# 只清自己名字的残留容器（绝不碰 TP4/vLLM 生产容器）
for host in "$H1" "$W1"; do
  h "$host" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
done

for host in "$H1" "$W1"; do
  h "$host" "test -f $MODEL_DIR/config.json" || die "$host 缺 $MODEL_DIR/config.json"
  n=$(h "$host" "ls $MODEL_DIR/*.safetensors 2>/dev/null | wc -l")
  [[ "$n" == "92" ]] || die "$host 权重 shard 数 $n != 92"
  h "$host" "test -f $DRAFTER_DIR/model.safetensors" || die "$host 缺 DFlash2 drafter"
  h "$host" "test -f $NCCL_PATCH/libnccl.so.2" || die "$host 缺补丁 NCCL $NCCL_PATCH/libnccl.so.2"
  h "$host" "docker image inspect $IMAGE >/dev/null 2>&1" || die "$host 缺镜像 $IMAGE"
  h "$host" "mkdir -p $KCACHE"
  free=$(free_gib "$host")
  if (( free < MEM_GATE_GIB )); then
    die "$host MemFree ${free}GiB < ${MEM_GATE_GIB}GiB：先停 TP4(stop-glm53-flash-tf-unc-tp4-v067.sh) 并做内存 trim(dgx-vllm-memory-trim) 再起 TP2"
  fi
  log "$host OK (shard=92, MemFree=${free}GiB)"
done
if port_busy; then
  die "dgx01 :$PORT 已被占用：先清掉占用者再起测试栈 (注意 llm-proxy primary_backends 含 dgx01/dgx02:8000)"
fi
log "preflight OK (双机文件/内存 gate 通过, :8000 空闲)"

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
GID_H=$(gid_of "$H1"); GID_W1=$(gid_of "$W1")
log "GID index: dgx01=$GID_H dgx02=$GID_W1"

# ---------------- 启动 (worker-first 1→0) ----------------
launch() {  # $1=rank
  local rank=$1 host gid hcas
  case $rank in
    0) host=$H1; gid=$GID_H;  hcas="rocep1s0f0,roceP2p1s0f0" ;;  # dgx01 朝 dgx02 的两个 HCA (10.0.0.101/.111)
    1) host=$W1; gid=$GID_W1; hcas="rocep1s0f0,roceP2p1s0f0" ;;  # dgx02 朝 dgx01 的两个 HCA (10.0.0.102/.112)
    *) die "bad rank $rank" ;;
  esac
  # NCCL 身份(逐字继承 TP4 宪法, 仅两处按 TP2 拓扑改写):
  #   ① NCCL_IB_HCA: 不再列 f0+f1 双向(那是四机环), 只列本机朝对端的两张 HCA ——
  #      dgx03↔dgx04 直连 = 10.0.2/24 同一 L2, 每机两张(rocep1s0 + roceP2p1s0 各一), 凑满补丁
  #      NCCL 的"恰好 2 个 listener GID"硬校验, 且双 HCA 聚合吃满 ~200Gbps;
  #      (TP4 列的 rocep1s0f0 在 dgx03 朝 dgx02 / 在 dgx04 朝 dgx01, TP2 作业里没有对端, 列了必挂)
  #   ② NCCL_IB_ADDR_RANGE: 10.0.0.0/22 → 10.0.2.0/24 (只覆盖直连段)
  # 其余(subnet-aware routing / skip-tree / switchless-ring / 4 channels)逐字不动。
  local nccl=(-e NCCL_NET=IB -e NCCL_IB_DISABLE=0
              -e NCCL_SOCKET_IFNAME=enP7s7
              -e NCCL_IB_HCA=$hcas
              -e NCCL_IB_GID_INDEX=$gid
              -e NCCL_IB_ADDR_RANGE=10.0.0.0/24
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
  h "$host" "docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} ${ENV_ARGS[*]} ${nccl[*]} \\
     -v $NCCL_PATCH:$NCCL_IN_CONTAINER:ro -v /models:/models:ro -v $KCACHE:/cache \\
     -v $SPILL_DIR:$SPILL_DIR \\
     $IMAGE tensorfold serve $MODEL_DIR --rank $rank \\
       $( [ "$rank" = 0 ] && echo "--name $SERVED_NAME --host 0.0.0.0 --port $PORT" ) ${SERVE_ARGS[*]}" >/dev/null \
    || die "rank$rank ($host) 启动失败"
  log "rank$rank ($host) 已启动"
}
launch 1; launch 0
log "两 rank 已启动 (1→0), 等待就绪 (首次含 CUDA kernel 编译, 最长 ${WAIT_TIMEOUT}s)"

# ---------------- 就绪等待 + refit 降窗重试 ----------------
refit=""
for attempt in 1 2; do
  start=$SECONDS
  until h "$H1" "curl -sf --max-time 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1"; do
    for spec in "0:$H1" "1:$W1"; do
      rank=${spec%%:*}; host=${spec#*:}
      alive=$(h "$host" "docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null || echo missing")
      if [[ "$alive" != "true" ]]; then
        refit=$(h "$H1" "docker logs $CONTAINER_NAME 2>&1 | sed -n 's/.*largest fitting prompt-plus-reply window: \\\([0-9]*\\\) tokens.*/\\\1/p' | tail -1" || true)
        if [[ -n "$refit" && $attempt == 1 ]]; then break 2; fi
        dump_logs "$H1"; dump_logs "$W1"
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
  SERVE_ARGS=(--tp 2 --master "$MASTER_ADDR" --master-port "$MASTER_PORT"
              --drafter "$DRAFTER_DIR" --context "$CONTEXT" --parallel "$PARALLEL"
              --max-tokens "$MAX_TOKENS" --thinking --vision
              --spill-gib "$SPILL_GIB" --snapshot-dir "$SPILL_DIR"
            --alias llm-proxy-bg --name-priority llm-proxy-bg=background)
  for host in "$H1" "$W1"; do
    h "$host" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
  done
  launch 1; launch 0
done
log "API 就绪 (耗时 $((SECONDS - start))s)"

# ---------------- 烟测 ----------------
smoke=$(h "$H1" "curl -s --max-time 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$SERVED_NAME\",\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}]}'" | \
  python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"].get("content") or ""; assert c.strip(), r; print(repr(c.strip()[:40]), r["usage"]["completion_tokens"], "tok")' 2>/dev/null) \
  || die "烟测失败 (查两机 docker logs)"
log "烟测 OK: $smoke"
log "✔ glm53-flash-tf-unc-tp2-test LIVE: http://192.168.90.101:$PORT/v1 (mesh) · model=$SERVED_NAME · ctx=$CONTEXT · par=$PARALLEL · kv=$KV · tp=2"
log "日志: ssh dgx01 docker logs -f $CONTAINER_NAME (rank0) | dgx02 同名容器 = rank1"
log "停止: bash /root/stop-glm53-flash-tf-unc-tp2-test.sh (master-first)"
