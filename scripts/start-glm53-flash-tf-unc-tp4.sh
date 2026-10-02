#!/usr/bin/env bash
# =====================================================================
# start-glm53-flash-tf-unc-tp4.sh
# GLM-5.3-Flash (neko-legends 去审查 EXL3) · TensorFold v0.6.0 + Mia 53 patches + TP4 overlay · 四机 TP4
# 基于现役 TP2 脚本(start-glm53-flash-tf-unc-tp2.sh)改编；NCCL 数据面身份注入逐字复用
# vLLM TP4 生产脚本(start-glm53-flash-nvfp4-tp4.sh, 2026-09-15 宪法)的实配 helpers：
#   权重平铺 /models 直读 / 端口 8000 / master 恒 dgx03 / worker-first 3→2→1→0 /
#   内存 gate / restart=no / refit 降窗重试 / 烟测验收 / 只清理自己的容器
# 依赖(四台都要): 镜像 tensorfold-glm53:v0.6.0 (GHCR digest 22789f0cb3dc pin)
#   /models/GLM-5.3-Flash-Uncensored-EXL3 (92 shard) + /models/GLM-5.3-Flash-DFlash2
#   /root/nccl-switchless/libnccl.so.2 (switchless 补丁 NCCL 2.30.7)
#   /root/tf-overlay-tp4/ (TP4 overlay 8 文件 + MANIFEST.md5, 四台 md5 一致)
# 用法: 只在 dgx03 执行:  bash /root/start-glm53-flash-tf-unc-tp4.sh
# 红线: 本脚本不替用户停任何在跑生产容器；:8000 被占即 die 并提示先跑 TP2 的 stop 脚本。
# =====================================================================
set -euo pipefail

# ---------------- 配置区 ----------------
CONTAINER_NAME="glm53-flash-tf-unc-tp4"
MODEL_DIR="/models/GLM-5.3-Flash-Uncensored-EXL3"
DRAFTER_DIR="/models/GLM-5.3-Flash-DFlash2"
PORT=8000                       # 集群铁律
MASTER_PORT=29551
MASTER_ADDR="192.168.89.103"    # rank0(dgx03) 管理网地址: TCPStore rendezvous + NCCL bootstrap
SERVED_NAME="glm53-flash-tf-unc-tp4"
H1="root@192.168.89.103"        # dgx03 (rank0 + API, master)
W1="root@192.168.89.104"        # dgx04 (rank1)
W2="root@192.168.89.101"        # dgx01 (rank2)
W3="root@192.168.89.102"        # dgx02 (rank3)
IMAGE="tensorfold-glm53:v0.6.0"
KCACHE="/root/tf-kernel-cache/glm53-tf-unc-tp4-v06"   # 每机各自编译缓存(与 TP2 的目录分开)
TF_OVERLAY="/root/tf-overlay-tp4"
NCCL_PATCH="/root/nccl-switchless"
NCCL_IN_CONTAINER="/opt/patched-nccl"
TF_PKG="/usr/local/lib/python3.12/dist-packages/tensorfold"   # 容器内 TF 包根（overlay 挂载目标）
# Mia TP2 现役档全量透传；TP4 差异: HC_SPLIT=0(hcsplit 是 2-rank 机制)、COMM=nccl(补丁栈)
CONTEXT=1048576
PARALLEL=6
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
  TF_GLM_HC_SPLIT=0
  TF_GLM_PREFILL_OVERLAP=0
  TF_GLM_KDA_CHUNKED=1
  TF_GLM_WIDE_GRAPHS=16
  TF_GLM_COPY_REPLY_MATCH=16
  TF_GLM_MULTI_LONE=1
  TF_GLM_MULTI_PREFILL=1
  TF_GLM_L2PF=1
  TF_GLM_EXL3_LOADS=nc
  TF_GLM_SHARED_PREFIX=1
  TF_GLM_CACHE_ENTRIES=16
  TF_GLM_CACHE_GIB=12.5
  TENSORFOLD_MEMORY_RESERVE_GIB=10
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
# 远端布局 = $TF_OVERLAY/tensorfold/...（与 git 仓/MANIFEST.md5 一致, manifest 校验 cd $TF_OVERLAY 跑）
OVERLAY_MOUNTS=(
  -v "$TF_OVERLAY/tensorfold/cli.py:$TF_PKG/cli.py:ro"
  -v "$TF_OVERLAY/tensorfold/cli_args.py:$TF_PKG/cli_args.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/__init__.py:$TF_PKG/families/glm5_next/__init__.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/engine.py:$TF_PKG/families/glm5_next/cuda/engine.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/hcsplit.py:$TF_PKG/families/glm5_next/cuda/hcsplit.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/multi.py:$TF_PKG/families/glm5_next/cuda/multi.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/segments.py:$TF_PKG/families/glm5_next/cuda/segments.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/split.py:$TF_PKG/families/glm5_next/cuda/split.py:ro"
  -v "$TF_OVERLAY/tensorfold/families/glm5_next/cuda/weights.py:$TF_PKG/families/glm5_next/cuda/weights.py:ro"
)
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
  h "$host" "cd $TF_OVERLAY && md5sum -c MANIFEST.md5 --quiet" || die "$host overlay md5 校验失败 ($TF_OVERLAY)"
  h "$host" "mkdir -p $KCACHE"
  free=$(free_gib "$host")
  if (( free < MEM_GATE_GIB )); then
    die "$host MemFree ${free}GiB < ${MEM_GATE_GIB}GiB：先停旧栈/清理再起 (docker ps 看在跑 LLM)"
  fi
  log "$host OK (shard=92, overlay md5 OK, MemFree=${free}GiB)"
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
  h "$host" "docker run -d --name $CONTAINER_NAME ${RUN_ARGS[*]} ${ENV_ARGS[*]} ${nccl[*]} ${OVERLAY_MOUNTS[*]} \
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
log "✔ glm53-flash-tf-unc-tp4 LIVE: http://192.168.90.103:$PORT/v1 (mesh) · model=$SERVED_NAME · ctx=$CONTEXT · par=$PARALLEL · kv=$KV · tp=4"
log "日志: ssh dgx03 docker logs -f $CONTAINER_NAME (rank0) | dgx04/dgx01/dgx02 同名容器 = rank1/2/3"
log "停止: bash /root/stop-glm53-flash-tf-unc-tp4.sh (master-first)"
