#!/usr/bin/env bash
# =====================================================================
# stop-glm53-flash-tf-unc-tp4.sh — TF TP4 停止 (master-first 铁律)
# 顺序: 先 master(dgx03) 后三个 worker；只动自己的容器名，不碰现役栈(2026-10-02 起为 vLLM TP4 四机容器)。
# =====================================================================
set -euo pipefail
CONTAINER_NAME="glm53-flash-tf-unc-tp4"
H1="root@192.168.89.103"
W1="root@192.168.89.104"
W2="root@192.168.89.101"
W3="root@192.168.89.102"
log() { printf '[stop-tf-tp4] %s\n' "$*"; }
h()   { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

log "master-first: 停 rank0 (dgx03) ..."
h "$H1" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
for host in "$W1" "$W2" "$W3"; do
  log "停 $host ..."
  h "$host" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
done
log "✔ TP4 全停 (四容器已删)"
