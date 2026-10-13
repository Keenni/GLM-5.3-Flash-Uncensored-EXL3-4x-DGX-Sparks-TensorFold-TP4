#!/usr/bin/env bash
# =====================================================================
# stop-glm53-flash-tf-unc-tp2-v067.sh — TF TP2 (镜像 v0.6.7) 停止 (master-first 铁律)
# 顺序: 先 master(dgx03) 后 worker(dgx04)；只动自己的容器名，不碰其它栈。
# =====================================================================
set -euo pipefail
CONTAINER_NAME="glm53-flash-tf-unc-tp2-v067"
H1="root@192.168.89.103"
W1="root@192.168.89.104"
log() { printf '[stop-tf-tp2-v067] %s\n' "$*"; }
h()   { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

log "master-first: 停 rank0 (dgx03) ..."
h "$H1" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
log "停 dgx04 ..."
h "$W1" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
log "✔ TF TP2 v067 全停 (两容器已删)"
