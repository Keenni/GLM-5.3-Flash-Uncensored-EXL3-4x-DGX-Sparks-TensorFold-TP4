#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="glm53-flash-tf-unc-tp2-test"
H1="root@192.168.89.101"
W1="root@192.168.89.102"
log() { printf '[stop-tf-tp2-test] %s\n' "$*"; }
h()   { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }
log "master-first: 停 rank0 (dgx01) ..."
h "$H1" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
log "停 dgx02 ..."
h "$W1" "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true" || true
log "✔ 测试栈全停"
