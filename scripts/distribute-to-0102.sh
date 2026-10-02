#!/usr/bin/env bash
# TP4 准备件分发：TF 镜像 10.6GiB + Uncensored EXL3 权重 164G → dgx01/dgx02
# 只读源(dgx03)、只写目标机目录；不碰任何在跑容器。已存在的分片跳过（幂等，可重跑）。
set -uo pipefail
LOG=/root/tf-tp4-distribute.log
exec >> "$LOG" 2>&1
echo "=== run start $(date '+%F %T') ==="

# dgx03 上 dgx01/dgx02 的 ssh 别名 HostName 指光纤段(10.0.2.x)，光纤未通=No route to host。
# 一律用管理网 IP 覆盖(skill 铁律)：dgx01=89.101, dgx02=89.102
m() {  # m <机器名或IP> <命令...>
  case $1 in
    dgx01) host=192.168.89.101 ;;
    dgx02) host=192.168.89.102 ;;
    *)     host=$1 ;;
  esac
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=10 "root@$host" "$@"
}

shard_count() { m "$1" "ls /models/GLM-5.3-Flash-Uncensored-EXL3/*.safetensors 2>/dev/null | wc -l"; }

# ---- 阶段 1: 镜像 dgx03 → dgx01 → dgx02 (ctr 走管理网, 逐台) ----
for dst in dgx01 dgx02; do
  if m "$dst" 'ctr -n moby images ls -q | grep -q "docker.io/library/tensorfold-glm53:v0.6.0"'; then
    echo "[image] $dst 已有, 跳过"
  else
    echo "[image] dgx03 -> $dst 开始 $(date '+%T')"
    ssh root@192.168.89.103 "ctr -n moby images export - docker.io/library/tensorfold-glm53:v0.6.0" | \
      m "$dst" "ctr -n moby images import -" && echo "[image] $dst import 完成 $(date '+%T')"
    # 验证: docker create 实测（铁律: 不能只看 inspect）
    m "$dst" "docker create --name tfimg-check tensorfold-glm53:v0.6.0 true >/dev/null 2>&1 && docker rm tfimg-check >/dev/null && echo '[image] $dst docker create 验证 OK' || echo '[image] $dst docker create 验证失败'"
  fi
done

# ---- 阶段 2: 权重 92 shard, dgx03 → dgx01 / dgx02 并行双流 ----
copy_missing() {
  dst_host=$1
  ssh root@192.168.89.103 "ls /models/GLM-5.3-Flash-Uncensored-EXL3/*.safetensors" | while read -r f; do
    base=$(basename "$f")
    if ! m "$dst_host" "test -s /models/GLM-5.3-Flash-Uncensored-EXL3/$base" 2>/dev/null; then
      echo "[weights] $dst_host <- $base $(date '+%T')"
      ssh root@192.168.89.103 "cat '$f'" | m "$dst_host" "cat > '/models/GLM-5.3-Flash-Uncensored-EXL3/.tmp.$base' && mv '/models/GLM-5.3-Flash-Uncensored-EXL3/.tmp.$base' '/models/GLM-5.3-Flash-Uncensored-EXL3/$base'"
    fi
  done
}
(
  copy_missing dgx01
  echo "[weights] dgx01 完成: $(shard_count dgx01)/92 $(date '+%T')"
) &
P1=$!
(
  copy_missing dgx02
  echo "[weights] dgx02 完成: $(shard_count dgx02)/92 $(date '+%T')"
) &
P2=$!
wait $P1 $P2
echo "[weights] dgx01=$(shard_count dgx01)/92 dgx02=$(shard_count dgx02)/92"
echo "=== run done $(date '+%F %T') rc=$? ==="
