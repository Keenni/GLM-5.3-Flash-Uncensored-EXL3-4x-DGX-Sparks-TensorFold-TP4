"""最小复现：TensorFold 自己的 comm.NCCL 在 world=4 下建链 + 一次 all_gather。
不加载模型、不碰权重，只验 NCCL 通路。四机各跑一份，rank0 最后起（TCPStore server）。"""
import os
import sys

import torch

from tensorfold.cuda.comm import NCCL

rank = int(os.environ.get("DIAG_RANK", "0"))
world = 4
master = os.environ.get("DIAG_MASTER", "192.168.89.103")
port = int(os.environ.get("DIAG_PORT", "29561"))

print(f"[diag] rank {rank}: TF_NCCL_LIB={os.environ.get('TF_NCCL_LIB', '(unset -> stock)')}", flush=True)
c = NCCL(rank, world, master, port)
print(f"[diag] rank {rank}: ncclCommInitRank OK (world={c.world})", flush=True)
x = torch.ones(4, dtype=torch.int32, device="cuda")
y = torch.empty(world * 4, dtype=torch.int32, device="cuda")
c.all_gather(x, y)
print(f"[diag] rank {rank}: all_gather OK -> {y.tolist()}", flush=True)
print(f"[diag] rank {rank}: PASS", flush=True)
