import inspect
from tensorfold import cli_args

# cli_args.build_parser 的真实签名
sig = inspect.signature(cli_args.build_parser)
print("build_parser params:", list(sig.parameters))

p = cli_args.build_parser({k: (lambda args: 0) for k in ("serve", "pull", "models", "update", "info")})
a = p.parse_args(["serve", "/models/x", "--tp", "4", "--rank", "2", "--master", "192.168.89.103"])
print("tp=4 parsed:", a.tp, "rank:", a.rank, "master:", a.master)
a2 = p.parse_args(["serve", "/models/x", "--tp", "2", "--rank", "1", "--master", "x"])
print("tp=2 still ok")
try:
    p.parse_args(["serve", "/models/x", "--tp", "3"])
    print("tp=3 accepted (BAD)")
except SystemExit:
    print("tp=3 rejected (OK)")

# family 门（不载模型，只验 tp 校验逻辑的报错路径）
from tensorfold.families.glm5_next import cuda_engine
try:
    cuda_engine("/models/x", tp=3)
    print("family tp=3 accepted (BAD)")
except ValueError as e:
    print("family tp=3 rejected:", str(e)[:60])
try:
    cuda_engine("/models/x", tp=4, rank=1, master="192.168.89.103")
    print("family tp=4 passed gate (would build engine)")
except ValueError as e:
    # rank=1, master given: gate 应放行, 后面在 import engine/读模型时炸（这里没真模型）——
    # 只要不报 tp 门错误就算通过
    if "serves CUDA with" in str(e):
        print("family tp=4 wrongly rejected:", str(e)[:80])
    else:
        print("family tp=4 gate passed; next error (expected, no real model):", str(e)[:60])
