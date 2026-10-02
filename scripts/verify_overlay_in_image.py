from tensorfold.families.glm5_next.cuda.engine import PARALLEL_MOST
from tensorfold.families.glm5_next.cuda.segments import MAX_SEGS
import inspect
from tensorfold import cli_args
from tensorfold.families.glm5_next import cuda_engine
from tensorfold.families.glm5_next.cuda import weights as W, split as S, hcsplit

eng = __import__("tensorfold.families.glm5_next.cuda.engine", fromlist=["x"])
cli_src = inspect.getsource(cli_args)
eng_src = inspect.getsource(eng)
w_src = inspect.getsource(W.load)
rr_src = inspect.getsource(S.RankReader.__init__)
hc_src = inspect.getsource(hcsplit)

checks = {
    "PARALLEL_MOST==6": PARALLEL_MOST == 6,
    "MAX_SEGS==6": MAX_SEGS == 6,
    "cli tp choices 1,2,4": "choices=(1, 2, 4)" in cli_src,
    "engine world attr": "self.world = int(world)" in eng_src,
    "engine NCCL world": "NCCL(rank, self.world" in eng_src,
    "engine bell per-rank": "tf_glm_request_{self.rank}_" in eng_src,
    "engine admit world": "world=self.world" in eng_src,
    "engine load world": "world=self.world, mtp=self.mtp_on" in eng_src,
    "weights load world kw": "world: int = 2, device" in w_src,
    "weights guard": "do not divide by" in w_src,
    "split RankReader world": "rank: int, world: int = 2" in rr_src,
    "split_bytes world kw": "def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int,\n                world: int = 2)" in inspect.getsource(S.split_bytes),
    "hcsplit TP4 guard": "unset it for TP4" in hc_src,
}
bad = [k for k, v in checks.items() if not v]
print("OVERLAY-VERIFY:", "ALL-OK" if not bad else "FAIL " + ",".join(bad))
