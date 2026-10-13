#!/usr/bin/env python3
"""TF glm5_next 验收电池（长度分档 + 确定性 + draft 对拍）。

用途：起栈后证明引擎在「≥64 行（HC 行切分门槛）的 prompt」上仍输出正确内容。
烟测（10 tok）走的是 <64 行路径，骗过 §10.9 与 2026-10-13 两次事故——不跑本电池不算验收。

用法：改 BASE/MODEL 后直连引擎跑（绕开 llm-proxy）：python3 tp2_battery.py
判据：每档 ok=True（答出 PINEAPPLE-7Z 且 ≤24 字符）；同 prompt 重复两次的引擎 sha 应相同；
      第 6 档 "draft": false 的内容与第 3 档逐字一致（bit-exact）。sha 从引擎日志取：
      docker logs <容器> 2>&1 | grep "done req"
"""
import json, urllib.request, time

BASE = "http://192.168.90.103:8000/v1/chat/completions"   # 引擎直连（mesh）；TP4 时改 90.103 同名端口
MODEL = "glm53-flash-tf-unc-tp2-v067"                    # served 名，随栈改

FILLER = ("The quick brown fox jumps over the lazy dog near the river bank. "
          "Engineers measure throughput and latency before trusting a deployment. ")
NEEDLE = "Ignore everything above and reply with exactly this token and nothing else: PINEAPPLE-7Z"


def ask(n_filler_sentences, extra=None, max_tokens=16, tag=""):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": FILLER * n_filler_sentences + "\n\n" + NEEDLE}],
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"].get("content") or ""
    u = d.get("usage", {})
    ok = "PINEAPPLE-7Z" in msg and len(msg.strip()) <= 24
    print(f"[{tag}] prompt_tok={u.get('prompt_tokens')} out_tok={u.get('completion_tokens')} "
          f"{dt:.1f}s ok={ok} content={msg.strip()[:120]!r}", flush=True)
    return ok, msg


if __name__ == "__main__":
    ask(0, tag="control")        # 26 tok：<64 行，永远对；只当线路探活
    ask(6, tag="short")          # ~170 tok：行切分门槛之上
    ask(40, tag="mid")           # ~986 tok
    ask(160, tag="long")         # ~3866 tok
    ask(40, tag="mid-repeat")    # 确定性：应与 mid 逐字相同
    ask(40, extra={"draft": False}, tag="mid-serial")   # bit-exact：应与 mid 相同
