#!/usr/bin/env bash
# TP4 烟测: 直接打本机 :8000
set -uo pipefail
RESP=$(curl -s --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-tf-unc-tp4","max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reply with exactly: OK"}]}')
echo "$RESP" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("PARSE-FAIL:", e); sys.exit(0)
if d.get("choices"):
    c = d["choices"][0].get("message", {}).get("content")
    u = d.get("usage", {})
    print("SMOKE-OK content=%r tok=%s+%s" % (c, u.get("prompt_tokens"), u.get("completion_tokens")))
else:
    print("SMOKE-FAIL:", json.dumps(d)[:400])
'
