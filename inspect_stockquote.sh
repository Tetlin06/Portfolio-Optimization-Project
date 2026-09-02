#!/usr/bin/env bash
set -euo pipefail
curl -L -sS --max-time 30 https://stockquote.io/static/script.js -o /tmp/stockquote-script.js
wc -c /tmp/stockquote-script.js
sed -n '1,260p' /tmp/stockquote-script.js
