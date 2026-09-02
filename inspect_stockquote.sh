#!/usr/bin/env bash
set -euo pipefail
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36'
curl -L -sS --max-time 30 -A "$UA" -c /tmp/sq.cookies https://www.stockquote.io/ -o /tmp/stockquote-page.html
curl -L -sS --max-time 60 -A "$UA" -b /tmp/sq.cookies -e https://www.stockquote.io/ https://www.stockquote.io/static/script.js -o /tmp/stockquote-script.js
wc -c /tmp/stockquote-page.html /tmp/stockquote-script.js
grep -nE 'fetch\(|/api/|download|search|histor|symbol|cusip' /tmp/stockquote-script.js | head -300 || true
sed -n '1,320p' /tmp/stockquote-script.js
