# verify_colab.py — verify Colab tab reachability via CDP (port 9222).
# Companion of start_colab_control.bat. Uses the Chrome DevTools Protocol over
# HTTP/websocket. NOTE: does NOT spawn powershell.exe or cmd.exe; uses only
# python stdlib, so it never creates a console window of its own.
import json, sys, time, urllib.request

def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode('utf-8', 'replace')

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '9222'
    print('=== VERIFY_COLAB ===', flush=True)
    print(f'STEP1: CDP version on 127.0.0.1:{port}', flush=True)
    try:
        ver = json.loads(http_get(f'http://127.0.0.1:{port}/json/version'))
        print('  browser:', ver.get('Browser'), flush=True)
        print('  ws  :', ver.get('webSocketDebuggerUrl'), flush=True)
    except Exception as e:
        print('  FAIL no CDP endpoint:', e, flush=True)
        print('=== VERIFY_COLAB FAIL ===', flush=True)
        return 1

    print('STEP2: open tabs', flush=True)
    try:
        tabs = json.loads(http_get(f'http://127.0.0.1:{port}/json/list'))
        colab = [t for t in tabs if 'colab' in t.get('title', '') or 'colab' in t.get('url', '')]
        print(f'  total tabs={len(tabs)} colab={len(colab)}', flush=True)
        for t in colab:
            print(f'  - {t.get("title")} | {t.get("url")} | {t.get("type")} | ws={t.get("webSocketDebuggerUrl")}', flush=True)
    except Exception as e:
        print('  FAIL reading tabs:', e, flush=True)
        print('=== VERIFY_COLAB FAIL ===', flush=True)
        return 1

    if not colab:
        print('  WARN no colab tab found (notebooks may be named differently)', flush=True)

    print('STEP3: done (CDP attached, tabs enumerated)', flush=True)
    print('=== VERIFY_COLAB OK ===', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())