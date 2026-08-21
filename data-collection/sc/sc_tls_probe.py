"""
sc_tls_probe.py — diagnose the TLS handshake failure against abcquality.org.

WHY
---
sc_crawler.py's plain `requests.get()` dies before it ever sends a request:

    SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]
             EOF occurred in violation of protocol'))

"EOF while reading" during the handshake means the SERVER hung up mid-TLS. The
endpoint itself is fine (it serves text/csv to a browser), so the server is
rejecting *this client's* TLS handshake. The usual suspects, in order:

  1. Old server TLS stack / narrow cipher list. Modern OpenSSL 3.x ships a
     restrictive default (SECLEVEL=2) and offers TLS 1.3 first; an older IIS box
     can bail out. Fix: custom SSLContext with SECLEVEL=1 and/or pinned TLS 1.2.
  2. WAF fingerprinting (JA3). Some WAFs drop handshakes whose TLS ClientHello
     doesn't look like a real browser. Python's hello is instantly identifiable.
     Fix: `curl_cffi` with `impersonate="chrome"`, which replays Chrome's exact
     ClientHello.
  3. HTTP/2-only edge. Fix: httpx with http2=True.
  4. Local TLS interception (corporate VPN / antivirus SSL inspection) mangling
     the handshake. Fix: nothing in code — test from another network.

This script tries each strategy against a tiny known-good URL (McCormick has 3
providers) and prints which ones succeed. Run it, paste the output, and we'll
wire the winning transport into sc_crawler.py.

    python sc_tls_probe.py

Optional deps (each is skipped cleanly if absent — install only if a strategy
below turns out to be the one that works):
    pip install httpx[http2] curl_cffi
"""

import ssl
import sys
import traceback

URL = 'https://abcquality.org/provider-search/excel/?county=McCormick'
HOST = 'abcquality.org'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

BROWSER_HEADERS = {
    'User-Agent': UA,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

results = []


def report(name, ok, detail=''):
    results.append((name, ok, detail))
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}' + (f'  -> {detail}' if detail else ''))


def _ok_body(text):
    """A real answer starts with the CSV header and has McCormick's 3 rows."""
    head = (text or '')[:80].replace('\n', ' ')
    if 'Provider Name' not in (text or ''):
        return False, f'unexpected body: {head!r}'
    n = len([ln for ln in text.strip().splitlines()[1:] if ln.strip()])
    return True, f'{n} data row(s); header ok'


# ---------------------------------------------------------------------------
# 0. environment
# ---------------------------------------------------------------------------

def show_env():
    print('Environment')
    print(f'  python  : {sys.version.split()[0]}')
    print(f'  ssl     : {ssl.OPENSSL_VERSION}')
    try:
        import requests
        print(f'  requests: {requests.__version__}')
        import urllib3
        print(f'  urllib3 : {urllib3.__version__}')
    except Exception as e:
        print(f'  requests/urllib3 unavailable: {e}')
    print()


# ---------------------------------------------------------------------------
# 1. raw handshake — is it TLS at all, and which version/cipher lands?
# ---------------------------------------------------------------------------

def probe_raw_handshake():
    import socket
    for label, mk in [
        ('default context', lambda: ssl.create_default_context()),
        ('TLS1.2 pinned', _ctx_tls12),
        ('SECLEVEL=1', _ctx_seclevel1),
        ('SECLEVEL=0 + TLS1.0 floor', _ctx_seclevel0),
    ]:
        try:
            ctx = mk()
            with socket.create_connection((HOST, 443), timeout=15) as sock:
                with ctx.wrap_socket(sock, server_hostname=HOST) as ss:
                    report(f'handshake / {label}', True,
                           f'{ss.version()} {ss.cipher()[0]}')
        except Exception as e:
            report(f'handshake / {label}', False, f'{type(e).__name__}: {e}')


def _ctx_tls12():
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def _ctx_seclevel1():
    ctx = ssl.create_default_context()
    ctx.set_ciphers('DEFAULT@SECLEVEL=1')
    return ctx


def _ctx_seclevel0():
    ctx = ssl.create_default_context()
    ctx.set_ciphers('DEFAULT@SECLEVEL=0')
    try:
        ctx.minimum_version = ssl.TLSVersion.TLSv1
    except Exception:
        pass
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ctx_seclevel1_tls12():
    """The combination that most often revives an old IIS/WAF endpoint."""
    ctx = ssl.create_default_context()
    ctx.set_ciphers('DEFAULT@SECLEVEL=1')
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return ctx


# ---------------------------------------------------------------------------
# 2. requests, plain and with a custom TLS adapter
# ---------------------------------------------------------------------------

def probe_requests_plain():
    import requests
    try:
        r = requests.get(URL, headers={'User-Agent': UA}, timeout=30)
        r.raise_for_status()
        report('requests / plain', *_ok_body(r.text))
    except Exception as e:
        report('requests / plain', False, f'{type(e).__name__}: {e}')


def probe_requests_browser_headers():
    import requests
    try:
        r = requests.get(URL, headers=BROWSER_HEADERS, timeout=30)
        r.raise_for_status()
        report('requests / browser headers', *_ok_body(r.text))
    except Exception as e:
        report('requests / browser headers', False, f'{type(e).__name__}: {e}')


def _tls_adapter(ctx_factory):
    from requests.adapters import HTTPAdapter

    class TlsAdapter(HTTPAdapter):
        def init_poolmanager(self, *a, **k):
            k['ssl_context'] = ctx_factory()
            return super().init_poolmanager(*a, **k)

        def proxy_manager_for(self, *a, **k):
            k['ssl_context'] = ctx_factory()
            return super().proxy_manager_for(*a, **k)

    return TlsAdapter()


def probe_requests_ctx(label, ctx_factory):
    import requests
    try:
        s = requests.Session()
        s.mount('https://', _tls_adapter(ctx_factory))
        r = s.get(URL, headers=BROWSER_HEADERS, timeout=30)
        r.raise_for_status()
        report(f'requests / {label}', *_ok_body(r.text))
    except Exception as e:
        report(f'requests / {label}', False, f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# 3. httpx (HTTP/2)
# ---------------------------------------------------------------------------

def probe_httpx():
    try:
        import httpx
    except ImportError:
        report('httpx / http2', False, 'not installed (pip install "httpx[http2]")')
        return
    for h2 in (True, False):
        label = f'httpx / http{"2" if h2 else "1.1"}'
        try:
            with httpx.Client(http2=h2, timeout=30,
                              headers=BROWSER_HEADERS,
                              follow_redirects=True) as c:
                r = c.get(URL)
                r.raise_for_status()
                report(label, *_ok_body(r.text))
        except Exception as e:
            report(label, False, f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# 4. curl_cffi — replays a real Chrome ClientHello (defeats JA3 blocking)
# ---------------------------------------------------------------------------

def probe_curl_cffi():
    try:
        from curl_cffi import requests as creq
    except ImportError:
        report('curl_cffi / impersonate=chrome', False,
               'not installed (pip install curl_cffi)')
        return
    for imp in ('chrome', 'chrome124', 'chrome110'):
        try:
            r = creq.get(URL, impersonate=imp, timeout=30)
            r.raise_for_status()
            report(f'curl_cffi / impersonate={imp}', *_ok_body(r.text))
            return  # first success is enough
        except Exception as e:
            report(f'curl_cffi / impersonate={imp}', False,
                   f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
# 5. stdlib urllib — different code path than requests
# ---------------------------------------------------------------------------

def probe_urllib():
    import urllib.request
    try:
        req = urllib.request.Request(URL, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            report('urllib / stdlib', *_ok_body(resp.read().decode('utf-8', 'replace')))
    except Exception as e:
        report('urllib / stdlib', False, f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    show_env()
    print(f'Target: {URL}\n')

    print('Raw TLS handshakes (no HTTP yet)')
    probe_raw_handshake()

    print('\nHTTP clients')
    probe_requests_plain()
    probe_requests_browser_headers()
    probe_requests_ctx('TLS1.2 pinned', _ctx_tls12)
    probe_requests_ctx('SECLEVEL=1', _ctx_seclevel1)
    probe_requests_ctx('SECLEVEL=1 + TLS1.2', _ctx_seclevel1_tls12)
    probe_urllib()
    probe_httpx()
    probe_curl_cffi()

    print('\n' + '=' * 62)
    wins = [n for n, ok, _ in results if ok and not n.startswith('handshake')]
    if wins:
        print('WORKING TRANSPORT(S):')
        for w in wins:
            print(f'  - {w}')
        print('\nPaste this output back; the first winner gets wired into '
              'sc_crawler.py.')
    else:
        print('Nothing worked. Two things to try by hand:\n'
              f'  openssl s_client -connect {HOST}:443 -servername {HOST} </dev/null\n'
              f'  curl -sS -o /dev/null -w "%{{http_code}}\\n" "{URL}"\n'
              'If curl works but Python does not, it is a TLS-fingerprint/WAF\n'
              'issue -> curl_cffi. If curl ALSO fails, suspect local TLS\n'
              'interception (VPN / antivirus) or your network -- try a hotspot.')
