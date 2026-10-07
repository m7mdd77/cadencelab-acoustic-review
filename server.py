import base64
import binascii
import json
import secrets
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from audio_engine import MAX_BYTES, compare, signal_fixture

ROOT = Path(__file__).parent
PORT = 3261
STATIC = {"/": ("index.html", "text/html"), "/app.mjs": ("app.mjs", "text/javascript"),
          "/style.css": ("style.css", "text/css")}
REPORTS = OrderedDict()


def remember_report(result):
    token = secrets.token_hex(24)
    REPORTS[token] = json.dumps(result, allow_nan=False).encode()
    while len(REPORTS) > 8:
        REPORTS.popitem(last=False)
    return '/api/reports/' + token


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, value, mime="application/json", attachment=False):
        body = json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="cadencelab-report.json"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self.allowed_host():
            return self.reply(403, {"error": "Loopback host required"})
        if self.path in STATIC:
            file, mime = STATIC[self.path]
            return self.reply(200, (ROOT / file).read_bytes(), mime)
        if self.path == "/api/fixture":
            result = compare(*signal_fixture())
            result["synthetic"] = True
            result['downloadUrl'] = remember_report(result)
            return self.reply(200, result)
        if self.path.startswith('/api/reports/'):
            token = self.path.removeprefix('/api/reports/')
            if token not in REPORTS:
                return self.reply(404, {'error': 'Report unavailable; analyze again'})
            return self.reply(200, REPORTS[token], mime='application/octet-stream', attachment=True)
        return self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed_host() or self.headers.get("Origin") not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            return self.reply(403, {"error": "Same-origin request required"})
        if self.path != "/api/compare":
            return self.reply(404, {"error": "Not found"})
        if self.headers.get("Content-Type") != "application/json":
            return self.reply(415, {"error": "JSON required"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 56 * 1024 * 1024:
                return self.reply(413, {"error": "Request exceeds size limit"})
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request")
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            data = []
            for key in ("baseline", "practice"):
                encoded = request.get(key)
                if not isinstance(encoded, str) or len(encoded) > ((MAX_BYTES + 2) // 3) * 4:
                    raise ValueError("Invalid audio payload size")
                data.append(base64.b64decode(encoded, validate=True))
            result = compare(*data, energy_delta_db=request.get("thresholdDb", 6))
            rich = request.get("includeMfcc", False)
            if not isinstance(rich, bool):
                raise ValueError("includeMfcc must be boolean")
            transcript = request.get("transcript", "")
            if not isinstance(transcript, str) or len(transcript) > 4000:
                raise ValueError("Transcript must be text of at most 4000 characters")
            if transcript.strip():
                from alignment import normalize_transcript, run
                normalize_transcript(transcript)
                checkpoint = ROOT / '.cache/models/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth'
                if not checkpoint.is_file():
                    return self.reply(503, {"error": "Speech model unavailable locally; run the alignment CLI setup first"})
                result["alignment"] = {"baseline": run(data[0], transcript),
                                       "practice": run(data[1], transcript),
                                       "timeWarpApplied": False}
            if rich:
                from speech_features import compare_mfcc
                result["mfcc"] = compare_mfcc(*data)
            result["synthetic"] = False
            result['downloadUrl'] = remember_report(result)
            return self.reply(200, result)
        except (ValueError, binascii.Error, UnicodeError) as exc:
            return self.reply(400, {"error": str(exc)[:200]})
        except Exception:
            return self.reply(500, {"error": "Analysis failed; no report was created"})


if __name__ == "__main__":
    print(f"CadenceLab: http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
