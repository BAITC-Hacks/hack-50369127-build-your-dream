"""Loopback-only web application using Python's standard library."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from .common import ROOT, encoded


def serve(agent, port=8080):
    action_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(payload, bytes):
                payload = encoded(payload)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def trusted(self):
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return host in allowed and (origin is None or origin in {f"http://{h}" for h in allowed})

        def do_GET(self):
            if not self.trusted():
                self.send(403, {"error": "Допускаются только локальные запросы."})
                return
            request = urlparse(self.path)
            try:
                if request.path == "/api/state":
                    self.send(200, agent.snapshot())
                elif request.path == "/api/export":
                    kind = parse_qs(request.query).get("kind", ["forecast"])[0]
                    self.send(200, ("\ufeff" + agent.export(kind)).encode("utf-8"), "text/csv; charset=utf-8", kind + ".csv")
                else:
                    name = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}.get(request.path)
                    if not name:
                        self.send(404, {"error": "Страница не найдена."})
                        return
                    path = ROOT / "web" / name
                    content_type = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}[path.suffix]
                    self.send(200, path.read_bytes(), content_type + "; charset=utf-8")
            except (ValueError, OSError) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self):
            if not self.trusted() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(403, {"error": "Ожидается локальный JSON-запрос."})
                return
            if not action_lock.acquire(blocking=False):
                self.send(409, {"error": "Агент уже выполняет расчёт. Дождитесь завершения."})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 25 * 1024 * 1024:
                    raise ValueError("Допустимый размер запроса: от 1 байта до 25 МБ.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Ожидается JSON-объект.")
                if self.path == "/api/demo":
                    result = agent.demo()
                elif self.path == "/api/import":
                    result = agent.import_csv(body.get("csv", ""), body.get("filename", "history.csv"),
                        float(body.get("timezone_offset_hours", 5)), float(body.get("power_scale", 1)))
                elif self.path == "/api/forecast":
                    result = agent.forecast(body["as_of"], int(body.get("hours", 48)), body.get("mode", "archive"), bool(body.get("refresh", False)))
                elif self.path == "/api/backtest":
                    result = agent.backtest(body.get("mode", "archive"), int(body.get("hours", 48)))
                else:
                    self.send(404, {"error": "Действие не найдено."})
                    return
                self.send(200, result)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            finally:
                action_lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Wind Agent: http://127.0.0.1:{port} (Ctrl+C — остановка)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
