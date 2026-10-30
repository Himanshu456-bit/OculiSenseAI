"""
REST API Server for Third-Party Wellness and Health Applications.
Enables local integrations with Raycast, Home Assistant, Slack bots, and external dashboards.
"""

from __future__ import annotations
import json
import logging
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from oculisense.daemon import OculiSenseDaemon

logger = logging.getLogger(__name__)


class _APIRequestHandler(BaseHTTPRequestHandler):
    daemon_instance: OculiSenseDaemon

    def log_message(self, format: str, *args: Any) -> None:
        # Route server logging through python logger
        logger.debug("OculiSense API: " + format % args)

    def _set_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers(204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        daemon = self.daemon_instance

        if path in ("/api/v1/status", "/api/status"):
            snap = getattr(daemon, "_last_snapshot", None)
            if snap is None:
                # Generate on the fly if not cached
                snap = daemon.step()

            payload = {
                "timestamp": snap.timestamp,
                "session_id": snap.session_id,
                "is_camera_active": getattr(snap, "is_camera_active", getattr(daemon, "is_camera_active", False)),
                "user_profile": {
                    "profile_id": daemon.config.profile.profile_id,
                    "name": daemon.config.profile.name,
                    "baseline_open_ear": daemon.config.profile.baseline_open_ear,
                    "baseline_closed_ear": daemon.config.profile.baseline_closed_ear,
                },
                "vision": {
                    "face_detected": snap.vision.face_detected,
                    "ear_avg": round(snap.vision.ear_avg, 3),
                    "ear_left": round(snap.vision.ear_left, 3),
                    "ear_right": round(snap.vision.ear_right, 3),
                    "gaze_eccentricity": round(getattr(snap.vision, "gaze_eccentricity", 0.0), 3),
                    "is_synthetic": snap.vision.is_synthetic,
                    "source_status": snap.vision.source_status,
                },
                "blink": {
                    "blink_rate_bpm": round(snap.blink.blink_rate_bpm, 1),
                    "perclos": round(snap.blink.perclos, 3),
                    "total_blinks": snap.blink.total_blinks,
                    "is_eyes_closed": snap.blink.is_eyes_closed,
                    "is_prolonged_closure": snap.blink.is_prolonged_closure,
                },
                "fatigue": {
                    "fatigue_score": round(snap.fatigue.fatigue_score, 1),
                    "fatigue_level": snap.fatigue.fatigue_level,
                    "is_warning": snap.fatigue.is_warning,
                    "microsleep_risk": snap.fatigue.microsleep_risk,
                },
                "strain": {
                    "strain_score": round(snap.strain.strain_score, 1),
                    "strain_level": snap.strain.strain_level,
                    "continuous_focus_sec": round(snap.strain.continuous_focus_sec, 1),
                    "is_warning": snap.strain.is_warning,
                    "gaze_eccentricity_factor": getattr(snap.strain, "gaze_eccentricity_factor", 0.0),
                },
                "breaks": {
                    "is_break_recommended": snap.breaks.is_break_recommended,
                    "is_in_break": snap.breaks.is_in_break,
                    "remaining_until_break_sec": round(snap.breaks.remaining_until_break_sec, 1),
                    "remaining_break_duration_sec": round(snap.breaks.remaining_break_duration_sec, 1),
                    "compliance_rate_pct": round(snap.breaks.compliance_rate_pct, 1),
                    "reason": snap.breaks.reason,
                },
                "presence": {
                    "is_present": snap.presence.is_present,
                    "status_label": snap.presence.status_label,
                    "away_duration_sec": round(snap.presence.away_duration_sec, 1),
                },
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

        elif path in ("/api/v1/metrics", "/api/metrics"):
            days = int(query.get("days", [7])[0])
            daily = daemon.db.get_daily_metrics(days=days)
            recent = daemon.db.get_recent_sessions(limit=10)
            payload = {
                "daily_trends": daily,
                "recent_sessions": recent,
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

        elif path in ("/api/v1/breaks", "/api/breaks"):
            limit = int(query.get("limit", [50])[0])
            events = daemon.db.get_break_events(limit=limit)
            payload = {
                "break_events": events,
                "total_prompted": daemon.break_advisor._total_prompted,
                "total_completed": daemon.break_advisor._total_completed,
                "total_skipped": daemon.break_advisor._total_skipped,
                "compliance_rate_pct": round(daemon.break_advisor.compliance_rate, 1),
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

        elif path in ("/api/v1/profiles", "/api/profiles"):
            profiles = daemon.db.list_user_profiles()
            payload = {
                "active_profile_id": daemon.config.profile.profile_id,
                "profiles": [
                    {
                        "profile_id": p.profile_id,
                        "name": p.name,
                        "baseline_open_ear": p.baseline_open_ear,
                        "baseline_closed_ear": p.baseline_closed_ear,
                        "dynamic_ear_threshold": round(p.dynamic_ear_threshold, 3),
                        "dynamic_squint_threshold": round(p.dynamic_squint_threshold, 3),
                        "calibrated_at": p.calibrated_at,
                    }
                    for p in profiles
                ],
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found", "path": path}).encode("utf-8"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        daemon = self.daemon_instance

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path in ("/api/v1/breaks/trigger", "/api/breaks/trigger"):
            reason = data.get("reason", "API_MANUAL_TRIGGER")
            daemon.break_advisor.trigger_dynamic_break(reason=reason)
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "message": f"Break triggered: {reason}"}).encode("utf-8"))

        elif path in ("/api/v1/breaks/snooze", "/api/breaks/snooze"):
            ok = daemon.break_advisor.snooze_break()
            self._set_headers(200 if ok else 400)
            self.wfile.write(json.dumps({"status": "ok" if ok else "error", "snoozed": ok}).encode("utf-8"))

        elif path in ("/api/v1/breaks/complete", "/api/breaks/complete"):
            daemon.break_advisor.complete_break()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "message": "Break marked completed"}).encode("utf-8"))

        elif path in ("/api/v1/camera/start", "/api/camera/start"):
            daemon.start_camera()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "is_camera_active": True}).encode("utf-8"))

        elif path in ("/api/v1/camera/stop", "/api/camera/stop"):
            daemon.stop_camera()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "is_camera_active": False}).encode("utf-8"))

        elif path in ("/api/v1/camera/toggle", "/api/camera/toggle"):
            active = daemon.toggle_camera()
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "ok", "is_camera_active": active}).encode("utf-8"))

        elif path in ("/api/v1/profiles/switch", "/api/profiles/switch"):
            profile_id = data.get("profile_id")
            if not profile_id:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Missing profile_id in body"}).encode("utf-8"))
                return
            ok = daemon.switch_user_profile(profile_id)
            if ok:
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok", "active_profile_id": profile_id}).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Profile '{profile_id}' not found"}).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found", "path": path}).encode("utf-8"))


class OculiSenseAPIServer:
    """
    Lightweight REST API service for OculiSenseAI integration with wellness platforms.
    """

    _active_servers: Dict[int, OculiSenseAPIServer] = {}

    @classmethod
    def get_active_server(cls, port: int) -> Optional[OculiSenseAPIServer]:
        return cls._active_servers.get(port)

    def __init__(self, daemon: OculiSenseDaemon, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.daemon = daemon
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return

        handler_cls = type(
            "BoundAPIRequestHandler",
            (_APIRequestHandler,),
            {"daemon_instance": self.daemon},
        )
        self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        OculiSenseAPIServer._active_servers[self.port] = self

        def _serve():
            logger.info("OculiSense REST API running on http://%s:%d", self.host, self.port)
            try:
                if self._server is not None:
                    self._server.serve_forever()
            except Exception as e:
                logger.debug("API server encountered exception: %s", e)

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        OculiSenseAPIServer._active_servers.pop(self.port, None)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("OculiSense REST API stopped.")

    def handle_request_in_process(
        self, path: str, method: str = "GET", data: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Process an API request in-memory without TCP socket overhead or network sandbox restrictions.
        """
        import io

        body_bytes = json.dumps(data).encode("utf-8") if data is not None else b""
        raw_http = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n\r\n"
        ).encode("utf-8") + body_bytes

        class _MockSocket:
            def __init__(self, payload: bytes):
                self._r = io.BytesIO(payload)
                self._w = io.BytesIO()

            def makefile(self, mode, *args, **kwargs):
                return self._r if "r" in mode else self._w

            def sendall(self, b):
                self._w.write(b)

        sock = _MockSocket(raw_http)
        handler_cls = type(
            "BoundAPIRequestHandler",
            (_APIRequestHandler,),
            {"daemon_instance": self.daemon},
        )
        handler_cls(sock, (self.host, self.port), self._server)

        out = sock._w.getvalue()
        parts = out.split(b"\r\n\r\n", 1)
        header_part = parts[0].decode("utf-8", errors="replace")
        body_part = parts[1].decode("utf-8", errors="replace") if len(parts) > 1 else "{}"

        status_line = header_part.splitlines()[0] if header_part else ""
        status_code = int(status_line.split()[1]) if len(status_line.split()) > 1 else 200
        try:
            res_data = json.loads(body_part)
        except Exception:
            res_data = {"raw": body_part}
        return status_code, res_data
