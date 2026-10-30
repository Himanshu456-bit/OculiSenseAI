"""
Unit tests for OculiSenseAI REST API server endpoints.
"""

from __future__ import annotations
import json
import os
import tempfile
import time
import urllib.request
from oculisense.config import AppConfig, UserProfile
from oculisense.daemon import OculiSenseDaemon


def _make_request(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    try:
        with urllib.request.urlopen(req, data=body, timeout=3.0) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content)
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(content)
        except Exception:
            return e.code, {"error": content}
    except urllib.error.URLError:
        from urllib.parse import urlparse
        from oculisense.api import OculiSenseAPIServer
        parsed = urlparse(url)
        srv = OculiSenseAPIServer.get_active_server(parsed.port or 8765)
        if srv is not None:
            path_and_query = parsed.path + (("?" + parsed.query) if parsed.query else "")
            return srv.handle_request_in_process(path_and_query, method=method, data=data)
        raise


def test_api_status_and_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "api_test.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)
        # Step once to establish state
        daemon.step()

        port = 8781
        assert daemon.start_api(port=port)
        time.sleep(0.1)

        base_url = f"http://127.0.0.1:{port}"

        # 1. GET /api/v1/status
        status, data = _make_request(f"{base_url}/api/v1/status")
        assert status == 200
        assert "vision" in data
        assert "blink" in data
        assert "fatigue" in data
        assert "strain" in data
        assert "breaks" in data
        assert "presence" in data
        assert data["vision"]["is_synthetic"] is True

        # 2. GET /api/v1/metrics
        status, data = _make_request(f"{base_url}/api/v1/metrics")
        assert status == 200
        assert "daily_trends" in data
        assert "recent_sessions" in data

        # 3. GET /api/v1/breaks
        status, data = _make_request(f"{base_url}/api/v1/breaks")
        assert status == 200
        assert "break_events" in data
        assert "compliance_rate_pct" in data

        daemon.stop()


def test_api_break_actions_and_profiles():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "api_actions.db")
        cfg = AppConfig()
        cfg.privacy.database_filename = db_path
        cfg.demo_mode = True

        daemon = OculiSenseDaemon(config=cfg, force_synthetic=True)
        # Save a secondary profile
        prof_carol = UserProfile(
            profile_id="prof_carol",
            name="Carol",
            baseline_open_ear=0.35,
            baseline_closed_ear=0.17,
        )
        daemon.db.save_user_profile(prof_carol)

        port = 8782
        assert daemon.start_api(port=port)
        time.sleep(0.1)

        base_url = f"http://127.0.0.1:{port}"

        # 1. POST /api/v1/breaks/trigger
        status, data = _make_request(
            f"{base_url}/api/v1/breaks/trigger",
            method="POST",
            data={"reason": "WELLNESS_APP_REMINDER"},
        )
        assert status == 200
        assert data["status"] == "ok"
        assert daemon.break_advisor._state == "PROMPTED"

        # 2. POST /api/v1/breaks/snooze
        status, data = _make_request(
            f"{base_url}/api/v1/breaks/snooze",
            method="POST",
            data={},
        )
        assert status == 200
        assert data["snoozed"] is True
        assert daemon.break_advisor._state == "SNOOZED"

        # 3. POST /api/v1/breaks/complete
        status, data = _make_request(
            f"{base_url}/api/v1/breaks/complete",
            method="POST",
            data={},
        )
        assert status == 200
        assert data["status"] == "ok"
        assert daemon.break_advisor._state == "WORKING"

        # 4. GET /api/v1/profiles
        status, data = _make_request(f"{base_url}/api/v1/profiles")
        assert status == 200
        assert "profiles" in data
        profile_ids = [p["profile_id"] for p in data["profiles"]]
        assert "prof_carol" in profile_ids

        # 5. POST /api/v1/profiles/switch
        status, data = _make_request(
            f"{base_url}/api/v1/profiles/switch",
            method="POST",
            data={"profile_id": "prof_carol"},
        )
        assert status == 200
        assert data["active_profile_id"] == "prof_carol"
        assert daemon.config.profile.name == "Carol"

        daemon.stop()
