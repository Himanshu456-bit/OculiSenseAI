"""
OculiSenseAI — Main Application Launcher.
An Edge-AI Biometric Protection System for Digital Eye Strain & Workload Fatigue.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time

# Ensure oculisense is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oculisense.config import AppConfig
from oculisense.daemon import OculiSenseDaemon
from oculisense.storage.exporter import ReportExporter


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_daemon_mode(daemon: OculiSenseDaemon, interval_sec: float = 1.0) -> None:
    """Run daemon headlessly and log status to terminal."""
    print("=" * 65)
    print("  OculiSenseAI — Headless Background Daemon Active")
    print("  Privacy-First Edge Processing • Press Ctrl+C to stop")
    if not daemon.is_camera_active:
        print("  [NOTE] Camera is OFF by default. Pass --start-camera to start immediately.")
    print("=" * 65)

    daemon.start(target_fps=30)
    try:
        while True:
            time.sleep(interval_sec)
            if not daemon.is_camera_active:
                sys.stdout.write("\r[STATUS: CAMERA OFF] Tracking idle. Zero calculations running...    ")
                sys.stdout.flush()
                continue

            s = daemon._last_snapshot or daemon.step()
            status_text = "PRESENT" if s.presence.is_present else "AWAY"
            blink_bpm = s.blink.blink_rate_bpm
            fatigue = s.fatigue.fatigue_score
            strain = s.strain.strain_score
            perclos_pct = s.blink.perclos * 100.0
            next_break_s = int(s.breaks.remaining_until_break_sec)

            src_tag = s.vision.display_indicator
            if s.vision.source_status in ("NO_CAMERA", "SEARCHING_CAMERA"):
                sys.stdout.write(
                    f"\r[{src_tag}] [STATUS: SEARCHING] Probing physical webcam index {daemon.tracker.active_camera_id}...                    "
                )
                sys.stdout.flush()
                continue

            if not s.vision.face_detected:
                sys.stdout.write(
                    f"\r[{src_tag}] [STATUS: AWAY   ] Searching for face... (camera active, low CPU mode)                    "
                )
                sys.stdout.flush()
                continue

            sys.stdout.write(
                f"\r[{src_tag}] [STATUS: {status_text:<7}] "
                f"EAR: {s.vision.ear_avg:.2f} | "
                f"Blink: {blink_bpm:4.1f} BPM | "
                f"PERCLOS: {perclos_pct:4.1f}% | "
                f"Fatigue: {fatigue:4.1f}/100 | "
                f"Strain: {strain:4.1f}/100 | "
                f"Break in: {next_break_s:3d}s "
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopping OculiSenseAI daemon...")
    finally:
        daemon.stop()
        print("Daemon stopped. Session successfully saved to local SQLite database.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OculiSenseAI: Edge-AI Eye Strain & Workload Fatigue Protection Daemon"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch desktop graphical interface (default)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Launch directly in compact floating Desktop Overlay HUD mode",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run headlessly as background daemon without GUI",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force synthetic camera feed (test without webcam)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and list all detected physical webcam device indices and exit",
    )
    parser.add_argument(
        "--export-report",
        type=str,
        default=None,
        help="Export HTML summary report to specified path and exit",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Export telemetry archive in JSON format to specified path and exit",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export daily telemetry metrics in CSV format to specified path and exit",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Activate a specific personalized user profile ID",
    )
    parser.add_argument(
        "--pomodoro",
        action="store_true",
        help="Enable 25-minute Pomodoro focus with 5-minute eye rest intervals",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        default=True,
        help="Enable spoken voice alerts for rest breaks (default: enabled)",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable spoken voice alerts (audio chimes only)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start local REST API service for third-party wellness app integration",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8765,
        help="Port for local REST API service (default: 8765)",
    )
    parser.add_argument(
        "--start-camera",
        dest="start_camera",
        action="store_true",
        default=True,
        help="Immediately start camera upon launch (default: enabled)",
    )
    parser.add_argument(
        "--no-camera",
        dest="start_camera",
        action="store_false",
        help="Do not start camera immediately upon launch",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list_cameras:
        from oculisense.vision.tracker import detect_available_cameras
        print("=" * 60)
        print("  OculiSenseAI — Physical Webcam Hardware Scanner")
        print("=" * 60)
        found = detect_available_cameras(max_probe=6)
        if found:
            print(f"Detected {len(found)} accessible physical webcam(s):")
            for idx in found:
                print(f"  ● Camera Index {idx} [LIVE WEBCAM {idx}]")
            print("\nTo launch OculiSense with a specific camera, run:")
            print(f"  python main.py --camera {found[0]}")
        else:
            print("No accessible physical webcams detected on indices 0-5.")
            print("Ensure camera drivers are installed and camera privacy permissions are enabled.")
        return

    # Load configuration
    config = AppConfig.load()
    config.demo_mode = bool(args.demo)
    config.vision.camera_id = args.camera
    if args.pomodoro:
        config.breaks.enable_pomodoro = True
    if args.no_voice:
        config.breaks.enable_voice_alert = False

    daemon = OculiSenseDaemon(
        config=config,
        force_synthetic=config.demo_mode,
        autostart_camera=args.start_camera,
    )

    if args.profile:
        daemon.switch_user_profile(args.profile)

    if args.api:
        daemon.start_api(port=args.api_port)
        print(f"OculiSense REST API active on http://127.0.0.1:{args.api_port}/api/v1/status")

    # Handle export requests
    if args.export_report:
        exporter = ReportExporter(daemon.db)
        exporter.export_html_report(args.export_report)
        print(f"HTML Report exported successfully to: {args.export_report}")
        return

    if args.export_json:
        exporter = ReportExporter(daemon.db)
        exporter.export_json(args.export_json)
        print(f"JSON archive exported successfully to: {args.export_json}")
        return

    if args.export_csv:
        exporter = ReportExporter(daemon.db)
        exporter.export_csv(args.export_csv)
        print(f"CSV metrics exported successfully to: {args.export_csv}")
        return

    # Choose run mode
    if args.daemon:
        run_daemon_mode(daemon)
    else:
        # Launch GUI
        try:
            import tkinter as tk
            from oculisense.ui.app import OculiSenseApp

            app = OculiSenseApp(daemon, start_in_overlay=args.overlay)
            app.run()
        except Exception as e:
            print(f"GUI display not available ({e}). Falling back to headless daemon mode.")
            run_daemon_mode(daemon)


if __name__ == "__main__":
    main()
