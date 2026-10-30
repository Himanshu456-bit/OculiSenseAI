"""
Standalone Test Runner for OculiSenseAI.
Discovers and executes all test functions without requiring external test runners.
"""

from __future__ import annotations
import inspect
import os
import sys
import time
import traceback

# Ensure root package is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import tests.test_geometry as t_geom
import tests.test_blink_detector as t_blink
import tests.test_squint_detector as t_squint
import tests.test_strain_analyzer as t_strain
import tests.test_fatigue_model as t_fatigue
import tests.test_break_advisor as t_break
import tests.test_database as t_db
import tests.test_presence_lock as t_lock
import tests.test_daemon_pipeline as t_daemon
import tests.test_api as t_api
import tests.test_camera_control as t_camera

MODULES_TO_TEST = [
    ("Geometry & EAR Math", t_geom),
    ("Blink Detection & PERCLOS", t_blink),
    ("Squint & Micro-Expression Detection", t_squint),
    ("Digital Eye Strain Index", t_strain),
    ("Workload Fatigue Model", t_fatigue),
    ("Break Advisor & 20-20-20 Rule", t_break),
    ("SQLite Storage & Privacy Purge", t_db),
    ("Presence & Auto-Lock Security", t_lock),
    ("Daemon Pipeline & Integration", t_daemon),
    ("Third-Party Wellness REST API", t_api),
    ("Camera Toggle & Rate Limiting", t_camera),
]


def run_all():
    print("=" * 70)
    print("  OculiSenseAI — Automated Biometric Test Suite")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    start_time = time.time()

    for module_name, mod in MODULES_TO_TEST:
        print(f"\n[SUITE] {module_name}")
        # Find all functions starting with test_
        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in test_funcs:
            total_tests += 1
            try:
                func()
                print(f"  [PASS] {name:<45}")
                passed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {name:<45} Error: {e}")
                traceback.print_exc()
                failed_tests += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {total_tests} executed in {elapsed:.3f}s")
    print(f"  PASSED: {passed_tests}")
    print(f"  FAILED: {failed_tests}")
    print("=" * 70)

    if failed_tests > 0:
        sys.exit(1)
    else:
        print("\nAll OculiSenseAI verification suites passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
