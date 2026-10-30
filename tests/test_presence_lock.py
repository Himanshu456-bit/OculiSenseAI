"""
Unit tests for the PresenceLockManager and workstation security layer.
"""

from oculisense.config import SecurityConfig
from oculisense.security.presence_lock import PresenceLockManager


def test_presence_state_transitions():
    cfg = SecurityConfig(away_timeout_sec=5.0, dry_run_lock=True, enable_presence_lock=True)
    lock_called = False

    def on_lock():
        nonlocal lock_called
        lock_called = True

    mgr = PresenceLockManager(config=cfg, on_lock_callback=on_lock)
    t = 1000.0

    # User present
    res = mgr.update(eyes_present=True, timestamp=t)
    assert res.is_present
    assert not res.lock_triggered
    assert res.status_label == "PRESENT"

    # User steps away for 2 seconds (under timeout)
    t += 2.0
    res = mgr.update(eyes_present=False, timestamp=t)
    assert not res.is_present
    assert not res.lock_triggered
    assert not lock_called

    # User stays away past 5 seconds -> triggers lock
    t += 4.0  # now 6.0 seconds away
    res = mgr.update(eyes_present=False, timestamp=t)
    assert not res.is_present
    assert res.lock_triggered
    assert lock_called
    assert res.status_label == "LOCKED"

    # User returns -> resets presence
    t += 1.0
    res = mgr.update(eyes_present=True, timestamp=t)
    assert res.is_present
    assert res.status_label == "PRESENT"
