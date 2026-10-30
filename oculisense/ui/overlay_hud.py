"""
Floating Desktop Overlay HUD Widget for OculiSenseAI.
Stays always-on-top of the user's active applications, providing unobtrusive
real-time biometric monitoring of eye strain, blinks per minute, EAR, and fatigue.
"""

from __future__ import annotations
import tkinter as tk
from typing import Callable, Optional

from oculisense.daemon import DaemonSnapshot
from oculisense.ui.components import (
    COLOR_ACCENT_AMBER,
    COLOR_ACCENT_CYAN,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
    COLOR_BG_CARD,
    COLOR_BG_DARK,
    COLOR_BORDER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
)


class OverlayHUDWindow(tk.Toplevel):
    """
    Compact, draggable, always-on-top desktop overlay that monitors user activity
    while they work across code editors, browsers, spreadsheets, and games.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_expand_dashboard: Callable[[], None],
        on_toggle_camera: Optional[Callable[[], None]] = None,
        on_force_rest: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.on_expand_dashboard = on_expand_dashboard
        self.on_toggle_camera = on_toggle_camera
        self.on_force_rest = on_force_rest

        self.title("OculiSenseAI Overlay")
        self.configure(bg="#020617")
        self.geometry("380x140")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        # Remove system window frame if supported, or provide custom styled draggable border
        try:
            self.overrideredirect(True)
        except Exception:
            pass

        # Position at top-right of primary screen
        try:
            sw = self.winfo_screenwidth()
            self.geometry(f"380x140+{sw - 400}+40")
        except Exception:
            pass

        # Enable mouse dragging anywhere on header
        self._drag_start_x = 0
        self._drag_start_y = 0

        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = tk.Frame(
            self,
            bg="#0f172a",
            bd=1,
            relief="solid",
            highlightbackground="#334155",
            highlightthickness=1,
        )
        main_frame.pack(fill="both", expand=True)

        # Top Draggable Header
        header = tk.Frame(main_frame, bg="#1e293b", height=28, cursor="fleur")
        header.pack(fill="x", side="top")
        header.bind("<Button-1>", self._start_drag)
        header.bind("<B1-Motion>", self._do_drag)

        lbl_drag = tk.Label(
            header,
            text="⣿ 👁️ OculiSense Overlay",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT_CYAN,
            bg="#1e293b",
            cursor="fleur",
        )
        lbl_drag.pack(side="left", padx=8, pady=3)
        lbl_drag.bind("<Button-1>", self._start_drag)
        lbl_drag.bind("<B1-Motion>", self._do_drag)

        # Health badge in header
        self.lbl_score_badge = tk.Label(
            header,
            text="100% FRESH",
            font=("Segoe UI", 8, "bold"),
            fg="#052e16",
            bg=COLOR_ACCENT_GREEN,
            padx=6,
            pady=1,
        )
        self.lbl_score_badge.pack(side="left", padx=4)

        # Header controls
        btn_expand = tk.Label(
            header,
            text="⛶ Dashboard",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg="#334155",
            padx=6,
            pady=1,
            cursor="hand2",
        )
        btn_expand.pack(side="right", padx=6)
        btn_expand.bind("<Button-1>", lambda e: self.on_expand_dashboard())

        # Middle Telemetry Grid
        grid_frame = tk.Frame(main_frame, bg="#0f172a")
        grid_frame.pack(fill="both", expand=True, padx=10, pady=6)

        # Metric 1: EAR
        col1 = tk.Frame(grid_frame, bg="#0f172a")
        col1.pack(side="left", expand=True, fill="both")
        tk.Label(col1, text="EAR", font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg="#0f172a").pack(anchor="w")
        self.lbl_ear = tk.Label(col1, text="0.000", font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT_CYAN, bg="#0f172a")
        self.lbl_ear.pack(anchor="w")

        # Metric 2: BPM
        col2 = tk.Frame(grid_frame, bg="#0f172a")
        col2.pack(side="left", expand=True, fill="both")
        tk.Label(col2, text="BLINK RATE", font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg="#0f172a").pack(anchor="w")
        self.lbl_bpm = tk.Label(col2, text="0.0 BPM", font=("Segoe UI", 12, "bold"), fg=COLOR_ACCENT_GREEN, bg="#0f172a")
        self.lbl_bpm.pack(anchor="w")

        # Metric 3: Strain
        col3 = tk.Frame(grid_frame, bg="#0f172a")
        col3.pack(side="left", expand=True, fill="both")
        tk.Label(col3, text="EYE STRAIN", font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg="#0f172a").pack(anchor="w")
        self.lbl_strain = tk.Label(col3, text="0 / 100", font=("Segoe UI", 12, "bold"), fg=COLOR_TEXT_MAIN, bg="#0f172a")
        self.lbl_strain.pack(anchor="w")

        # Metric 4: Fatigue
        col4 = tk.Frame(grid_frame, bg="#0f172a")
        col4.pack(side="left", expand=True, fill="both")
        tk.Label(col4, text="FATIGUE", font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg="#0f172a").pack(anchor="w")
        self.lbl_fatigue = tk.Label(col4, text="0 / 100", font=("Segoe UI", 12, "bold"), fg=COLOR_TEXT_MAIN, bg="#0f172a")
        self.lbl_fatigue.pack(anchor="w")

        # Bottom Quick Action Bar
        bottom = tk.Frame(main_frame, bg="#090d16", height=26)
        bottom.pack(fill="x", side="bottom", padx=6, pady=4)

        self.lbl_status = tk.Label(
            bottom,
            text="● INITIALIZING",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#090d16",
        )
        self.lbl_status.pack(side="left", padx=4)

        if self.on_force_rest:
            btn_rest = tk.Label(
                bottom,
                text="🛑 Force Rest",
                font=("Segoe UI", 7, "bold"),
                fg="#fca5a5",
                bg="#450a0a",
                padx=6,
                pady=1,
                cursor="hand2",
            )
            btn_rest.pack(side="right", padx=3)
            btn_rest.bind("<Button-1>", lambda e: self.on_force_rest())

        if self.on_toggle_camera:
            self.btn_cam = tk.Label(
                bottom,
                text="📷 Cam",
                font=("Segoe UI", 7, "bold"),
                fg=COLOR_TEXT_MAIN,
                bg="#1e293b",
                padx=6,
                pady=1,
                cursor="hand2",
            )
            self.btn_cam.pack(side="right", padx=3)
            self.btn_cam.bind("<Button-1>", lambda e: self.on_toggle_camera())

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _do_drag(self, event: tk.Event) -> None:
        deltax = event.x - self._drag_start_x
        deltay = event.y - self._drag_start_y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")

    def update_telemetry(self, s: DaemonSnapshot) -> None:
        """Update live telemetry badges and labels from DaemonSnapshot."""
        if not self.winfo_exists():
            return

        # Overall Score & Badge
        score = s.overall_score
        if score >= 80.0:
            badge_text = f"{score:.0f}% FRESH"
            badge_fg, badge_bg = "#052e16", COLOR_ACCENT_GREEN
        elif score >= 60.0:
            badge_text = f"{score:.0f}% GOOD"
            badge_fg, badge_bg = "#082f49", COLOR_ACCENT_CYAN
        elif score >= 40.0:
            badge_text = f"{score:.0f}% STRAINED"
            badge_fg, badge_bg = "#451a03", COLOR_ACCENT_AMBER
        else:
            badge_text = f"{score:.0f}% CRITICAL"
            badge_fg, badge_bg = "#450a0a", COLOR_ACCENT_RED

        self.lbl_score_badge.config(text=badge_text, fg=badge_fg, bg=badge_bg)

        # EAR
        self.lbl_ear.config(text=f"{s.vision.ear_avg:.3f}")
        if s.blink.is_eyes_closed:
            self.lbl_ear.config(fg=COLOR_ACCENT_AMBER)
        else:
            self.lbl_ear.config(fg=COLOR_ACCENT_CYAN)

        # BPM
        bpm = s.blink.blink_rate_bpm
        self.lbl_bpm.config(text=f"{bpm:.1f} BPM")
        if bpm < 10.0 and s.vision.face_detected:
            self.lbl_bpm.config(fg=COLOR_ACCENT_AMBER)
        else:
            self.lbl_bpm.config(fg=COLOR_ACCENT_GREEN)

        # Strain & Fatigue
        self.lbl_strain.config(text=f"{s.strain.strain_score:.0f} / 100")
        if s.strain.strain_score >= 70.0:
            self.lbl_strain.config(fg=COLOR_ACCENT_RED)
        elif s.strain.strain_score >= 40.0:
            self.lbl_strain.config(fg=COLOR_ACCENT_AMBER)
        else:
            self.lbl_strain.config(fg=COLOR_TEXT_MAIN)

        self.lbl_fatigue.config(text=f"{s.fatigue.fatigue_score:.0f} / 100")
        if s.fatigue.fatigue_score >= 75.0:
            self.lbl_fatigue.config(fg=COLOR_ACCENT_RED)
        elif s.fatigue.fatigue_score >= 45.0:
            self.lbl_fatigue.config(fg=COLOR_ACCENT_AMBER)
        else:
            self.lbl_fatigue.config(fg=COLOR_TEXT_MAIN)

        # Status text
        if not s.is_camera_active:
            self.lbl_status.config(text="● CAMERA OFF", fg=COLOR_TEXT_MUTED)
        elif not s.vision.face_detected:
            self.lbl_status.config(text="● SEARCHING FACE", fg=COLOR_ACCENT_AMBER)
        elif s.squint.is_squinting:
            self.lbl_status.config(text="● SQUINTING", fg=COLOR_ACCENT_AMBER)
        elif s.blink.is_eyes_closed:
            self.lbl_status.config(text="● BLINK / CLOSED", fg=COLOR_ACCENT_CYAN)
        else:
            self.lbl_status.config(text="● TRACKING ACTIVE", fg=COLOR_ACCENT_GREEN)
