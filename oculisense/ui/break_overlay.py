"""
Gentle 20-20-20 Break Reminder & Forced Rest Lockdown Overlay Window.
Encourages ocular relaxation, counts down rest periods, tracks biometric compliance,
and enforces workstation activity pauses when critical eye strain is detected.
"""

from __future__ import annotations
import tkinter as tk
from typing import Callable, Optional

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


class BreakOverlayWindow(tk.Toplevel):
    """
    Rest reminder dialog and forced rest overlay with countdown and compliance detection.
    Guaranteed singleton: at most ONE break dialog can ever exist at a time.
    """
    _active_instance: Optional[BreakOverlayWindow] = None

    def __new__(cls, parent: tk.Widget, *args, **kwargs) -> BreakOverlayWindow:
        if (
            cls._active_instance is not None
            and cls._active_instance.winfo_exists()
        ):
            try:
                cls._active_instance.lift()
                cls._active_instance.focus_force()
            except Exception:
                pass
            return cls._active_instance
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        parent: tk.Widget,
        on_snooze: Optional[Callable[[], None]] = None,
        on_skip: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        is_forced: bool = False,
        overall_score: float = 100.0,
        strain_score: float = 0.0,
        fatigue_score: float = 0.0,
        bpm: float = 16.0,
        ear: float = 0.30,
        least_ear: float = 0.30,
        on_lock_workstation: Optional[Callable[[], None]] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if on_snooze is not None:
                self.on_snooze = on_snooze
            if on_skip is not None:
                self.on_skip = on_skip
            if on_complete is not None:
                self.on_complete = on_complete
            if on_lock_workstation is not None:
                self.on_lock_workstation = on_lock_workstation
            return

        super().__init__(parent)
        BreakOverlayWindow._active_instance = self
        self._initialized = True
        self.on_snooze = on_snooze
        self.on_skip = on_skip
        self.on_complete = on_complete
        self.on_lock_workstation = on_lock_workstation
        self.is_forced = is_forced
        self.overall_score = overall_score
        self.strain_score = strain_score
        self.fatigue_score = fatigue_score
        self.bpm = bpm
        self.ear = ear
        self.least_ear = least_ear

        self.title("OculiSenseAI — Workstation Rest Enforcement")
        self.configure(bg="#050814" if self.is_forced else COLOR_BG_DARK)
        self.attributes("-topmost", True)

        if self.is_forced:
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                self.state("zoomed")
            try:
                self.grab_set()
            except Exception:
                pass
        else:
            self.geometry("480x350")
            self.resizable(False, False)
            try:
                self.eval(f"tk::PlaceWindow {str(self)} center")
            except Exception:
                pass

        # Intercept window manager close button ('X') as skip to avoid infinite popup loop
        self.protocol("WM_DELETE_WINDOW", self._do_skip)

        self._build_ui()

    def _build_ui(self) -> None:
        if self.is_forced:
            self._build_forced_lockdown_ui()
        else:
            self._build_standard_ui()

    def _build_standard_ui(self) -> None:
        main_frame = tk.Frame(
            self,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground=COLOR_BORDER,
        )
        main_frame.pack(fill="both", expand=True, padx=16, pady=16)

        # Badge
        lbl_badge = tk.Label(
            main_frame,
            text="👁️ 20-20-20 HEALTH BREAK",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT_CYAN,
            bg=COLOR_BG_CARD,
        )
        lbl_badge.pack(pady=(16, 4))

        # Title
        lbl_title = tk.Label(
            main_frame,
            text="Give Your Eyes a 20-Second Rest",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_CARD,
        )
        lbl_title.pack(pady=(0, 6))

        # Scientific guidance
        lbl_desc = tk.Label(
            main_frame,
            text="Look at an object at least 20 feet (6m) away.\nThis allows your ciliary eye muscles to completely un-focus and relax.",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
            justify="center",
        )
        lbl_desc.pack(pady=(0, 10))

        # Countdown Label
        self.lbl_countdown = tk.Label(
            main_frame,
            text="20s",
            font=("Segoe UI", 36, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg=COLOR_BG_CARD,
        )
        self.lbl_countdown.pack(pady=(0, 4))

        # Biometric Compliance status
        self.lbl_status = tk.Label(
            main_frame,
            text="Biometric tracking active: looking away will be recorded.",
            font=("Segoe UI", 8, "italic"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
        )
        self.lbl_status.pack(pady=(0, 14))

        # Buttons frame
        btn_frame = tk.Frame(main_frame, bg=COLOR_BG_CARD)
        btn_frame.pack(fill="x", padx=20)

        btn_snooze = tk.Button(
            btn_frame,
            text="Snooze (5 min)",
            command=self._do_snooze,
            bg="#334155",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 9),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
        )
        btn_snooze.pack(side="left", expand=True, padx=4)

        btn_skip = tk.Button(
            btn_frame,
            text="Skip Break",
            command=self._do_skip,
            bg="#27272a",
            fg="#a1a1aa",
            font=("Segoe UI", 9),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
        )
        btn_skip.pack(side="right", expand=True, padx=4)

    def _build_forced_lockdown_ui(self) -> None:
        """Builds full-screen modal lock enforcement overlay."""
        container = tk.Frame(self, bg="#050814")
        container.place(relx=0.5, rely=0.5, anchor="center")

        card = tk.Frame(
            container,
            bg="#0f172a",
            bd=2,
            relief="solid",
            highlightbackground=COLOR_ACCENT_RED,
            padx=40,
            pady=30,
        )
        card.pack()

        # Warning Header
        tk.Label(
            card,
            text="⛔ FORCED WORKSTATION REST LOCKDOWN",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_ACCENT_RED,
            bg="#0f172a",
        ).pack(pady=(0, 6))

        tk.Label(
            card,
            text="CRITICAL DIGITAL EYE STRAIN & WORKLOAD FATIGUE DETECTED",
            font=("Segoe UI", 12, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg="#0f172a",
        ).pack(pady=(0, 10))

        # Score & Telemetry Banner
        self.lbl_metrics_banner = tk.Label(
            card,
            text=(
                f"Overall Eye Wellness Score: {self.overall_score:.0f}% [CRITICAL LOW]\n"
                f"Eye Strain: {self.strain_score:.1f}/100  |  Fatigue: {self.fatigue_score:.1f}/100  |  "
                f"Blinks: {self.bpm:.1f} BPM  |  EAR: {self.ear:.3f}  |  Least EAR (20s): {self.least_ear:.3f}"
            ),
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT_AMBER,
            bg="#1e293b",
            padx=16,
            pady=8,
            relief="groove",
            justify="center",
        )
        self.lbl_metrics_banner.pack(fill="x", pady=(0, 16))

        # Instructions
        tk.Label(
            card,
            text=(
                "Your workstation activity has been temporarily paused to protect your eyesight.\n"
                "Screens suppress blinking by over 60%, starving your corneas of lubrication.\n\n"
                "PLEASE STEP AWAY FROM THIS DEVICE FOR A WHILE AND LET YOUR EYES REST.\n"
                "Look out of a window or at an object at least 20 feet away to relax ciliary muscles."
            ),
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#0f172a",
            justify="center",
        ).pack(pady=(0, 16))

        # Big Countdown
        self.lbl_countdown = tk.Label(
            card,
            text="20s",
            font=("Segoe UI", 48, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg="#0f172a",
        )
        self.lbl_countdown.pack(pady=(0, 4))

        # Biometric Compliance status
        self.lbl_status = tk.Label(
            card,
            text="🔍 Biometric Sensor Active: Waiting for user to step away or close eyes...",
            font=("Segoe UI", 10, "italic"),
            fg=COLOR_ACCENT_CYAN,
            bg="#0f172a",
        )
        self.lbl_status.pack(pady=(0, 20))

        # Action Buttons
        btn_frame = tk.Frame(card, bg="#0f172a")
        btn_frame.pack(fill="x")

        btn_lock = tk.Button(
            btn_frame,
            text="🔒 Lock Workstation Screen Now (Win+L)",
            command=self._do_lock_workstation,
            bg="#dc2626",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        )
        btn_lock.pack(side="left", padx=8, expand=True)

        btn_override = tk.Button(
            btn_frame,
            text="⚠️ Emergency Override",
            command=self._do_override,
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
        )
        btn_override.pack(side="right", padx=8, expand=True)

    def _do_lock_workstation(self) -> None:
        """Trigger OS screen lock."""
        if self.on_lock_workstation:
            try:
                self.on_lock_workstation()
            except Exception:
                pass

    def _do_override(self) -> None:
        """Safe emergency override with confirmation dialog."""
        from tkinter import messagebox
        confirm = messagebox.askyesno(
            "Emergency Override Confirmation",
            "Your eyes are under severe physiological strain.\n\n"
            "Are you sure you want to bypass this rest lockdown?",
            parent=self,
        )
        if confirm:
            self._do_skip()

    def update_countdown(
        self,
        seconds_remaining: float,
        is_in_rest: bool,
        overall_score: Optional[float] = None,
        bpm: Optional[float] = None,
        ear: Optional[float] = None,
        least_ear: Optional[float] = None,
    ) -> None:
        """Update live countdown and telemetry from daemon status."""
        sec = max(0, int(round(seconds_remaining)))
        self.lbl_countdown.config(text=f"{sec}s")

        if is_in_rest:
            self.lbl_status.config(
                text="✓ Rest verified: Eyes resting / away from screen.",
                fg=COLOR_ACCENT_GREEN,
            )
        else:
            if self.is_forced:
                self.lbl_status.config(
                    text="⚠️ Please step away from your display or close your eyes now.",
                    fg=COLOR_ACCENT_AMBER,
                )
            else:
                self.lbl_status.config(
                    text="Please look away from your display.",
                    fg=COLOR_TEXT_MUTED,
                )

        if overall_score is not None:
            self.overall_score = overall_score
        if bpm is not None:
            self.bpm = bpm
        if ear is not None:
            self.ear = ear
        if least_ear is not None:
            self.least_ear = least_ear

        if hasattr(self, "lbl_metrics_banner") and self.lbl_metrics_banner.winfo_exists():
            self.lbl_metrics_banner.config(
                text=(
                    f"Overall Eye Wellness Score: {self.overall_score:.0f}% [CRITICAL LOW]\n"
                    f"Eye Strain: {self.strain_score:.1f}/100  |  Fatigue: {self.fatigue_score:.1f}/100  |  "
                    f"Blinks: {self.bpm:.1f} BPM  |  EAR: {self.ear:.3f}  |  Least EAR (20s): {self.least_ear:.3f}"
                )
            )

        if sec <= 0:
            if self.on_complete:
                try:
                    self.on_complete()
                except Exception:
                    pass
            self.destroy()

    def _do_snooze(self) -> None:
        if self.on_snooze:
            try:
                self.on_snooze()
            except Exception:
                pass
        self.destroy()

    def _do_skip(self) -> None:
        if self.on_skip:
            try:
                self.on_skip()
            except Exception:
                pass
        self.destroy()

    def destroy(self) -> None:
        try:
            self.grab_release()
        except Exception:
            pass
        if BreakOverlayWindow._active_instance is self:
            BreakOverlayWindow._active_instance = None
        self._initialized = False
        try:
            super().destroy()
        except Exception:
            pass
