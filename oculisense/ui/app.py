"""
Main Desktop GUI Application for OculiSenseAI.
Features real-time biometric telemetry, eye aspect ratio waveforms,
interactive fatigue/strain gauges, break overlay management, and reporting.
"""

from __future__ import annotations
import math
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, List, Optional

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:
    Image, ImageTk = None, None

from oculisense.config import UserProfile
from oculisense.daemon import DaemonSnapshot, OculiSenseDaemon
from oculisense.storage.exporter import ReportExporter
from oculisense.ui.break_overlay import BreakOverlayWindow
from oculisense.ui.overlay_hud import OverlayHUDWindow
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
    CircularGauge,
    EARGraph,
)


class AnalyticsHistoryWindow(tk.Toplevel):
    """
    Ergonomics Analytics Dashboard window featuring:
    - 7-day daily trend charts (Mean Fatigue & Mean Strain).
    - Recorded Break Compliance History log.
    - Multi-user profile management and switching.
    """

    def __init__(
        self,
        parent: tk.Widget,
        daemon: OculiSenseDaemon,
        on_profile_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.daemon = daemon
        self.on_profile_changed = on_profile_changed

        self.title("OculiSenseAI — Analytics, Break History & Profiles")
        self.geometry("820x600")
        self.minsize(740, 520)
        self.configure(bg=COLOR_BG_DARK)

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        # Tab 1: Daily Trends Chart
        tab_trends = tk.Frame(notebook, bg=COLOR_BG_DARK)
        notebook.add(tab_trends, text=" 📊 Daily Trends ")
        self._build_trends_tab(tab_trends)

        # Tab 2: Break Compliance Log
        tab_breaks = tk.Frame(notebook, bg=COLOR_BG_DARK)
        notebook.add(tab_breaks, text=" 👁️ Break Compliance Log ")
        self._build_breaks_tab(tab_breaks)

        # Tab 3: User Profiles & Calibration
        tab_profiles = tk.Frame(notebook, bg=COLOR_BG_DARK)
        notebook.add(tab_profiles, text=" 👤 User Profiles ")
        self._build_profiles_tab(tab_profiles)

    def _build_trends_tab(self, parent: tk.Frame) -> None:
        daily = self.daemon.db.get_daily_metrics(days=7)

        header = tk.Frame(parent, bg=COLOR_BG_DARK)
        header.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(
            header,
            text="7-Day Ergonomics Trend Rollup",
            font=("Segoe UI", 12, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK,
        ).pack(side="left")

        legend = tk.Frame(header, bg=COLOR_BG_DARK)
        legend.pack(side="right")
        tk.Label(legend, text="■ Fatigue Score", font=("Segoe UI", 8, "bold"), fg=COLOR_ACCENT_AMBER, bg=COLOR_BG_DARK).pack(side="left", padx=6)
        tk.Label(legend, text="■ Strain Index", font=("Segoe UI", 8, "bold"), fg=COLOR_ACCENT_CYAN, bg=COLOR_BG_DARK).pack(side="left", padx=6)

        canvas = tk.Canvas(parent, bg=COLOR_BG_CARD, height=220, highlightthickness=1, highlightbackground=COLOR_BORDER)
        canvas.pack(fill="x", padx=12, pady=6)

        # Draw chart
        w = 780
        h = 220
        pad_l, pad_r, pad_t, pad_b = 50, 20, 25, 35

        # Horizontal gridlines
        for level in [25, 50, 75, 100]:
            y = pad_t + (100 - level) * (h - pad_t - pad_b) / 100.0
            canvas.create_line(pad_l, y, w - pad_r, y, fill="#273549", dash=(2, 4))
            canvas.create_text(pad_l - 6, y, text=f"{level}", fill=COLOR_TEXT_MUTED, anchor="e", font=("Segoe UI", 7))

        if daily:
            n = len(daily)
            slot_w = (w - pad_l - pad_r) / max(1, n)
            bar_w = max(10.0, min(24.0, slot_w * 0.3))

            for i, d in enumerate(daily):
                cx = pad_l + (i + 0.5) * slot_w
                fatigue_val = min(100.0, max(0.0, float(d.get("mean_fatigue") or 0.0)))
                strain_val = min(100.0, max(0.0, float(d.get("mean_strain") or 0.0)))

                chart_h = h - pad_t - pad_b
                y_fatigue = pad_t + (100.0 - fatigue_val) * (chart_h / 100.0)
                y_strain = pad_t + (100.0 - strain_val) * (chart_h / 100.0)
                base_y = h - pad_b

                # Fatigue bar (Amber)
                canvas.create_rectangle(cx - bar_w - 2, y_fatigue, cx - 2, base_y, fill=COLOR_ACCENT_AMBER, outline="")
                # Strain bar (Cyan)
                canvas.create_rectangle(cx + 2, y_strain, cx + bar_w + 2, base_y, fill=COLOR_ACCENT_CYAN, outline="")

                # Date label
                date_str = str(d.get("log_date", ""))[5:]
                canvas.create_text(cx, base_y + 14, text=date_str, fill=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        else:
            canvas.create_text(
                w / 2.0, h / 2.0,
                text="No daily sessions logged yet.\nWork with OculiSense to record 7-day trend metrics.",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 10),
                justify="center",
            )

        # Recent sessions table
        table_frame = tk.Frame(parent, bg=COLOR_BG_DARK)
        table_frame.pack(fill="both", expand=True, padx=12, pady=6)

        columns = ("date", "active_mins", "mean_fatigue", "mean_strain", "blinks", "squint_mins")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        tree.heading("date", text="Date")
        tree.heading("active_mins", text="Active Mins")
        tree.heading("mean_fatigue", text="Mean Fatigue")
        tree.heading("mean_strain", text="Mean Strain")
        tree.heading("blinks", text="Total Blinks")
        tree.heading("squint_mins", text="Squint Time")

        tree.column("date", width=120)
        tree.column("active_mins", width=90)
        tree.column("mean_fatigue", width=100)
        tree.column("mean_strain", width=100)
        tree.column("blinks", width=90)
        tree.column("squint_mins", width=100)

        for d in daily:
            tree.insert("", "end", values=(
                d.get("log_date", "-"),
                f"{d.get('active_minutes', 0)} min",
                f"{d.get('mean_fatigue', 0.0)}/100",
                f"{d.get('mean_strain', 0.0)}/100",
                d.get("total_blinks", 0),
                f"{d.get('total_squint_mins', 0.0)} min",
            ))
        tree.pack(fill="both", expand=True)

    def _build_breaks_tab(self, parent: tk.Frame) -> None:
        advisor = self.daemon.break_advisor
        events = self.daemon.db.get_break_events(limit=50)

        summary_box = tk.Frame(parent, bg=COLOR_BG_CARD, bd=1, relief="solid", highlightbackground=COLOR_BORDER)
        summary_box.pack(fill="x", padx=12, pady=8)

        stats = [
            ("TOTAL PROMPTED", f"{advisor._total_prompted}"),
            ("COMPLETED RESTS", f"{advisor._total_completed}"),
            ("SKIPPED RESTS", f"{advisor._total_skipped}"),
            ("ADHERENCE RATE", f"{advisor.compliance_rate:.1f}%"),
        ]
        for title, val in stats:
            card = tk.Frame(summary_box, bg=COLOR_BG_CARD, padx=16, pady=8)
            card.pack(side="left", expand=True)
            tk.Label(card, text=title, font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD).pack()
            tk.Label(card, text=val, font=("Segoe UI", 14, "bold"), fg=COLOR_ACCENT_GREEN if "%" in val else COLOR_TEXT_MAIN, bg=COLOR_BG_CARD).pack()

        table_frame = tk.Frame(parent, bg=COLOR_BG_DARK)
        table_frame.pack(fill="both", expand=True, padx=12, pady=6)

        cols = ("time", "reason", "outcome", "duration")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        tree.heading("time", text="Timestamp")
        tree.heading("reason", text="Trigger Reason")
        tree.heading("outcome", text="Outcome")
        tree.heading("duration", text="Rest Duration")

        tree.column("time", width=140)
        tree.column("reason", width=200)
        tree.column("outcome", width=120)
        tree.column("duration", width=100)

        for ev in events:
            import datetime
            ts_str = datetime.datetime.fromtimestamp(ev.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
            tree.insert("", "end", values=(
                ts_str,
                ev.get("reason", "-"),
                ev.get("outcome", "-"),
                f"{ev.get('duration_sec', 0):.1f}s",
            ))
        tree.pack(fill="both", expand=True)

    def _build_profiles_tab(self, parent: tk.Frame) -> None:
        profiles = self.daemon.db.list_user_profiles()
        curr_prof = self.daemon.config.profile

        active_box = tk.Frame(parent, bg=COLOR_BG_CARD, bd=1, relief="solid", highlightbackground=COLOR_BORDER, padx=16, pady=12)
        active_box.pack(fill="x", padx=12, pady=8)

        tk.Label(active_box, text="ACTIVE PERSONALIZED PROFILE", font=("Segoe UI", 9, "bold"), fg=COLOR_ACCENT_CYAN, bg=COLOR_BG_CARD).pack(anchor="w")
        info_text = (
            f"User Name: {curr_prof.name} (ID: {curr_prof.profile_id})\n"
            f"Open EAR Baseline: {curr_prof.baseline_open_ear:.3f} | "
            f"Closed EAR Baseline: {curr_prof.baseline_closed_ear:.3f}\n"
            f"Calibrated Threshold: {curr_prof.dynamic_ear_threshold:.3f} | "
            f"Squint Threshold: {curr_prof.dynamic_squint_threshold:.3f}\n"
            f"Last Calibrated: {curr_prof.calibrated_at or 'Uncalibrated Default'}"
        )
        self.lbl_profile_info = tk.Label(active_box, text=info_text, font=("Segoe UI", 9), fg=COLOR_TEXT_MAIN, bg=COLOR_BG_CARD, justify="left")
        self.lbl_profile_info.pack(anchor="w", pady=(4, 0))

        # Switch profile row
        switch_frame = tk.Frame(parent, bg=COLOR_BG_DARK)
        switch_frame.pack(fill="x", padx=12, pady=8)

        tk.Label(switch_frame, text="Switch User Profile:", font=("Segoe UI", 9), fg=COLOR_TEXT_MAIN, bg=COLOR_BG_DARK).pack(side="left", padx=4)

        profile_map = {f"{p.name} ({p.profile_id})": p.profile_id for p in profiles}
        combo_vals = list(profile_map.keys()) or [f"{curr_prof.name} ({curr_prof.profile_id})"]
        combo_prof = ttk.Combobox(switch_frame, values=combo_vals, state="readonly", width=28)
        combo_prof.set(combo_vals[0])
        combo_prof.pack(side="left", padx=6)

        def _do_switch():
            chosen = combo_prof.get()
            pid = profile_map.get(chosen)
            if pid:
                self.daemon.switch_user_profile(pid)
                p = self.daemon.config.profile
                new_info = (
                    f"User Name: {p.name} (ID: {p.profile_id})\n"
                    f"Open EAR Baseline: {p.baseline_open_ear:.3f} | "
                    f"Closed EAR Baseline: {p.baseline_closed_ear:.3f}\n"
                    f"Calibrated Threshold: {p.dynamic_ear_threshold:.3f} | "
                    f"Squint Threshold: {p.dynamic_squint_threshold:.3f}\n"
                    f"Last Calibrated: {p.calibrated_at or 'Uncalibrated Default'}"
                )
                self.lbl_profile_info.config(text=new_info)
                if self.on_profile_changed:
                    self.on_profile_changed()
                messagebox.showinfo("Profile Switched", f"Activated personalized profile for '{p.name}'.")

        btn_switch = tk.Button(
            switch_frame,
            text="Activate Profile",
            command=_do_switch,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_switch.pack(side="left", padx=6)

        # Create new profile frame
        create_frame = tk.Frame(parent, bg=COLOR_BG_CARD, bd=1, relief="solid", highlightbackground=COLOR_BORDER, padx=16, pady=12)
        create_frame.pack(fill="x", padx=12, pady=12)

        tk.Label(create_frame, text="CREATE NEW PERSONALIZED PROFILE", font=("Segoe UI", 9, "bold"), fg=COLOR_ACCENT_GREEN, bg=COLOR_BG_CARD).pack(anchor="w")

        form = tk.Frame(create_frame, bg=COLOR_BG_CARD)
        form.pack(fill="x", pady=6)

        tk.Label(form, text="Profile ID:", font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD).grid(row=0, column=0, sticky="w", pady=4)
        entry_pid = tk.Entry(form, width=20, bg="#0f172a", fg=COLOR_TEXT_MAIN, insertbackground=COLOR_TEXT_MAIN)
        entry_pid.grid(row=0, column=1, padx=8, pady=4, sticky="w")

        tk.Label(form, text="Display Name:", font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD).grid(row=1, column=0, sticky="w", pady=4)
        entry_pname = tk.Entry(form, width=20, bg="#0f172a", fg=COLOR_TEXT_MAIN, insertbackground=COLOR_TEXT_MAIN)
        entry_pname.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        def _do_create():
            pid = entry_pid.get().strip()
            pname = entry_pname.get().strip()
            if not pid or not pname:
                messagebox.showwarning("Input Error", "Please provide both Profile ID and Display Name.")
                return
            new_p = UserProfile(profile_id=pid, name=pname)
            self.daemon.create_or_update_user_profile(new_p)
            if self.on_profile_changed:
                self.on_profile_changed()
            messagebox.showinfo("Profile Created", f"Profile '{pname}' created and activated!\nClick Calibrate to personalize EAR thresholds.")
            self.destroy()

        btn_create = tk.Button(
            create_frame,
            text="Create & Activate Profile",
            command=_do_create,
            bg="#059669",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
        )
        btn_create.pack(anchor="w", pady=(6, 0))


class OculiSenseApp:
    """
    Main desktop window controller for OculiSenseAI.
    """

    def __init__(
        self,
        daemon: OculiSenseDaemon,
        root: Optional[tk.Tk] = None,
        start_in_overlay: bool = False,
    ) -> None:
        self.daemon = daemon
        self.root = root or tk.Tk()
        self.root.title("OculiSenseAI — Edge-AI Biometric Eye Strain & Workload Fatigue")
        self.root.geometry("980x700")
        self.root.minsize(880, 620)
        self.root.configure(bg=COLOR_BG_DARK)

        self.exporter = ReportExporter(self.daemon.db)
        self._overlay_window: Optional[BreakOverlayWindow] = None
        self._history_window: Optional[AnalyticsHistoryWindow] = None
        self._hud_overlay: Optional[OverlayHUDWindow] = None
        self._is_overlay_mode: bool = False
        self._last_snapshot: Optional[DaemonSnapshot] = None
        self._vis_photo_img: Optional[Any] = None

        self._build_ui()

        # Connect daemon listener
        self.daemon.register_listener(self._on_snapshot_received)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        if start_in_overlay:
            self.toggle_overlay_mode()

    def _build_ui(self) -> None:
        # Header banner
        header = tk.Frame(self.root, bg=COLOR_BG_CARD, height=60, bd=0)
        header.pack(fill="x", side="top", padx=16, pady=(12, 8))

        # Title & Subtitle
        title_box = tk.Frame(header, bg=COLOR_BG_CARD)
        title_box.pack(side="left", padx=16, pady=8)

        lbl_app_name = tk.Label(
            title_box,
            text="OculiSenseAI",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_ACCENT_CYAN,
            bg=COLOR_BG_CARD,
        )
        lbl_app_name.pack(anchor="w")

        lbl_app_tag = tk.Label(
            title_box,
            text="Edge-AI Biometric Protection for Digital Eye Strain & Workload Fatigue",
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
        )
        lbl_app_tag.pack(anchor="w")

        # Right Header Pills
        status_box = tk.Frame(header, bg=COLOR_BG_CARD)
        status_box.pack(side="right", padx=16, pady=8)

        self.btn_toggle_hud = tk.Button(
            status_box,
            text="🗖 Overlay Mode",
            command=self.toggle_overlay_mode,
            bg="#1e293b",
            fg=COLOR_ACCENT_CYAN,
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
        )
        self.btn_toggle_hud.pack(side="right", padx=6)

        self.lbl_overall_score = tk.Label(
            status_box,
            text="HEALTH: 100% [FRESH]",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg="#052e16",
            padx=10,
            pady=4,
        )
        self.lbl_overall_score.pack(side="right", padx=6)

        self.lbl_profile = tk.Label(
            status_box,
            text=f"👤 {self.daemon.config.profile.name}",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg="#1e293b",
            padx=10,
            pady=4,
        )
        self.lbl_profile.pack(side="right", padx=6)

        self.lbl_status_pill = tk.Label(
            status_box,
            text="● INITIALIZING",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg="#1e3a2f",
            padx=10,
            pady=4,
        )
        self.lbl_status_pill.pack(side="right", padx=6)

        # Main Body Container
        main_content = tk.Frame(self.root, bg=COLOR_BG_DARK)
        main_content.pack(fill="both", expand=True, padx=16, pady=4)

        # Top Section: Video / Face Canvas + Circular Gauges
        top_row = tk.Frame(main_content, bg=COLOR_BG_DARK)
        top_row.pack(fill="x", pady=6)

        # 1. Vision Landmark Canvas
        vis_box = tk.Frame(
            top_row,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground=COLOR_BORDER,
        )
        vis_box.pack(side="left", fill="both", expand=True, padx=(0, 8))

        vis_header = tk.Frame(vis_box, bg=COLOR_BG_CARD)
        vis_header.pack(fill="x", padx=10, pady=6)
        tk.Label(
            vis_header,
            text="BIOMETRIC OCULAR TRACKER",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
        ).pack(side="left")
        self.lbl_vis_mode = tk.Label(
            vis_header,
            text="[LIVE WEBCAM 0] • INITIALIZING",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg=COLOR_BG_CARD,
        )
        self.lbl_vis_mode.pack(side="right")

        self.vis_canvas = tk.Canvas(
            vis_box,
            width=360,
            height=160,
            bg="#020617",
            highlightthickness=0,
        )
        self.vis_canvas.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        # 2. Gauges Box
        gauges_box = tk.Frame(
            top_row,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground=COLOR_BORDER,
        )
        gauges_box.pack(side="right", padx=(8, 0))

        tk.Label(
            gauges_box,
            text="REAL-TIME BIOMETRIC METERS",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
        ).pack(anchor="w", padx=12, pady=6)

        gauges_inner = tk.Frame(gauges_box, bg=COLOR_BG_CARD)
        gauges_inner.pack(padx=10, pady=(0, 10))

        self.gauge_fatigue = CircularGauge(gauges_inner, title="Fatigue Score", size=135)
        self.gauge_fatigue.pack(side="left", padx=8)

        self.gauge_strain = CircularGauge(gauges_inner, title="Strain Index", size=135)
        self.gauge_strain.pack(side="left", padx=8)

        # Middle Section: 4 Stat Cards
        cards_row = tk.Frame(main_content, bg=COLOR_BG_DARK)
        cards_row.pack(fill="x", pady=6)

        self.card_bpm = self._create_stat_card(cards_row, "BLINK FREQUENCY", "-- bpm", "Normal: 15-20")
        self.card_bpm.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.card_focus = self._create_stat_card(cards_row, "SCREEN TIME", "-- min", "Continuous Focus")
        self.card_focus.pack(side="left", fill="x", expand=True, padx=4)

        self.card_perclos = self._create_stat_card(cards_row, "PERCLOS (CLOSURE)", "-- %", "Eye closure ratio")
        self.card_perclos.pack(side="left", fill="x", expand=True, padx=4)

        self.card_break = self._create_stat_card(cards_row, "NEXT 20-20-20 BREAK", "--:--", "Adherence: 100%")
        self.card_break.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Bottom Section: Live EAR Oscilloscope Graph
        graph_box = tk.Frame(
            main_content,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground=COLOR_BORDER,
        )
        graph_box.pack(fill="both", expand=True, pady=6)

        g_header = tk.Frame(graph_box, bg=COLOR_BG_CARD)
        g_header.pack(fill="x", padx=10, pady=4)
        tk.Label(
            g_header,
            text="LIVE EYE ASPECT RATIO (EAR) TELEMETRY WAVEFORM",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG_CARD,
        ).pack(side="left")

        self.lbl_ear_live = tk.Label(
            g_header,
            text="EAR: 0.00",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT_CYAN,
            bg=COLOR_BG_CARD,
        )
        self.lbl_ear_live.pack(side="right")

        self.ear_graph = EARGraph(graph_box, width=700, height=100)
        self.ear_graph.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Action / Control Footer Bar
        footer = tk.Frame(self.root, bg=COLOR_BG_CARD, height=54)
        footer.pack(fill="x", side="bottom", padx=16, pady=(4, 12))

        # Camera Start/Stop Toggle Button
        is_cam_on = self.daemon.is_camera_active
        self.btn_camera = tk.Button(
            footer,
            text="⏹ Stop Camera" if is_cam_on else "▶ Start Camera",
            command=self._on_toggle_camera_clicked,
            bg="#dc2626" if is_cam_on else "#059669",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
        )
        self.btn_camera.pack(side="left", padx=(12, 6), pady=10)

        # Camera Index Selector Dropdown
        lbl_cam = tk.Label(
            footer, text="Camera:", font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD
        )
        lbl_cam.pack(side="left", padx=(4, 2), pady=10)

        active_cam = getattr(self.daemon.tracker, "active_camera_id", 0)
        self.combo_camera = ttk.Combobox(
            footer,
            values=["Camera 0", "Camera 1", "Camera 2", "Camera 3", "Rescan Cameras..."],
            state="readonly",
            width=14,
            postcommand=self._refresh_camera_dropdown,
        )
        self.combo_camera.set(f"Camera {active_cam}")
        self.combo_camera.bind("<<ComboboxSelected>>", self._on_camera_selected)
        self.combo_camera.pack(side="left", padx=(2, 6), pady=10)

        # Scenario dropdown (Demo mode)
        lbl_scenario = tk.Label(
            footer, text="Behavior Scenario:", font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD
        )
        lbl_scenario.pack(side="left", padx=(6, 4), pady=10)

        self.combo_scenario = ttk.Combobox(
            footer,
            values=["CYCLIC", "NORMAL", "STRAINED", "FATIGUED", "AWAY"],
            state="readonly",
            width=11,
        )
        self.combo_scenario.set("CYCLIC")
        self.combo_scenario.bind("<<ComboboxSelected>>", self._on_scenario_changed)
        self.combo_scenario.pack(side="left", padx=4, pady=10)

        # Quick Calibration Button
        self.btn_calibrate = tk.Button(
            footer,
            text="⚡ Calibrate (5s)",
            command=self._on_calibrate_clicked,
            bg="#334155",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        self.btn_calibrate.pack(side="left", padx=6, pady=10)

        # Take Break Button
        btn_take_break = tk.Button(
            footer,
            text="👁️ 20-20-20 Rest",
            command=self._on_manual_break_clicked,
            bg="#065f46",
            fg="#a7f3d0",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        btn_take_break.pack(side="left", padx=6, pady=10)

        # Force Rest Lockdown Button
        btn_force_rest = tk.Button(
            footer,
            text="🛑 Test Forced Rest",
            command=self._force_rest_lockdown,
            bg="#7f1d1d",
            fg="#fecaca",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        btn_force_rest.pack(side="left", padx=6, pady=10)

        # Export Report Button
        btn_export = tk.Button(
            footer,
            text="📊 Export HTML Report",
            command=self._on_export_clicked,
            bg="#1e3a8a",
            fg="#bfdbfe",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        btn_export.pack(side="right", padx=12, pady=10)

        # Trends & Compliance History Button
        btn_history = tk.Button(
            footer,
            text="📈 Trends & History",
            command=self._on_history_clicked,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        btn_history.pack(side="right", padx=4, pady=10)

    def _create_stat_card(self, parent: tk.Widget, title: str, default_val: str, sub: str) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=COLOR_BG_CARD,
            bd=1,
            relief="solid",
            highlightbackground=COLOR_BORDER,
            padx=12,
            pady=10,
        )
        lbl_t = tk.Label(card, text=title, font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD)
        lbl_t.pack(anchor="w")

        lbl_v = tk.Label(card, text=default_val, font=("Segoe UI", 16, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_BG_CARD)
        lbl_v.pack(anchor="w", pady=(2, 2))
        card._val_label = lbl_v  # type: ignore

        lbl_s = tk.Label(card, text=sub, font=("Segoe UI", 7), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD)
        lbl_s.pack(anchor="w")
        card._sub_label = lbl_s  # type: ignore

        return card

    def _on_scenario_changed(self, event=None) -> None:
        scenario = self.combo_scenario.get()
        self.daemon.tracker.set_synthetic_mode(scenario)

    def _refresh_camera_dropdown(self) -> None:
        """Scan and populate combobox with detected physical webcam indices."""
        try:
            from oculisense.vision.tracker import detect_available_cameras
            found = detect_available_cameras(max_probe=6)
            if found:
                self.combo_camera["values"] = [f"Camera {i}" for i in found] + ["Rescan Cameras..."]
            else:
                self.combo_camera["values"] = ["Camera 0", "Camera 1", "Camera 2", "Camera 3", "Rescan Cameras..."]
        except Exception:
            pass

    def _on_camera_selected(self, event=None) -> None:
        val = self.combo_camera.get()
        if "Rescan" in val:
            from oculisense.vision.tracker import detect_available_cameras
            found = detect_available_cameras(max_probe=6)
            curr = self.daemon.tracker.active_camera_id
            if found:
                new_vals = [f"Camera {i}" for i in found] + ["Rescan Cameras..."]
                self.combo_camera["values"] = new_vals
                chosen = curr if curr in found else found[0]
                self.combo_camera.set(f"Camera {chosen}")
                self.daemon.switch_camera(chosen)
                messagebox.showinfo(
                    "Webcam Scan",
                    f"Detected {len(found)} accessible camera(s): {found}.\nActive: Camera {chosen}.",
                )
            else:
                self.combo_camera["values"] = ["Camera 0", "Camera 1", "Camera 2", "Camera 3", "Rescan Cameras..."]
                self.combo_camera.set(f"Camera {curr}")
                messagebox.showwarning(
                    "Webcam Scan",
                    "No physical webcams detected on indices 0-5.\nCheck device connection and permissions.",
                )
            return

        if val.startswith("Camera "):
            try:
                cam_idx = int(val.split()[1])
                success = self.daemon.switch_camera(cam_idx)
                if not success and self.daemon.is_camera_active:
                    messagebox.showwarning(
                        "Camera Switch",
                        f"Could not open Camera {cam_idx}.\nPlease verify that the webcam is connected and accessible.",
                    )
            except Exception as e:
                pass

    def _on_toggle_camera_clicked(self) -> None:
        new_state = self.daemon.toggle_camera()
        if new_state:
            self.btn_camera.config(text="⏹ Stop Camera", bg="#dc2626")
            self.btn_calibrate.config(state="normal")
        else:
            self.btn_camera.config(text="▶ Start Camera", bg="#059669")
            self.btn_calibrate.config(state="disabled")

    def _on_calibrate_clicked(self) -> None:
        if not self.daemon.is_camera_active:
            messagebox.showwarning(
                "Camera Disabled",
                "Cannot calibrate while the camera is off.\nPlease click 'Start Camera' first.",
            )
            return
        if self.daemon._last_snapshot and not self.daemon._last_snapshot.vision.face_detected:
            messagebox.showwarning(
                "No Face Detected",
                "Cannot calibrate while no face is detected.\nPlease center your face in front of the camera first.",
            )
            return
        started = self.daemon.start_calibration(duration_sec=5.0)
        if started:
            self.btn_calibrate.config(text="Calibrating...", state="disabled")

    def _on_manual_break_clicked(self) -> None:
        self.daemon.break_advisor.trigger_dynamic_break(reason="MANUAL_TRIGGER")

    def _on_export_clicked(self) -> None:
        summary = self.daemon.aggregator.get_session_summary(
            breaks_prompted=self.daemon.break_advisor._total_prompted,
            breaks_completed=self.daemon.break_advisor._total_completed,
            compliance_rate=self.daemon.break_advisor.compliance_rate,
        )
        report_path = os.path.abspath("oculisense_report.html")
        self.exporter.export_html_report(report_path, session_summary=summary)
        messagebox.showinfo(
            "Report Exported",
            f"OculiSenseAI Biometric Report successfully generated!\n\nSaved to:\n{report_path}",
        )

    def _on_history_clicked(self) -> None:
        if self._history_window is None or not self._history_window.winfo_exists():
            self._history_window = AnalyticsHistoryWindow(
                self.root,
                self.daemon,
                on_profile_changed=self._on_profile_changed,
            )
        else:
            self._history_window.lift()

    def _on_profile_changed(self) -> None:
        if hasattr(self, "lbl_profile"):
            self.lbl_profile.config(text=f"👤 {self.daemon.config.profile.name}")

    def _on_snapshot_received(self, s: DaemonSnapshot) -> None:
        """Invoked from daemon thread, dispatch to Tkinter main thread with coalescing."""
        self._last_snapshot = s
        if not getattr(self, "_ui_update_pending", False):
            self._ui_update_pending = True
            try:
                self.root.after_idle(self._flush_ui_update)
            except Exception:
                self._ui_update_pending = False

    def _flush_ui_update(self) -> None:
        self._ui_update_pending = False
        if self._last_snapshot is not None:
            self._update_ui_elements(self._last_snapshot)

    def _update_ui_elements(self, s: DaemonSnapshot) -> None:
        # Camera button & mode label synchronization
        if not s.is_camera_active:
            self.btn_camera.config(text="▶ Start Camera", bg="#059669")
            self.lbl_vis_mode.config(text="CAMERA OFF", fg=COLOR_TEXT_MUTED)
            self.lbl_status_pill.config(
                text="● CAMERA OFF",
                fg=COLOR_TEXT_MUTED,
                bg="#1e293b",
            )
            self.btn_calibrate.config(text="⚡ Calibrate (5s)", state="disabled")
            self.card_bpm._val_label.config(text="-- bpm")  # type: ignore
            self.card_bpm._sub_label.config(text="Camera Paused")  # type: ignore
            self.card_focus._val_label.config(text="-- min")  # type: ignore
            self.card_focus._sub_label.config(text="Strain Level: IDLE")  # type: ignore
            self.card_perclos._val_label.config(text="-- %")  # type: ignore
            self.card_perclos._sub_label.config(text="Camera Paused")  # type: ignore
            rem_m = int(s.breaks.remaining_until_break_sec // 60)
            rem_s = int(s.breaks.remaining_until_break_sec % 60)
            self.card_break._val_label.config(text=f"{rem_m:02d}:{rem_s:02d}")  # type: ignore
            self.card_break._sub_label.config(text="Camera Paused")  # type: ignore
            self.gauge_fatigue.set_value(0.0)
            self.gauge_strain.set_value(0.0)
            self.ear_graph.push(0.0)
            self.lbl_ear_live.config(text="EAR: -- (Camera Off)")
            self._render_visualizer(s)
            return

        self.btn_camera.config(text="⏹ Stop Camera", bg="#dc2626")
        src_tag = s.vision.display_indicator
        tag_color = COLOR_ACCENT_GREEN if not s.vision.is_synthetic else COLOR_ACCENT_AMBER
        self.lbl_vis_mode.config(text=f"{src_tag} • {s.vision.source_status}", fg=tag_color)

        # 1. Status Pill
        if s.is_calibrating:
            pct = int(s.calibration_progress * 100)
            self.lbl_status_pill.config(
                text=f"● {src_tag} CALIBRATING {pct}%",
                fg=COLOR_ACCENT_AMBER,
                bg="#3d2806",
            )
            self.btn_calibrate.config(text="Calibrating...", state="disabled")
        elif not s.presence.is_present:
            self.lbl_status_pill.config(
                text=f"● {src_tag} AWAY FROM DESK",
                fg=COLOR_ACCENT_RED,
                bg="#3f1212",
            )
            self.btn_calibrate.config(text="⚡ Calibrate (5s)", state="normal")
        elif s.squint.is_squinting:
            self.lbl_status_pill.config(
                text=f"● {src_tag} SQUINT / STRAIN DETECTED",
                fg=COLOR_ACCENT_AMBER,
                bg="#3d2806",
            )
            self.btn_calibrate.config(text="⚡ Calibrate (5s)", state="normal")
        elif s.fatigue.is_warning:
            self.lbl_status_pill.config(
                text=f"● {src_tag} FATIGUE ELEVATED",
                fg=COLOR_ACCENT_RED,
                bg="#3f1212",
            )
            self.btn_calibrate.config(text="⚡ Calibrate (5s)", state="normal")
        else:
            self.lbl_status_pill.config(
                text=f"● {src_tag} TRACKING (HEALTHY)",
                fg=COLOR_ACCENT_GREEN,
                bg="#1e3a2f",
            )
            self.btn_calibrate.config(text="⚡ Calibrate (5s)", state="normal")

        # Update Overall Health Badge
        score = s.overall_score
        if score >= 80.0:
            self.lbl_overall_score.config(text=f"HEALTH: {score:.0f}% [FRESH]", fg=COLOR_ACCENT_GREEN, bg="#052e16")
        elif score >= 60.0:
            self.lbl_overall_score.config(text=f"HEALTH: {score:.0f}% [GOOD]", fg=COLOR_ACCENT_CYAN, bg="#082f49")
        elif score >= 40.0:
            self.lbl_overall_score.config(text=f"HEALTH: {score:.0f}% [STRAIN]", fg=COLOR_ACCENT_AMBER, bg="#451a03")
        else:
            self.lbl_overall_score.config(text=f"HEALTH: {score:.0f}% [CRITICAL]", fg=COLOR_ACCENT_RED, bg="#450a0a")

        # Forward telemetry to floating overlay HUD if active
        if self._hud_overlay is not None and self._hud_overlay.winfo_exists():
            self._hud_overlay.update_telemetry(s)

        # 2. Update Gauges
        self.gauge_fatigue.set_value(s.fatigue.fatigue_score)
        self.gauge_strain.set_value(s.strain.strain_score)

        # 3. Update Stat Cards
        bpm_val = s.blink.blink_rate_bpm
        if s.blink.total_blinks == 0 and s.is_camera_active:
            self.card_bpm._val_label.config(text="-- bpm")  # type: ignore
            self.card_bpm._sub_label.config(text="Blink to calculate BPM")  # type: ignore
        else:
            self.card_bpm._val_label.config(text=f"{bpm_val:.1f} bpm")  # type: ignore
            self.card_bpm._sub_label.config(text=f"{s.blink.total_blinks} total blinks")  # type: ignore

        focus_mins = s.strain.continuous_focus_sec / 60.0
        self.card_focus._val_label.config(text=f"{focus_mins:.1f} min")  # type: ignore
        self.card_focus._sub_label.config(text=f"Strain Level: {s.strain.strain_level}")  # type: ignore

        self.card_perclos._val_label.config(text=f"{s.blink.perclos * 100:.1f} %")  # type: ignore
        self.card_perclos._sub_label.config(  # type: ignore
            text="Prolonged: YES" if s.blink.is_prolonged_closure else "Normal Blinks"
        )

        rem_m = int(s.breaks.remaining_until_break_sec // 60)
        rem_s = int(s.breaks.remaining_until_break_sec % 60)
        self.card_break._val_label.config(text=f"{rem_m:02d}:{rem_s:02d}")  # type: ignore
        self.card_break._sub_label.config(  # type: ignore
            text=f"Adherence: {s.breaks.compliance_rate_pct:.0f}%"
        )

        # 4. EAR Waveform Graph
        self.ear_graph.set_thresholds(
            self.daemon.blink_detector.closed_threshold,
            self.daemon.squint_detector.squint_upper_threshold,
        )
        self.ear_graph.push(s.vision.ear_avg)
        self.lbl_ear_live.config(
            text=(
                f"EAR: {s.vision.ear_avg:.3f} | Min (20s): {s.breaks.least_recorded_ear:.3f}"
                + (f" [{min(20.0, s.breaks.eval_elapsed_sec):.0f}/20s]" if s.breaks.is_evaluating else "")
            )
        )

        # 5. Draw Visualizer Face / Eyes Canvas
        self._render_visualizer(s)

        # 6. Break Overlay management & Forced Rest Enforcement
        # Evaluated strictly according to the break advisor's 20-second continuous evaluation window
        if s.breaks.is_break_recommended:
            is_forced = s.is_force_stop_needed or s.breaks.reason in (
                "CRITICAL_LOW_SCORE",
                "CRITICAL_LOW_EAR_SCORE",
                "FATIGUE_THRESHOLD",
                "STRAIN_THRESHOLD",
                "MANUAL_FORCE_REST",
            )
            if self._overlay_window is None or not self._overlay_window.winfo_exists():
                self._overlay_window = BreakOverlayWindow(
                    self.root,
                    on_snooze=lambda: self.daemon.break_advisor.snooze_break(),
                    on_skip=lambda: self.daemon.break_advisor.skip_break(),
                    on_complete=lambda: self.daemon.break_advisor.complete_break(),
                    is_forced=is_forced,
                    overall_score=s.overall_score,
                    strain_score=s.strain.strain_score,
                    fatigue_score=s.fatigue.fatigue_score,
                    bpm=s.blink.blink_rate_bpm,
                    ear=s.vision.ear_avg,
                    least_ear=s.breaks.least_recorded_ear,
                    on_lock_workstation=lambda: self.daemon.presence_manager.trigger_immediate_lock(),
                )
            if self._overlay_window is not None and self._overlay_window.winfo_exists():
                self._overlay_window.update_countdown(
                    s.breaks.remaining_break_duration_sec,
                    is_in_rest=s.breaks.is_in_break,
                    overall_score=s.overall_score,
                    bpm=s.blink.blink_rate_bpm,
                    ear=s.vision.ear_avg,
                    least_ear=s.breaks.least_recorded_ear,
                )
        else:
            if self._overlay_window is not None and self._overlay_window.winfo_exists():
                try:
                    self._overlay_window.destroy()
                except Exception:
                    pass
            self._overlay_window = None

    def toggle_overlay_mode(self) -> None:
        """Switch between full desktop dashboard and floating overlay HUD."""
        self._is_overlay_mode = not self._is_overlay_mode
        if self._is_overlay_mode:
            self.root.withdraw()
            if self._hud_overlay is None or not self._hud_overlay.winfo_exists():
                self._hud_overlay = OverlayHUDWindow(
                    self.root,
                    on_expand_dashboard=self.toggle_overlay_mode,
                    on_toggle_camera=self._on_toggle_camera_clicked,
                    on_force_rest=self._force_rest_lockdown,
                )
            if self._last_snapshot is not None and self._hud_overlay is not None:
                self._hud_overlay.update_telemetry(self._last_snapshot)
        else:
            if self._hud_overlay is not None and self._hud_overlay.winfo_exists():
                try:
                    self._hud_overlay.destroy()
                except Exception:
                    pass
            self._hud_overlay = None
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

    def _force_rest_lockdown(self) -> None:
        """Manually trigger the forced rest lockdown overlay."""
        self.daemon.break_advisor.trigger_dynamic_break(reason="MANUAL_FORCE_REST")

    def _render_visualizer(self, s: DaemonSnapshot) -> None:
        """Render the actual live webcam video feed and biometric tracking HUD."""
        c = self.vis_canvas
        c.delete("all")
        w = c.winfo_width() if c.winfo_width() > 10 else int(c.cget("width") or 360)
        h = c.winfo_height() if c.winfo_height() > 10 else int(c.cget("height") or 160)
        cx, cy = w / 2.0, h / 2.0

        if not s.is_camera_active or s.vision.source_status == "CAMERA_OFF":
            c.create_text(
                cx, cy - 14,
                text="📷 CAMERA IS OFF",
                fill=COLOR_TEXT_MAIN,
                font=("Segoe UI", 12, "bold"),
            )
            c.create_text(
                cx, cy + 12,
                text="Click 'Start Camera' below to begin ocular tracking",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 9),
            )
            c.create_text(
                cx, h - 14,
                text="Tracking Paused | No Biometrics Computed | Low CPU Mode",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 8),
            )
            return

        # 1. SHOWCASE ACTUAL LIVE WEBCAM VIDEO FEED (PRIORITY)
        if s.vision.annotated_frame is not None and cv2 is not None and Image is not None and ImageTk is not None:
            try:
                frame = s.vision.annotated_frame
                fh, fw = frame.shape[:2]

                # Scale to fit canvas while preserving aspect ratio
                scale = min(w / float(fw), h / float(fh))
                target_w = max(10, int(fw * scale))
                target_h = max(10, int(fh * scale))

                # Fast OpenCV interpolation then RGB convert
                resized_bgr = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                photo_img = ImageTk.PhotoImage(pil_img, master=self.root)
                self._vis_photo_img = photo_img
                c._image_ref = photo_img
                c.create_image(cx, cy, image=photo_img)

                # Real-Time HUD Overlay on top of actual live webcam feed
                src_tag = s.vision.display_indicator
                status_color = COLOR_ACCENT_GREEN if s.vision.face_detected else COLOR_ACCENT_AMBER
                status_text = "● LIVE FEED: TRACKING" if s.vision.face_detected else "● LIVE FEED: SEARCHING FACE"

                # Top-left badge
                badge_x1 = max(4.0, cx - target_w / 2.0 + 8.0)
                badge_y1 = max(4.0, cy - target_h / 2.0 + 8.0)
                c.create_rectangle(
                    badge_x1, badge_y1,
                    badge_x1 + 175.0, badge_y1 + 20.0,
                    fill="#020617", outline=status_color, width=1,
                )
                c.create_text(
                    badge_x1 + 87.0, badge_y1 + 10.0,
                    text=f"{src_tag} {status_text}",
                    fill=status_color,
                    font=("Segoe UI", 8, "bold"),
                )

                # Bottom live telemetry HUD banner
                banner_y2 = cy + target_h / 2.0
                banner_y1 = banner_y2 - 22.0
                c.create_rectangle(
                    cx - target_w / 2.0, banner_y1,
                    cx + target_w / 2.0, banner_y2,
                    fill="#020617", outline="",
                )
                blink_tag = " [👁 BLINK!]" if s.blink.blink_detected_this_frame else ""
                hud_color = "#4ade80" if s.blink.blink_detected_this_frame else "#38bdf8"
                c.create_text(
                    cx, banner_y1 + 11.0,
                    text=(
                        f"EAR: {s.vision.ear_avg:.3f} | "
                        f"Blinks: {s.blink.total_blinks}{blink_tag} | "
                        f"BPM: {s.blink.blink_rate_bpm:.1f} | "
                        f"Strain: {s.strain.strain_score:.0f} | "
                        f"Fatigue: {s.fatigue.fatigue_score:.0f} | "
                        f"Health: {s.overall_score:.0f}%"
                    ),
                    fill=hud_color,
                    font=("Segoe UI", 8, "bold"),
                )
                return
            except Exception as e:
                pass

        # Fallback ONLY when no video feed is delivered
        if s.vision.source_status in ("NO_CAMERA", "SEARCHING_CAMERA"):
            c.create_text(
                cx, cy - 14,
                text=f"📷 {s.vision.display_indicator} SEARCHING FOR WEBCAM...",
                fill=COLOR_ACCENT_AMBER,
                font=("Segoe UI", 11, "bold"),
            )
            c.create_text(
                cx, cy + 12,
                text="Probing hardware... Connect webcam or select another Camera index below.",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 8),
            )
            return

        if not s.vision.face_detected:
            c.create_text(
                cx, cy - 10,
                text="⚠️ NO FACE DETECTED IN CAMERA FEED",
                fill=COLOR_ACCENT_AMBER,
                font=("Segoe UI", 11, "bold"),
            )
            c.create_text(
                cx, cy + 14,
                text="Please look directly at your webcam to track EAR, blinks and fatigue.",
                fill=COLOR_TEXT_MUTED,
                font=("Segoe UI", 9),
            )
            return

        # If camera is tracking but frame object was None, display telemetry box (no cartoon oval)
        c.create_rectangle(cx - 160, cy - 40, cx + 160, cy + 40, fill="#0f172a", outline=COLOR_ACCENT_CYAN, width=1)
        c.create_text(
            cx, cy - 12,
            text=f"👁️ BIOMETRIC TRACKING ACTIVE: EAR {s.vision.ear_avg:.3f}",
            fill=COLOR_ACCENT_CYAN,
            font=("Segoe UI", 11, "bold"),
        )
        c.create_text(
            cx, cy + 14,
            text=f"Blinks: {s.blink.total_blinks}  |  Rate: {s.blink.blink_rate_bpm:.1f} BPM  |  Health: {s.overall_score:.0f}%",
            fill=COLOR_TEXT_MAIN,
            font=("Segoe UI", 9, "bold"),
        )

    def _on_closing(self) -> None:
        if self._overlay_window is not None and self._overlay_window.winfo_exists():
            try:
                self._overlay_window.destroy()
            except Exception:
                pass
            self._overlay_window = None
        self.daemon.stop()
        self.root.destroy()

    def run(self) -> None:
        """Start daemon loop and enter Tkinter main loop."""
        self.daemon.start()
        self.root.mainloop()
