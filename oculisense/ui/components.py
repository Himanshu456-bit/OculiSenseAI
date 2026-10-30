"""
Custom Canvas Widgets and UI Components for OculiSenseAI.
Provides high-FPS custom canvas graphs, circular meters, and telemetry cards.
"""

from __future__ import annotations
import collections
import math
import tkinter as tk
from typing import Deque, Optional, Tuple

# Modern Dark Theme Palette
COLOR_BG_DARK = "#0F172A"       # Slate 900
COLOR_BG_CARD = "#1E293B"       # Slate 800
COLOR_BG_CARD_HOVER = "#334155" # Slate 700
COLOR_BORDER = "#334155"        # Slate 700
COLOR_TEXT_MAIN = "#F8FAFC"     # Slate 50
COLOR_TEXT_MUTED = "#94A3B8"    # Slate 400
COLOR_ACCENT_CYAN = "#38BDF8"   # Sky 400
COLOR_ACCENT_GREEN = "#10B981"  # Emerald 500
COLOR_ACCENT_AMBER = "#F59E0B"  # Amber 500
COLOR_ACCENT_RED = "#EF4444"    # Red 500


class EARGraph(tk.Canvas):
    """
    Oscilloscope-style scrolling telemetry graph for Eye Aspect Ratio.
    Draws live baseline, threshold markers, and blink dips.
    """

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 400,
        height: int = 120,
        max_points: int = 150,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=COLOR_BG_CARD,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            **kwargs,
        )
        self.w = width
        self.h = height
        self.max_points = max_points
        self.data: Deque[float] = collections.deque(maxlen=max_points)
        self.threshold: float = 0.21
        self.squint_threshold: float = 0.26

    def set_thresholds(self, closed_th: float, squint_th: float) -> None:
        self.threshold = closed_th
        self.squint_threshold = squint_th

    def push(self, ear: float) -> None:
        self.data.append(max(0.0, min(0.5, ear)))
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")

        # Background grid lines
        for y_norm in [0.1, 0.2, 0.3, 0.4]:
            y_pos = self.h - int(y_norm * (self.h / 0.45))
            self.create_line(0, y_pos, self.w, y_pos, fill="#273549", dash=(2, 4))

        # Closed Threshold line (Red)
        y_th = self.h - int(self.threshold * (self.h / 0.45))
        self.create_line(0, y_th, self.w, y_th, fill="#EF4444", dash=(4, 4), width=1)
        self.create_text(
            self.w - 5,
            y_th - 8,
            text=f"Close: {self.threshold:.2f}",
            fill="#EF4444",
            anchor="e",
            font=("Segoe UI", 7, "bold"),
        )

        # Squint Threshold line (Amber)
        y_sq = self.h - int(self.squint_threshold * (self.h / 0.45))
        self.create_line(0, y_sq, self.w, y_sq, fill="#F59E0B", dash=(2, 4), width=1)

        if len(self.data) < 2:
            return

        # Plot EAR trace
        step = self.w / float(self.max_points - 1)
        coords = []
        for i, val in enumerate(self.data):
            x = i * step
            y = self.h - int(val * (self.h / 0.45))
            y = max(4, min(self.h - 4, y))
            coords.extend([x, y])

        self.create_line(coords, fill=COLOR_ACCENT_CYAN, width=2, smooth=True)

        # Live current point
        if coords:
            curr_x, curr_y = coords[-2], coords[-1]
            self.create_oval(
                curr_x - 3, curr_y - 3, curr_x + 3, curr_y + 3,
                fill=COLOR_TEXT_MAIN, outline=COLOR_ACCENT_CYAN
            )


class CircularGauge(tk.Canvas):
    """
    Circular percentage meter for Fatigue and Strain indices.
    Color changes dynamically: Green -> Amber -> Red.
    """

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Fatigue",
        size: int = 130,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=COLOR_BG_CARD,
            highlightthickness=0,
            **kwargs,
        )
        self.size = size
        self.title = title
        self.value: float = 0.0
        self.redraw()

    def set_value(self, val: float) -> None:
        self.value = max(0.0, min(100.0, val))
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        cx = self.size / 2.0
        cy = self.size / 2.0 + 4
        radius = (self.size / 2.0) - 16

        # Background track arc (240 degree sweep)
        start_angle = -210
        extent_angle = -240
        self.create_arc(
            cx - radius, cy - radius, cx + radius, cy + radius,
            start=start_angle, extent=extent_angle,
            style="arc", outline="#334155", width=8
        )

        # Value color
        if self.value < 45.0:
            color = COLOR_ACCENT_GREEN
        elif self.value < 75.0:
            color = COLOR_ACCENT_AMBER
        else:
            color = COLOR_ACCENT_RED

        # Active arc
        active_extent = extent_angle * (self.value / 100.0)
        if abs(active_extent) > 0.1:
            self.create_arc(
                cx - radius, cy - radius, cx + radius, cy + radius,
                start=start_angle, extent=active_extent,
                style="arc", outline=color, width=8
            )

        # Numerical readout
        self.create_text(
            cx, cy - 4,
            text=f"{int(round(self.value))}",
            fill=COLOR_TEXT_MAIN,
            font=("Segoe UI", 16, "bold"),
        )
        self.create_text(
            cx, cy + 15,
            text=self.title.upper(),
            fill=COLOR_TEXT_MUTED,
            font=("Segoe UI", 7, "bold"),
        )
