"""
Report Exporter for OculiSenseAI.
Exports biometric telemetry and session analytics to JSON, CSV, and HTML reports.
"""

from __future__ import annotations
import csv
import datetime
import json
from typing import Any, Dict, List, Optional

from oculisense.analytics.aggregator import SessionSummary
from oculisense.storage.database import DatabaseManager


class ReportExporter:
    """Generates user-facing reports and data archives."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def export_csv(self, filepath: str, days: int = 7) -> None:
        """Export daily telemetry rollup to CSV."""
        rows = self.db.get_daily_metrics(days=days)
        if not rows:
            rows = [{
                "log_date": datetime.date.today().isoformat(),
                "active_minutes": 0,
                "mean_fatigue": 0.0,
                "max_fatigue": 0.0,
                "mean_strain": 0.0,
                "max_strain": 0.0,
                "total_blinks": 0,
                "total_squint_mins": 0.0,
            }]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def export_json(self, filepath: str, session_summary: Optional[SessionSummary] = None) -> None:
        """Export session and recent metrics to JSON."""
        data: Dict[str, Any] = {
            "exported_at": datetime.datetime.now().isoformat(),
            "daily_trends": self.db.get_daily_metrics(days=14),
            "recent_sessions": self.db.get_recent_sessions(limit=5),
            "break_compliance_events": self.db.get_break_events(limit=50),
            "user_profiles": [
                {
                    "profile_id": p.profile_id,
                    "name": p.name,
                    "baseline_open_ear": p.baseline_open_ear,
                    "baseline_closed_ear": p.baseline_closed_ear,
                    "calibrated_at": p.calibrated_at,
                }
                for p in self.db.list_user_profiles()
            ],
        }
        if session_summary:
            data["current_session"] = {
                "duration_minutes": session_summary.duration_minutes,
                "total_blinks": session_summary.total_blinks,
                "avg_blink_rate_bpm": session_summary.avg_blink_rate_bpm,
                "avg_fatigue": session_summary.avg_fatigue_score,
                "peak_fatigue": session_summary.peak_fatigue_score,
                "avg_strain": session_summary.avg_strain_score,
                "peak_strain": session_summary.peak_strain_score,
                "compliance_rate_pct": session_summary.compliance_rate_pct,
                "screen_time_minutes": session_summary.screen_time_minutes,
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_html_report(
        self,
        filepath: str,
        session_summary: Optional[SessionSummary] = None,
    ) -> None:
        """
        Generates a standalone, beautiful HTML analytics report with SVG charts and insights.
        """
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
        daily = self.db.get_daily_metrics(days=7)

        # Fallback values if no session summary passed
        dur = session_summary.duration_minutes if session_summary else 0.0
        blinks = session_summary.total_blinks if session_summary else 0
        bpm = session_summary.avg_blink_rate_bpm if session_summary else 0.0
        fatigue = session_summary.avg_fatigue_score if session_summary else 0.0
        strain = session_summary.avg_strain_score if session_summary else 0.0
        compliance = session_summary.compliance_rate_pct if session_summary else 100.0
        screen_time = session_summary.screen_time_minutes if session_summary else 0.0

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OculiSenseAI — Executive Biometric Health Report</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-card: #161b22;
            --border: #30363d;
            --text-primary: #c9d1d9;
            --text-heading: #58a6ff;
            --accent-green: #2ea043;
            --accent-amber: #d29922;
            --accent-red: #f85149;
            --accent-cyan: #38bdf8;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: var(--text-heading);
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 0;
            color: #8b949e;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .card .title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
            margin-bottom: 8px;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #f0f6fc;
        }}
        .card .sub {{
            font-size: 13px;
            color: #8b949e;
            margin-top: 5px;
        }}
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #f0f6fc;
            margin-top: 0;
            font-size: 18px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: #8b949e;
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-good {{ background: rgba(46,160,67,0.2); color: #3fb950; }}
        .badge-warn {{ background: rgba(210,153,34,0.2); color: #e3b341; }}
        .badge-crit {{ background: rgba(248,81,73,0.2); color: #ff7b72; }}
        .privacy-box {{
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
            font-size: 13px;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>OculiSenseAI — Biometric Ergonomics Report</h1>
        <p>Generated on {now_str} • Privacy-First On-Device Edge Analytics</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="title">Screen Time</div>
            <div class="value">{screen_time}m</div>
            <div class="sub">Total Session: {dur}m</div>
        </div>
        <div class="card">
            <div class="title">Blink Frequency</div>
            <div class="value">{bpm} <span style="font-size:16px;">bpm</span></div>
            <div class="sub">{blinks} total blinks</div>
        </div>
        <div class="card">
            <div class="title">Avg Fatigue Score</div>
            <div class="value" style="color: {'#3fb950' if fatigue < 40 else ('#e3b341' if fatigue < 70 else '#ff7b72')};">{fatigue}</div>
            <div class="sub">Out of 100</div>
        </div>
        <div class="card">
            <div class="title">Break Compliance</div>
            <div class="value" style="color: #38bdf8;">{compliance}%</div>
            <div class="sub">20-20-20 Adherence</div>
        </div>
    </div>

    <div class="section">
        <h2>Clinical Insights & Recommendations</h2>
        <ul>
            <li><strong>Blink Rate Dynamics:</strong> Your recorded blink rate was <strong>{bpm} BPM</strong>. The physiological baseline during conversation is 15-20 BPM. When concentrating on screens, blink rates often plunge below 10 BPM, accelerating tear evaporation and inducing digital asthenopia.</li>
            <li><strong>Digital Eye Strain:</strong> Session mean strain registered at <strong>{strain}/100</strong>. Keep screen brightness aligned with ambient lux and maintain a viewing distance of 20–28 inches (50–70 cm).</li>
            <li><strong>20-20-20 Adherence:</strong> Your rest compliance was <strong>{compliance}%</strong>. Taking regular 20-second breaks to view distant objects relaxes the ciliary muscle in the eye and drastically reduces optical fatigue.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Recent Daily Trends</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Active Minutes</th>
                    <th>Mean Fatigue</th>
                    <th>Max Strain</th>
                    <th>Total Blinks</th>
                    <th>Squint Time</th>
                </tr>
            </thead>
            <tbody>
"""
        if daily:
            for d in daily:
                html_content += f"""
                <tr>
                    <td>{d.get('log_date', '-')}</td>
                    <td>{d.get('active_minutes', 0)} mins</td>
                    <td><span class="badge {'badge-good' if d.get('mean_fatigue', 0) < 45 else 'badge-warn'}">{d.get('mean_fatigue', 0)}</span></td>
                    <td>{d.get('max_strain', 0)}</td>
                    <td>{d.get('total_blinks', 0)}</td>
                    <td>{d.get('total_squint_mins', 0)} mins</td>
                </tr>
"""
        else:
            html_content += """
                <tr>
                    <td colspan="6" style="text-align:center; color:#8b949e;">No historical daily sessions logged yet. Active data will populate here.</td>
                </tr>
"""

        break_events = self.db.get_break_events(limit=15)
        html_content += """
    <div class="section">
        <h2>Break Compliance & Rest History</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Trigger Reason</th>
                    <th>Outcome</th>
                    <th>Rest Duration</th>
                </tr>
            </thead>
            <tbody>
"""
        if break_events:
            for ev in break_events:
                ts_str = datetime.datetime.fromtimestamp(ev.get("timestamp", 0)).strftime("%H:%M:%S")
                outcome = ev.get("outcome", "UNKNOWN")
                badge_class = "badge-good" if outcome == "COMPLETED" else ("badge-warn" if outcome == "SNOOZED" else "badge-crit")
                html_content += f"""
                <tr>
                    <td>{ts_str}</td>
                    <td>{ev.get('reason', '-')}</td>
                    <td><span class="badge {badge_class}">{outcome}</span></td>
                    <td>{ev.get('duration_sec', 0):.0f}s</td>
                </tr>
"""
        else:
            html_content += """
                <tr>
                    <td colspan="4" style="text-align:center; color:#8b949e;">No break events logged in this session yet. Take regular breaks to populate compliance tracking.</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="privacy-box">
        <strong>🔒 OculiSenseAI Privacy Architecture:</strong>
        All biometric metrics are processed exclusively in RAM at the edge. No facial images, ocular video recordings, or identity vectors are stored on disk or transmitted over network sockets. Only aggregated scalar telemetry is retained in your local SQLite database.
    </div>
</div>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
