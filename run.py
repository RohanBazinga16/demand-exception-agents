"""
Command-line runner.

  python run.py --mode rules                  # no API key needed (deterministic baseline)
  python run.py --mode agents                 # 4-agent crew on Gemini (needs GEMINI_API_KEY)
  python run.py --mode agents --approve ask   # approve / reject each recommendation in the terminal
  python run.py --report runs/<file>.json     # re-print the evaluation for a saved run

Every run is saved to runs/ as JSON (recommendations, tool calls, trace, approvals) — the audit trail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from evaluation import full_report, print_report  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def ask_approvals(run):
    approvals = {}
    recs = (run.get("recommendations") or {}).get("recommendations", [])
    name = input("Approver name: ").strip() or "planner"
    for r in recs:
        print(f"\n{r['iwl_id']} | {r['final_bucket']} | {r['cause']} | {r['action']} {r['params_json']} "
              f"| horizon {r['horizon_change_pct']:+.1f}%\n  {r['rationale']}  evidence={r['evidence_note_ids']}")
        d = input("  [a]pprove / [r]eject / [e]dit params? ").strip().lower()
        entry = dict(decision="approve" if d.startswith(("a", "e")) else "reject", approver=name,
                     ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        if d.startswith("e"):
            entry["params_json"] = input("  new params JSON: ").strip()
        approvals[r["iwl_id"]] = entry
    return approvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rules", "agents"], default="rules")
    ap.add_argument("--model", default=None, help="Gemini model, e.g. gemini-3.8-flash")
    ap.add_argument("--fallback", default=None,
                    help="comma-separated backup models used if the main one stays overloaded (503), "
                         "e.g. gemini-3-flash-preview,gemini-2.5-flash-lite")
    ap.add_argument("--max-rpm", type=int, default=8)
    ap.add_argument("--approve", choices=["all", "ask", "none"], default="all")
    ap.add_argument("--report", default=None, help="path to a saved run JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.report:
        run = json.load(open(a.report))
    elif a.mode == "rules":
        from rules import run_rules
        run = run_rules()
    else:
        from crew import run_agents
        fb = [m for m in a.fallback.split(",") if m.strip()] if a.fallback else None
        run = run_agents(model=a.model, max_rpm=a.max_rpm, verbose=not a.quiet, fallbacks=fb)
        if run.get("retry_events"):
            print(f"\nNote: {len(run['retry_events'])} transient API errors were retried; see retry_events in the run file.")

    if not a.report:
        if a.approve == "ask":
            run["approvals"] = ask_approvals(run)
        elif a.approve == "all":
            run["approvals"] = {r["iwl_id"]: dict(decision="approve", approver="auto (CLI --approve all)",
                                                  ts=time.strftime("%Y-%m-%d %H:%M:%S"))
                                for r in (run.get("recommendations") or {}).get("recommendations", [])}
        os.makedirs(RUNS, exist_ok=True)
        path = os.path.join(RUNS, f"{run['mode']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        json.dump(run, open(path, "w"), indent=1, default=str)
        print(f"\nSaved run -> {path}  ({run['seconds']}s)")

    if run.get("recommendations") is None:
        print("WARNING: the planning agent's output could not be parsed into the schema. Raw output:\n",
              run.get("recommendations_raw", "")[:3000])
    print_report(full_report(run, run.get("approvals")))


if __name__ == "__main__":
    main()
