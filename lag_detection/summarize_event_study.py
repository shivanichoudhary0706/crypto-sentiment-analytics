"""One-off helper: compact table across every reports/event_study/*_summary.json.
Run from project root: python summarize_event_study.py
"""
import json
import glob

for f in sorted(glob.glob("reports/event_study/*_summary.json")):
    s = json.load(open(f, encoding="utf-8"))
    name = f.split("/")[-1].replace("\\", "/").split("/")[-1][:-13]
    pos, neg = s.get("positive"), s.get("negative")
    pos_s = f"n={pos['n_events']} onset={pos['onset_hour']} med_exit={pos['median_optimal_exit_hour']}" if pos else "n/a"
    neg_s = f"n={neg['n_events']} onset={neg['onset_hour']} med_exit={neg['median_optimal_exit_hour']}" if neg else "n/a"
    print(f"{s['cause_coin']:<9} -> {s['price_target']:<9} ({s['relationship']:<9}) | +: {pos_s:<40} | -: {neg_s}")