import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

# Setup path
DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"

def build_ascii_bar(val: int, max_val: int, width: int = 30) -> str:
    """Builds a visual ASCII bar chart representation."""
    if max_val == 0:
        return ""
    fill_len = int(round(val * width / max_val))
    bar = "#" * fill_len + "-" * (width - fill_len)
    return bar

def analyze_logs(log_path: Path, alert_threshold: float = 0.70):
    """Analyzes the prediction log file and prints performance metrics."""
    if not log_path.exists():
        print(f"[-] No prediction log file found at: {log_path.resolve()}")
        print("[*] Perform some RAG queries first to populate the prediction log.")
        return

    queries: List[Dict] = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
    except Exception as e:
        print(f"[-] Error reading prediction log file: {e}")
        return

    total_queries = len(queries)
    if total_queries == 0:
        print("[!] The prediction log file is empty.")
        return

    # 1. Aggregates
    latencies = [q["latency_sec"] for q in queries]
    confidences = [q["confidence"] for q in queries]
    
    avg_latency = sum(latencies) / total_queries
    avg_confidence = sum(confidences) / total_queries
    
    # 2. Alerts on low confidence
    low_confidence_queries = [q for q in queries if q["confidence"] < alert_threshold]
    alert_count = len(low_confidence_queries)

    # 3. Defect & Severity Distributions
    defect_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    
    for q in queries:
        defect = q.get("predicted_defect", "unknown")
        severity = q.get("predicted_severity", "unknown")
        
        defect_counts[defect] = defect_counts.get(defect, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    # Print Report
    print("=" * 60)
    print(" [MONITOR] INDUSTRIAL DEFECT DETECTION SYSTEM MONITORING REPORT")
    print("=" * 60)
    print(f" Total Audited Queries      : {total_queries}")
    print(f" Average Inference Latency : {avg_latency:.4f} seconds")
    print(f" Average VLM Confidence    : {avg_confidence:.2%} (Target: >85%)")
    print("-" * 60)
    
    # Alert notifications
    if alert_count > 0:
        print(f" [ALERT] Low Confidence Alerts (< {alert_threshold:.0%}) : {alert_count} / {total_queries} queries!")
        for idx, alert in enumerate(low_confidence_queries[-3:]): # show last 3 alerts
            print(f"   - Alert {idx+1}: '{alert['question'][:40]}...' -> Predicted: {alert['predicted_defect']} (Conf: {alert['confidence']:.2f})")
    else:
        print(" [PASS] Alerts: All queries exceeded confidence thresholds.")
    print("-" * 60)

    # Defect class histogram
    print(" Category Distribution:")
    max_defect = max(defect_counts.values()) if defect_counts else 0
    for defect, count in sorted(defect_counts.items(), key=lambda item: item[1], reverse=True):
        bar = build_ascii_bar(count, max_defect)
        percentage = count / total_queries
        print(f"   {defect:<15} : {count:>3} ({percentage:>6.1%}) | {bar}")
        
    print("-" * 60)
    # Severity distribution
    print(" Severity Level Distribution:")
    max_severity = max(severity_counts.values()) if severity_counts else 0
    for severity, count in sorted(severity_counts.items(), key=lambda item: item[1], reverse=True):
        bar = build_ascii_bar(count, max_severity)
        percentage = count / total_queries
        print(f"   {severity:<15} : {count:>3} ({percentage:>6.1%}) | {bar}")
        
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Monitor and analyze prediction logs from RAG assistant.")
    parser.add_argument("--logs", type=str, default=str(DEFAULT_LOG_FILE), help="Path to predictions.jsonl file.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold to trigger alerts.")
    args = parser.parse_args()

    analyze_logs(Path(args.logs), args.threshold)

if __name__ == "__main__":
    main()
