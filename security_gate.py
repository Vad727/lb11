import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semgrep", required=True, help="Path to Semgrep JSON report")
    parser.add_argument(
        "--threshold",
        default="ERROR",
        choices=["INFO", "WARNING", "ERROR"],
        help="Fail workflow if result severity is at or above this level",
    )
    args = parser.parse_args()

    severity_order = {"INFO": 1, "WARNING": 2, "ERROR": 3}
    threshold_value = severity_order[args.threshold]

    report_path = Path(args.semgrep)
    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        return 1

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    blocking = []

    for item in results:
        severity = item.get("extra", {}).get("severity", "INFO")
        if severity_order.get(severity, 0) >= threshold_value:
            blocking.append(item)

    print(f"Total findings: {len(results)}")
    print(f"Blocking findings: {len(blocking)}")

    if blocking:
        print("[SECURITY GATE] Workflow blocked due to high-severity findings.")
        for item in blocking:
            path = item.get("path", "unknown")
            line = item.get("start", {}).get("line", "?")
            rule = item.get("check_id", "unknown_rule")
            msg = item.get("extra", {}).get("message", "")
            sev = item.get("extra", {}).get("severity", "INFO")
            print(f"- {sev} | {rule} | {path}:{line} | {msg}")
        return 1

    print("[SECURITY GATE] No blocking findings. Workflow passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())