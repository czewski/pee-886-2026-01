import csv
import json
from pathlib import Path
from typing import Dict, List

__all__ = ["save_metrics_json", "save_test_predictions_csv"]


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """Persist metrics dictionary as formatted JSON."""
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)


def save_test_predictions_csv(rows: List[Dict], output_path: Path) -> None:
    """Persist per-sample predictions to CSV when rows are available."""
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
