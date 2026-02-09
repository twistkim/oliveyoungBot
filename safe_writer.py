import csv
import json
from pathlib import Path
from datetime import datetime

def append_row(csv_path: str, fieldnames: list[str], row: dict):
    path = Path(csv_path)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

def save_state(state_path: str, data: dict):
    data = dict(data)
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    Path(state_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state(state_path: str) -> dict | None:
    p = Path(state_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))