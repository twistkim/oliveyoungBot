# upload_sheets.py
"""Upload pc_ads.csv and m_products.csv to Google Sheets.

Creates a date-stamped spreadsheet in the specified Google Drive folder
with two sheets: pc_ads and m_products.

Usage:
  python upload_sheets.py --folder_id YOUR_FOLDER_ID
  python upload_sheets.py --folder_id YOUR_FOLDER_ID --date 2026-02-09
  python upload_sheets.py --folder_id YOUR_FOLDER_ID --pc pc_ads.csv --m m_products.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CREDENTIALS_FILE = Path(__file__).parent / "ajdbot-d5f8c4f16102.json"


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def upload(folder_id: str, pc_path: Path, m_path: Path, title: str) -> str:
    creds = Credentials.from_service_account_file(str(CREDENTIALS_FILE), scopes=SCOPES)
    gc = gspread.authorize(creds)

    spreadsheet = gc.create(title, folder_id=folder_id)
    print(f"Created: {title}")
    print(f"URL: {spreadsheet.url}")

    # --- pc_ads ---
    pc_data = read_csv(pc_path)
    pc_sheet = spreadsheet.sheet1
    pc_sheet.update_title("pc_ads")
    if pc_data:
        pc_sheet.update(pc_data, value_input_option="RAW")
        print(f"  pc_ads: {len(pc_data) - 1} rows uploaded")

    # --- m_products ---
    m_data = read_csv(m_path)
    m_sheet = spreadsheet.add_worksheet(title="m_products", rows=len(m_data) + 1, cols=20)
    if m_data:
        m_sheet.update(m_data, value_input_option="RAW")
        print(f"  m_products: {len(m_data) - 1} rows uploaded")

    return spreadsheet.url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_id", required=True, help="Google Drive folder ID")
    ap.add_argument("--pc", default="pc_ads.csv", help="PC ads CSV path")
    ap.add_argument("--m", default="m_products.csv", help="Mobile products CSV path")
    ap.add_argument("--date", default=str(date.today()), help="Date label (default: today)")
    args = ap.parse_args()

    pc_path = Path(args.pc)
    m_path = Path(args.m)

    if not pc_path.exists():
        raise FileNotFoundError(f"{pc_path} not found")
    if not m_path.exists():
        raise FileNotFoundError(f"{m_path} not found")

    title = f"{args.date}_수집결과"
    url = upload(args.folder_id, pc_path, m_path, title)
    print(f"\nDone! {url}")


if __name__ == "__main__":
    main()
