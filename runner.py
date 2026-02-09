# runner.py
"""Run keyword collection with safe, incremental saving.

What this does:
- Reads already-split keyword files under DEFAULT_SPLIT_DIR (e.g., keywords_split/*.txt)
- Processes keywords in round-robin order across categories
- Saves results incrementally (append to CSV per keyword) so crashes don't lose progress
- Writes a checkpoint state.json so you can resume a finite "once" run

Outputs:
- pc_ads.csv (from https://ad.search.naver.com/search.naver?where=ad&query=...)
- m_products.csv (from https://m.search.naver.com/search.naver?where=m&query=...)

Usage examples:
- One pass through all keywords (recommended for 5,000+):
  python runner.py --mode once --sleep 0.8

- Infinite loop (keeps cycling):
  python runner.py --mode loop --sleep 1.0

- Use Playwright fallback (if HTML parsing often returns 0):
  python runner.py --mode once --use_playwright
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterator
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from keyword_manager import DEFAULT_SPLIT_DIR


PC_AD_URL = "https://m.ad.search.naver.com/search.naver?where=ad&query={q}"
MOBILE_URL = "https://m.search.naver.com/search.naver?where=m&query={q}"


PC_FIELDS = [
    "keyword",
    "category",
    "rank",
    "site_name",
    "title",
    "desc",
    "display_url",
    "landing_href",
    "fetched_at",
    "source_url",
]

M_FIELDS = [
    "keyword",
    "category",
    "rank",
    "product_name",
    "price",
    "seller",
    "fetched_at",
    "source_url",
]


def safe_slug(s: str, max_len: int = 80) -> str:
    """Make a filesystem-safe slug."""
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z가-힣_\-]", "", s)
    return s[:max_len] if len(s) > max_len else s

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def build_headers(mobile: bool = False) -> dict:
    if mobile:
        ua = (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        )
    else:
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    return {
        "User-Agent": ua,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def fetch_html_requests(url: str, mobile: bool, timeout: int = 15) -> tuple[str, str, int]:
    r = requests.get(url, headers=build_headers(mobile=mobile), timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text, r.url, r.status_code


async def fetch_html_playwright(url: str, mobile: bool, timeout_ms: int = 20000) -> str:
    """Optional fallback when requests HTML is incomplete due to JS rendering."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=build_headers(mobile=mobile)["User-Agent"],
            locale="ko-KR",
            viewport={"width": 390, "height": 844} if mobile else {"width": 1280, "height": 720},
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(800)
        html = await page.content()
        await context.close()
        await browser.close()
        return html


def looks_blocked(html: str) -> bool:
    h = html.lower()
    # Common signals for bot-block, consent, or unexpected pages
    signals = [
        "captcha",
        "자동입력",
        "보안문자",
        "접근이 제한",
        "일시적으로 차단",
        "robots",
        "unusual traffic",
        "동의",
        "consent",
        "redirect",
    ]
    return any(sig.lower() in h for sig in signals)

def parse_pc_ads(html: str, keyword: str, category: str, source_url: str, max_items: int) -> List[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: List[dict] = []

    items = soup.select("li.lst")
    rank = 0

    for li in items:
        site_el = li.select_one("a.site")
        site_name = clean_text(site_el.get_text(" ", strip=True)) if site_el else ""

        title_spans = li.select("a.tit_wrap span.lnk_tit")
        titles = [clean_text(sp.get_text(" ", strip=True)) for sp in title_spans]
        titles = [t for t in titles if t]
        title = " | ".join(titles)

        desc_el = li.select_one("a.link_desc")
        desc = clean_text(desc_el.get_text(" ", strip=True)) if desc_el else ""

        url_el = li.select_one("span.lnk_url_area a.url")
        display_url = clean_text(url_el.get_text(" ", strip=True)) if url_el else ""

        landing_el = li.select_one("a.tit_wrap")
        landing_href = landing_el.get("href") if landing_el else ""

        if not (site_name or title or desc):
            continue

        rank += 1
        rows.append(
            {
                "keyword": keyword,
                "category": category,
                "rank": rank,
                "site_name": site_name,
                "title": title,
                "desc": desc,
                "display_url": display_url,
                "landing_href": landing_href,
                "fetched_at": now_iso(),
                "source_url": source_url,
            }
        )

        if max_items and rank >= max_items:
            break

    return rows


def parse_mobile_products(html: str, keyword: str, category: str, source_url: str, max_items: int) -> List[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows: List[dict] = []

    # 1차: 샘플 클래스 기반
    items = soup.select("li.ds9RptR1")
    rank = 0

    for li in items:
        name_el = li.select_one("strong.MpU43GH6 span") or li.select_one("strong.MpU43GH6")
        product_name = clean_text(name_el.get_text(" ", strip=True)) if name_el else ""

        price_el = li.select_one("span.Lkx7X0Il")
        price = clean_text(price_el.get_text(" ", strip=True)) if price_el else ""

        seller_el = li.select_one("span.PtxugWXH")
        seller = clean_text(seller_el.get_text(" ", strip=True)) if seller_el else ""

        if not product_name:
            continue

        rank += 1
        rows.append(
            {
                "keyword": keyword,
                "category": category,
                "rank": rank,
                "product_name": product_name,
                "price": price,
                "seller": seller,
                "fetched_at": now_iso(),
                "source_url": source_url,
            }
        )

        if max_items and rank >= max_items:
            break

    # 2차 fallback: 위 결과가 아예 없을 때 느슨하게 추출
    if not rows:
        candidates = soup.select("li strong span, li strong")
        for el in candidates:
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt or len(txt) < 6:
                continue
            rank += 1
            rows.append(
                {
                    "keyword": keyword,
                    "category": category,
                    "rank": rank,
                    "product_name": txt,
                    "price": "",
                    "seller": "",
                    "fetched_at": now_iso(),
                    "source_url": source_url,
                }
            )
            if max_items and rank >= max_items:
                break

    return rows


def append_rows(csv_path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    """Append rows to CSV; writes header if file doesn't exist."""
    if not rows:
        return

    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_keywords_file(fp: Path) -> List[str]:
    lines = fp.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    seen = set()
    for line in lines:
        kw = line.strip()
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out


def build_round_robin_once(split_dir: Path, state: dict | None) -> Tuple[List[Tuple[str, List[str]]], List[int]]:
    """Prepare (category -> keyword list) and positions for a finite one-pass run.

    state format for positions:
      {
        "positions": {"skincare": 120, "makeup": 55, ...}
      }

    positions are next index to process for each category.
    """
    files = sorted(split_dir.glob("*.txt"))
    buckets: List[Tuple[str, List[str]]] = []

    for fp in files:
        category = fp.stem
        kws = load_keywords_file(fp)
        if kws:
            buckets.append((category, kws))

    if not buckets:
        raise RuntimeError(f"No keyword files found in {split_dir} (or all empty).")

    pos_map = (state or {}).get("positions", {})
    positions: List[int] = []
    for category, kws in buckets:
        p = int(pos_map.get(category, 0))
        # clamp
        if p < 0:
            p = 0
        if p > len(kws):
            p = len(kws)
        positions.append(p)

    return buckets, positions


def iter_round_robin_once(buckets: List[Tuple[str, List[str]]], positions: List[int]) -> Iterator[Tuple[str, str, int]]:
    """Yield each remaining keyword exactly once, in round-robin order.

    Yields: (category, keyword, index_in_category)
    """
    n = len(buckets)
    remaining = True
    idx = 0

    while remaining:
        remaining = False
        for _ in range(n):
            category, kws = buckets[idx]
            p = positions[idx]
            if p < len(kws):
                remaining = True
                kw = kws[p]
                yield category, kw, p
                positions[idx] = p + 1
            idx = (idx + 1) % n


def load_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(state_path: Path, state: dict) -> None:
    state = dict(state)
    state["saved_at"] = now_iso()
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def process_keyword(
    keyword: str,
    category: str,
    out_pc: Path,
    out_m: Path,
    max_pc: int,
    max_m: int,
    use_playwright: bool,
    timeout: int,
    debug: bool,
    debug_dump: bool,
    debug_dump_dir: Path,
    debug_dump_budget: int,
    debug_dumps_done_ref: dict,
) -> Tuple[int, int]:
    """Collect both PC ads and mobile products for a single keyword.

    Returns: (pc_rows_count, mobile_rows_count)
    """
    q = quote(keyword)
    pc_url = PC_AD_URL.format(q=q)
    m_url = MOBILE_URL.format(q=q)

    pc_rows: List[dict] = []
    m_rows: List[dict] = []

    # PC first
    try:
        pc_html, pc_final_url, pc_status = fetch_html_requests(pc_url, mobile=True, timeout=timeout)
        pc_rows = parse_pc_ads(pc_html, keyword, category, pc_url, max_items=max_pc)
    except Exception:
        pc_rows = []
        pc_html = ""
        pc_final_url = pc_url
        pc_status = "-"

    # Mobile second
    try:
        m_html, m_final_url, m_status = fetch_html_requests(m_url, mobile=True, timeout=timeout)
        m_rows = parse_mobile_products(m_html, keyword, category, m_url, max_items=max_m)
    except Exception:
        m_rows = []
        m_html = ""
        m_final_url = m_url
        m_status = "-"

    # Optional Playwright fallback if either returned 0
    if use_playwright and (not pc_rows or not m_rows):
        import asyncio

        if not pc_rows:
            try:
                pc_html = asyncio.run(fetch_html_playwright(pc_url, mobile=True))
                pc_rows = parse_pc_ads(pc_html, keyword, category, pc_url, max_items=max_pc)
            except Exception:
                pc_rows = []

        if not m_rows:
            try:
                m_html = asyncio.run(fetch_html_playwright(m_url, mobile=True))
                m_rows = parse_mobile_products(m_html, keyword, category, m_url, max_items=max_m)
            except Exception:
                m_rows = []

    if debug and (len(pc_rows) == 0 and len(m_rows) == 0):
        # Print small diagnostics
        try:
            pc_len = len(pc_html) if 'pc_html' in locals() and pc_html else 0
        except Exception:
            pc_len = 0
        try:
            m_len = len(m_html) if 'm_html' in locals() and m_html else 0
        except Exception:
            m_len = 0

        pc_info = f"pc_final={locals().get('pc_final_url', pc_url)} status={locals().get('pc_status', '-') } len={pc_len} blocked={looks_blocked(locals().get('pc_html','') or '')}"
        m_info = f"m_final={locals().get('m_final_url', m_url)} status={locals().get('m_status', '-') } len={m_len} blocked={looks_blocked(locals().get('m_html','') or '')}"
        print(f"  [DEBUG] {pc_info}")
        print(f"  [DEBUG] {m_info}")

    if debug_dump and (len(pc_rows) == 0 and len(m_rows) == 0):
        # Dump a few samples to inspect HTML when selectors fail or blocked.
        done = int(debug_dumps_done_ref.get('done', 0))
        if done < debug_dump_budget:
            debug_dump_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{safe_slug(category)}__{safe_slug(keyword)}"
            pc_path = debug_dump_dir / f"pc__{slug}.html"
            m_path = debug_dump_dir / f"m__{slug}.html"
            try:
                pc_path.write_text(locals().get('pc_html','') or '', encoding='utf-8')
                m_path.write_text(locals().get('m_html','') or '', encoding='utf-8')
                debug_dumps_done_ref['done'] = done + 1
                print(f"  [DUMP] saved HTML -> {pc_path} , {m_path}")
            except Exception as e:
                print(f"  [DUMP][ERROR] {e}")

    # ✅ Incremental save (append) so progress is not lost
    append_rows(out_pc, PC_FIELDS, pc_rows)
    append_rows(out_m, M_FIELDS, m_rows)

    return (len(pc_rows), len(m_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", default=str(DEFAULT_SPLIT_DIR), help="Directory containing split keyword txt files")
    ap.add_argument("--mode", choices=["once", "loop"], default="once", help="Run once through all keywords or loop forever")
    ap.add_argument("--sleep", type=float, default=0.8, help="Base sleep seconds between keywords")
    ap.add_argument("--jitter", type=float, default=0.4, help="Random jitter added to sleep (0~jitter)")
    ap.add_argument("--out_pc", default="pc_ads.csv", help="PC output CSV")
    ap.add_argument("--out_m", default="m_products.csv", help="Mobile output CSV")
    ap.add_argument("--max_pc", type=int, default=5, help="Max PC ads rows per keyword")
    ap.add_argument("--max_m", type=int, default=5, help="Max mobile product rows per keyword")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    ap.add_argument("--use_playwright", action="store_true", help="Use Playwright fallback when parsing returns 0")
    ap.add_argument("--state", default="state.json", help="Checkpoint state file")
    ap.add_argument("--debug", action="store_true", help="Print extra debug info when PC/M results are 0")
    ap.add_argument("--debug_dump", action="store_true", help="Dump HTML to debug_html/ when PC/M results are 0")
    ap.add_argument("--debug_limit", type=int, default=30, help="Max number of debug dumps per run")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    out_pc = Path(args.out_pc)
    out_m = Path(args.out_m)
    state_path = Path(args.state)

    state = load_state(state_path) or {}
    debug_ref = {"done": 0}

    if args.mode == "once":
        buckets, positions = build_round_robin_once(split_dir, state)

        total_pc = 0
        total_m = 0
        total_done = 0

        for category, keyword, index_in_category in iter_round_robin_once(buckets, positions):
            try:
                pc_n, m_n = process_keyword(
                    keyword=keyword,
                    category=category,
                    out_pc=out_pc,
                    out_m=out_m,
                    max_pc=args.max_pc,
                    max_m=args.max_m,
                    use_playwright=args.use_playwright,
                    timeout=args.timeout,
                    debug=args.debug,
                    debug_dump=args.debug_dump,
                    debug_dump_dir=Path("debug_html"),
                    debug_dump_budget=args.debug_limit,
                    debug_dumps_done_ref=debug_ref,
                )
                total_pc += pc_n
                total_m += m_n

                # progress log
                print(f"[{category}] {keyword}  -> PC:{pc_n} / M:{m_n}")

                # ✅ checkpoint per keyword
                pos_map = state.get("positions", {})
                pos_map[category] = index_in_category + 1
                state["positions"] = pos_map
                state["last"] = {"category": category, "keyword": keyword}
                save_state(state_path, state)

            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user (Ctrl+C). State saved.")
                save_state(state_path, state)
                break
            except Exception as e:
                # Save error to state and continue
                state["last_error"] = {"category": category, "keyword": keyword, "error": str(e)}
                save_state(state_path, state)
                print(f"[ERROR] [{category}] {keyword} -> {e}")

            total_done += 1
            time.sleep(max(0.0, args.sleep + random.uniform(0, args.jitter)))

        print(f"\n✅ DONE (once). keywords={total_done}, pc_rows={total_pc}, m_rows={total_m}")

    else:
        # loop forever: cycle through split files continually
        # This does not try to resume exact position; it just keeps running.
        # We still checkpoint the last processed keyword.
        files = sorted(split_dir.glob("*.txt"))
        if not files:
            raise RuntimeError(f"No keyword files found in {split_dir}.")

        buckets: List[Tuple[str, List[str]]] = []
        for fp in files:
            category = fp.stem
            kws = load_keywords_file(fp)
            if kws:
                buckets.append((category, kws))

        if not buckets:
            raise RuntimeError(f"All keyword files in {split_dir} are empty.")

        pos = [0] * len(buckets)
        idx = 0

        while True:
            category, kws = buckets[idx]
            keyword = kws[pos[idx]]
            pos[idx] = (pos[idx] + 1) % len(kws)
            idx = (idx + 1) % len(buckets)

            try:
                pc_n, m_n = process_keyword(
                    keyword=keyword,
                    category=category,
                    out_pc=out_pc,
                    out_m=out_m,
                    max_pc=args.max_pc,
                    max_m=args.max_m,
                    use_playwright=args.use_playwright,
                    timeout=args.timeout,
                    debug=args.debug,
                    debug_dump=args.debug_dump,
                    debug_dump_dir=Path("debug_html"),
                    debug_dump_budget=args.debug_limit,
                    debug_dumps_done_ref=debug_ref,
                )

                print(f"[{category}] {keyword}  -> PC:{pc_n} / M:{m_n}")

                state["last"] = {"category": category, "keyword": keyword}
                save_state(state_path, state)

            except KeyboardInterrupt:
                print("\n⏹️ Stopped by user (Ctrl+C). State saved.")
                save_state(state_path, state)
                break
            except Exception as e:
                state["last_error"] = {"category": category, "keyword": keyword, "error": str(e)}
                save_state(state_path, state)
                print(f"[ERROR] [{category}] {keyword} -> {e}")

            time.sleep(max(0.0, args.sleep + random.uniform(0, args.jitter)))


if __name__ == "__main__":
    main()