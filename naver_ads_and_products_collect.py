import argparse
import csv
import random
import re
import time
from datetime import datetime
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


PC_AD_URL = "https://ad.search.naver.com/search.naver?where=ad&query={q}"
MOBILE_URL = "https://m.search.naver.com/search.naver?where=m&query={q}"


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


def fetch_html_requests(url: str, mobile: bool, timeout: int = 15) -> str:
    r = requests.get(url, headers=build_headers(mobile=mobile), timeout=timeout)
    r.raise_for_status()
    return r.text


async def fetch_html_playwright(url: str, mobile: bool, timeout_ms: int = 20000) -> str:
    # Optional fallback (브라우저 렌더링)
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

        # 가벼운 대기 (네이버가 lazy load 할 수 있어)
        await page.wait_for_timeout(800)

        html = await page.content()
        await context.close()
        await browser.close()
        return html


def parse_pc_ads(html: str, keyword: str, source_url: str, max_items: int) -> list[dict]:
    """
    PC 광고 페이지(ad.search.naver.com)에서:
    - site_name: a.site
    - title: a.tit_wrap 안 span.lnk_tit 여러 개를 합침
    - desc: a.link_desc
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

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

        # 유효성: site/title/desc 중 하나라도 있어야
        if not (site_name or title or desc):
            continue

        rank += 1
        rows.append(
            {
                "keyword": keyword,
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


def parse_mobile_products(html: str, keyword: str, source_url: str, max_items: int) -> list[dict]:
    """
    모바일 검색(m.search.naver.com)에서 (샘플 기준):
    - li.ds9RptR1 (상품 카드)
      - product_name: strong.MpU43GH6 span
      - price: span.Lkx7X0Il
      - seller: span.PtxugWXH
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

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

        # fallback: 이름은 있는데 가격/판매처가 비는 경우가 가끔 있음
        if not product_name:
            continue

        rank += 1
        rows.append(
            {
                "keyword": keyword,
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

    # 2차 fallback (클래스가 바뀌었을 때 대비):
    # 위 결과가 아예 없으면, strong 태그 중 mark 포함 텍스트를 상품명 후보로 잡는 아주 느슨한 방식
    if not rows:
        candidates = soup.select("li strong span, li strong")
        for el in candidates:
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt:
                continue
            # 너무 짧거나 일반 텍스트는 제외 (필요하면 조건 조절)
            if len(txt) < 6:
                continue

            rank += 1
            rows.append(
                {
                    "keyword": keyword,
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


def read_keywords(path: str) -> list[str]:
    kws: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if not kw:
                continue
            kws.append(kw)
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for k in kws:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keywords", required=True, help="keywords.txt (one keyword per line)")
    ap.add_argument("--out_pc", default="pc_ads.csv", help="output csv for PC ads")
    ap.add_argument("--out_m", default="m_products.csv", help="output csv for mobile products")
    ap.add_argument("--max_pc", type=int, default=10, help="max ads per keyword (PC)")
    ap.add_argument("--max_m", type=int, default=10, help="max products per keyword (Mobile)")
    ap.add_argument("--sleep_min", type=float, default=0.6, help="min sleep between requests")
    ap.add_argument("--sleep_max", type=float, default=1.2, help="max sleep between requests")
    ap.add_argument("--use_playwright", action="store_true", help="use Playwright fallback when parsing gets 0 rows")
    args = ap.parse_args()

    keywords = read_keywords(args.keywords)

    all_pc_rows: list[dict] = []
    all_m_rows: list[dict] = []

    # playwright는 필요할 때만 import/실행
    import asyncio

    for kw in tqdm(keywords, desc="Collecting"):
        q = quote(kw)

        # 1) PC ads
        pc_url = PC_AD_URL.format(q=q)
        pc_html = ""
        pc_rows = []
        try:
            pc_html = fetch_html_requests(pc_url, mobile=False)
            pc_rows = parse_pc_ads(pc_html, kw, pc_url, max_items=args.max_pc)
        except Exception as e:
            pc_rows = []
            # 필요하면 로그 출력
            # print(f"[PC] fetch/parse error for {kw}: {e}")

        if args.use_playwright and not pc_rows:
            try:
                pc_html = asyncio.run(fetch_html_playwright(pc_url, mobile=False))
                pc_rows = parse_pc_ads(pc_html, kw, pc_url, max_items=args.max_pc)
            except Exception as e:
                pc_rows = []
                # print(f"[PC][PW] error for {kw}: {e}")

        all_pc_rows.extend(pc_rows)

        time.sleep(random.uniform(args.sleep_min, args.sleep_max))

        # 2) Mobile products
        m_url = MOBILE_URL.format(q=q)
        m_html = ""
        m_rows = []
        try:
            m_html = fetch_html_requests(m_url, mobile=True)
            m_rows = parse_mobile_products(m_html, kw, m_url, max_items=args.max_m)
        except Exception as e:
            m_rows = []
            # print(f"[M] fetch/parse error for {kw}: {e}")

        if args.use_playwright and not m_rows:
            try:
                m_html = asyncio.run(fetch_html_playwright(m_url, mobile=True))
                m_rows = parse_mobile_products(m_html, kw, m_url, max_items=args.max_m)
            except Exception as e:
                m_rows = []
                # print(f"[M][PW] error for {kw}: {e}")

        all_m_rows.extend(m_rows)

        time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    # 저장
    pc_fields = [
        "keyword", "rank", "site_name", "title", "desc",
        "display_url", "landing_href", "fetched_at", "source_url",
    ]
    m_fields = [
        "keyword", "rank", "product_name", "price", "seller",
        "fetched_at", "source_url",
    ]

    write_csv(args.out_pc, all_pc_rows, pc_fields)
    write_csv(args.out_m, all_m_rows, m_fields)

    print(f"✅ PC ads saved: {args.out_pc} ({len(all_pc_rows)} rows)")
    print(f"✅ Mobile products saved: {args.out_m} ({len(all_m_rows)} rows)")


if __name__ == "__main__":
    main()