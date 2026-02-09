# keyword_manager.py
from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

DEFAULT_SPLIT_DIR = Path("keywords_split")

def load_keywords(path: str | Path) -> List[str]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    # 공백/중복 제거 + 빈줄 제거
    seen = set()
    out = []
    for line in lines:
        kw = line.strip()
        if not kw:
            continue
        if kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out

def classify_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """
    규칙 기반 분류 예시.
    rules에 너 업종에 맞춰 키워드 패턴을 계속 추가하면 됨.
    """
    rules: List[Tuple[str, List[str]]] = [
        # Olive Young–focused categories
        ("oy_brand", [r"올리브영", r"올영", r"olive\s*young", r"oliveyoung"]),

        # 스킨케어
        ("skincare", [
            r"스킨", r"토너", r"로션", r"에센스", r"세럼", r"앰플", r"크림", r"아이크림",
            r"미스트", r"오일", r"패드", r"마스크", r"팩", r"모공", r"각질", r"필링",
            r"클렌징", r"폼클렌징", r"클렌징오일", r"클렌징밤", r"선크림", r"선케어", r"선스틱",
            r"수분", r"보습", r"진정", r"미백", r"주름", r"탄력", r"리프팅", r"장벽", r"세라마이드",
            r"트러블", r"여드름", r"피지", r"블랙헤드", r"화이트헤드", r"민감", r"홍조", r"레티놀",
            r"비타민\s*c", r"나이아신", r"히알루론", r"시카", r"병풀", r"판테놀", r"BHA", r"AHA", r"PHA"
        ]),

        # 메이크업
        ("makeup", [
            r"쿠션", r"파운데이션", r"컨실러", r"파우더", r"프라이머", r"베이스", r"톤업",
            r"립", r"틴트", r"립스틱", r"블러셔", r"치크", r"섀도우", r"아이섀도우", r"마스카라",
            r"아이라이너", r"브로우", r"하이라이터", r"쉐딩", r"픽서", r"세팅", r"네일"
        ]),

        # 헤어케어
        ("hair", [
            r"샴푸", r"트리트먼트", r"컨디셔너", r"헤어팩", r"헤어마스크", r"헤어오일", r"에센스",
            r"두피", r"탈모", r"헤어토닉", r"헤어미스트", r"헤어스프레이", r"왁스", r"염색", r"컬러"
        ]),

        # 바디/핸드
        ("body", [
            r"바디워시", r"바디로션", r"바디크림", r"바디미스트", r"스크럽", r"바디오일",
            r"핸드크림", r"핸드워시", r"풋", r"데오", r"데오드란트", r"제모", r"왁싱"
        ]),

        # 향수/프래그런스
        ("fragrance", [r"향수", r"퍼퓸", r"코롱", r"디퓨저", r"캔들", r"룸스프레이"]),

        # 헬스/이너뷰티
        ("health", [
            r"비타민", r"유산균", r"오메가", r"프로틴", r"단백질", r"콜라겐", r"마그네슘",
            r"철분", r"루테인", r"다이어트", r"슬리밍", r"보충제", r"영양제", r"이너뷰티"
        ]),

        # 뷰티툴/디바이스
        ("device_tool", [
            r"뷰러", r"퍼프", r"브러시", r"스펀지", r"화장솜", r"면봉", r"고데기",
            r"드라이기", r"미용기기", r"디바이스", r"마사지기", r"LED", r"클렌징기"
        ]),

        # 남성/그루밍
        ("men", [r"남성", r"맨즈", r"올인원", r"쉐이빙", r"면도", r"애프터쉐이브"]),

        # 베이비/패밀리
        ("baby_family", [r"아기", r"베이비", r"키즈", r"임산부", r"패밀리", r"기저귀", r"물티슈"]),

        # 위에 어디에도 안 걸리면 기타
        ("etc", [r"."])
    ]

    compiled = [(name, [re.compile(pat, re.IGNORECASE) for pat in pats]) for name, pats in rules]

    buckets: Dict[str, List[str]] = {name: [] for name, _ in rules}
    buckets["other"] = []

    for kw in keywords:
        placed = False
        for name, pats in compiled:
            if any(p.search(kw) for p in pats):
                buckets[name].append(kw)
                placed = True
                break
        if not placed:
            buckets["other"].append(kw)

    # 비어있는 분류는 제거(선택)
    buckets = {k: v for k, v in buckets.items() if v}
    return buckets

def write_split_files(buckets: Dict[str, List[str]], out_dir: str | Path = DEFAULT_SPLIT_DIR) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    for category, kws in buckets.items():
        fp = out_dir / f"{category}.txt"
        fp.write_text("\n".join(kws) + "\n", encoding="utf-8")
        created.append(fp)
    return created

def split_keywords(input_path: str | Path = "keywords.txt", out_dir: str | Path = DEFAULT_SPLIT_DIR) -> List[Path]:
    keywords = load_keywords(input_path)
    buckets = classify_keywords(keywords)
    files = write_split_files(buckets, out_dir)
    return files

def iter_split_files(out_dir: str | Path = DEFAULT_SPLIT_DIR) -> List[Path]:
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*.txt"))
    return files

def round_robin_keywords(out_dir: str | Path = DEFAULT_SPLIT_DIR) -> Iterator[Tuple[str, str]]:
    """
    (분류명, 키워드) 형태로 무한 순환 제너레이터.
    - 분류 파일들을 읽고
    - 각 분류에서 1개씩 번갈아가며 뽑음
    """
    files = iter_split_files(out_dir)
    buckets: List[Tuple[str, List[str]]] = []

    for fp in files:
        category = fp.stem
        kws = load_keywords(fp)
        if kws:
            buckets.append((category, kws))

    if not buckets:
        raise RuntimeError(f"No keyword files found in {out_dir} or all empty.")

    idx = 0
    pos = [0] * len(buckets)

    while True:
        category, kws = buckets[idx]
        yield category, kws[pos[idx]]
        pos[idx] = (pos[idx] + 1) % len(kws)
        idx = (idx + 1) % len(buckets)

if __name__ == "__main__":
    created = split_keywords("keywords.txt", DEFAULT_SPLIT_DIR)
    print("✅ created:", [str(p) for p in created])