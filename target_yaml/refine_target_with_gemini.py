#!/usr/bin/env python3
"""
Gemini-powered refinement of target lexicon (L2/L3) with robust batching.

This script mirrors the resilient batching pattern from `analyze_corpus.py`.
It processes keywords in large windows (default 1000), retries each batch with
exponential backoff, automatically splits troublesome batches, and resumes
from previous progress by appending to the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
TARGET_DIR = ROOT / 'target_yaml'
RAW_DIR = TARGET_DIR / 'gemini_raw'
DEFAULT_OUTPUT = TARGET_DIR / 'target_lexicon_l2.generated.csv'
DEFAULT_TAXONOMY = TARGET_DIR / 'target_taxonomy.csv'

API_KEY = os.environ.get('GEMINI_API_KEY')
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview')

SEP_RE = re.compile(r'[,;/；、，/／|｜\s　]+')


@dataclass(frozen=True)
class KeywordRecord:
    keyword: str
    l1: str
    l2: str
    l3: str


@dataclass(frozen=True)
class ProcessingConfig:
    l2_to_l1: Dict[str, str]
    l1_to_l2: Dict[str, List[str]]
    max_retries: int
    pause: float
    raw_dir: Path


def split_terms(value: str | None) -> List[str]:
    if not value:
        return []
    return [piece.strip() for piece in SEP_RE.split(str(value)) if piece.strip()]


def collect_keywords(limit: int | None) -> List[str]:
    counters: Counter[str] = Counter()
    for csv_path in sorted(DATA_DIR.glob('*.csv')):
        try:
            with csv_path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for column in ('llm_keywords', 'primary_keywords', 'secondary_keywords', '対象物_all', '対象物_top3'):
                        if column in row:
                            counters.update(split_terms(row[column]))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f'[warn] failed to read {csv_path.name}: {exc}', file=sys.stderr)
    ordered = [kw for kw, _ in counters.most_common()]
    if limit is not None:
        return ordered[:limit]
    return ordered


def load_taxonomy(path: Path) -> ProcessingConfig:
    l2_to_l1: Dict[str, str] = {}
    l1_to_l2: Dict[str, List[str]] = defaultdict(list)
    if not path.exists():
        print(f'[warn] taxonomy not found at {path}; L1 will default to NONE', file=sys.stderr)
        return ProcessingConfig(l2_to_l1=l2_to_l1, l1_to_l2=l1_to_l2, max_retries=0, pause=0.0, raw_dir=RAW_DIR)

    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            l1 = (row.get('L1') or '').strip()
            l2 = (row.get('L2') or '').strip()
            if not l2:
                continue
            l2_to_l1[l2] = l1 or 'NONE'
            if l1:
                l1_to_l2[l1].append(l2)
    return ProcessingConfig(l2_to_l1=l2_to_l1, l1_to_l2=l1_to_l2, max_retries=0, pause=0.0, raw_dir=RAW_DIR)


def ensure_output(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['keyword', 'l1', 'l2', 'l3'])


def load_existing(path: Path) -> Dict[str, KeywordRecord]:
    existing: Dict[str, KeywordRecord] = {}
    if not path.exists():
        return existing
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keyword = (row.get('keyword') or '').strip()
            if keyword:
                existing[keyword] = KeywordRecord(
                    keyword=keyword,
                    l1=(row.get('l1') or 'NONE').strip(),
                    l2=(row.get('l2') or 'NONE').strip(),
                    l3=(row.get('l3') or 'NONE').strip(),
                )
    return existing


def append_records(path: Path, rows: Sequence[KeywordRecord]) -> None:
    if not rows:
        return
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row.keyword, row.l1, row.l2, row.l3])


def chunk_keywords(items: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index:index + batch_size])


def build_prompt(keywords: Sequence[str], l1_to_l2: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    lines.append('あなたは日本語の分類設計の専門家です。')
    lines.append('与えられたキーワードを既存のL2カテゴリに割り当て、必要であれば具体名としてL3も指定してください。')
    lines.append('必ずJSON配列のみを返し、余計な文章やコードフェンスを出力しないでください。')
    lines.append('フォーマット例: [{"keyword": "入力キーワード", "l2": "カテゴリ名", "l3": "具体名 or NONE"}]')
    lines.append('制約:')
    lines.append(' - keywordは入力と完全一致の表記を使う。')
    lines.append(' - 提示したL2以外は使用禁止。該当が無ければ "NONE"。')
    lines.append(' - L3が無い場合は "NONE"。')
    lines.append(f' - 要素数は{len(keywords)}件で、入力順を保持すること。')
    lines.append('\n利用可能なL1→L2:')
    for l1, l2_list in l1_to_l2.items():
        joined = ', '.join(l2_list[:50])
        lines.append(f'{l1}: {joined}')
    lines.append('\nキーワード:')
    for keyword in keywords:
        lines.append(f'- {keyword}')
    lines.append('\nJSONのみを返してください。')
    return '\n'.join(lines)


def configure_client():
    try:
        import google.genai as genai
    except Exception as exc:  # pragma: no cover - dependency check
        raise SystemExit('google-genai をインストールしてください (pip install google-genai)') from exc
    if not API_KEY:
        raise SystemExit('GEMINI_API_KEY が必要です')
    return genai.Client(api_key=API_KEY)


def call_model(client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            'temperature': 0.0,
            'max_output_tokens': 4096,
            'response_mime_type': 'application/json',
        },
    )
    text = getattr(response, 'text', None)
    if text:
        return text
    candidates = getattr(response, 'candidates', None) or []
    parts: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        if isinstance(content, list):
            for part in content:
                part_text = getattr(part, 'text', '')
                if part_text:
                    parts.append(part_text)
        else:
            parts.append(str(candidate))
    if parts:
        return '\n'.join(parts)
    raise RuntimeError('model returned empty response')


def extract_json_array(payload: str) -> str:
    match = re.search(r'\[.*\]', payload, flags=re.S)
    if match:
        snippet = match.group(0)
        json.loads(snippet)
        return snippet
    fenced = re.search(r'```json\s*(\[.*?\])\s*```', payload, flags=re.S)
    if fenced:
        snippet = fenced.group(1)
        json.loads(snippet)
        return snippet
    raise ValueError('no JSON array found in model output')


def normalise_results(keywords: Sequence[str], entries: Sequence[dict], l2_to_l1: Dict[str, str]) -> List[KeywordRecord]:
    if len(entries) != len(keywords):
        raise ValueError(f'expected {len(keywords)} items, received {len(entries)}')
    rows: List[KeywordRecord] = []
    for keyword, entry in zip(keywords, entries):
        if not isinstance(entry, dict):
            raise ValueError('parsed entry is not an object')
        parsed_keyword = (entry.get('keyword') or '').strip()
        if parsed_keyword != keyword:
            raise ValueError(f'keyword mismatch: expected "{keyword}" but got "{parsed_keyword}"')
        l2 = (entry.get('l2') or 'NONE').strip()
        l3 = (entry.get('l3') or 'NONE').strip()
        l1 = l2_to_l1.get(l2, 'NONE') if l2 != 'NONE' else 'NONE'
        rows.append(KeywordRecord(keyword=keyword, l1=l1, l2=l2, l3=l3))
    return rows


def save_raw(raw_dir: Path, label: str, attempt: int, payload: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f'{label}_try{attempt}.txt'
    with path.open('w', encoding='utf-8') as handle:
        handle.write(payload)


def process_chunk(client, keywords: Sequence[str], label: str, config: ProcessingConfig) -> List[KeywordRecord]:
    if not keywords:
        return []

    prompt = build_prompt(keywords, config.l1_to_l2)
    for attempt in range(1, config.max_retries + 1):
        payload = ''
        try:
            payload = call_model(client, prompt)
            json_text = extract_json_array(payload)
            parsed = json.loads(json_text)
            rows = normalise_results(keywords, parsed, config.l2_to_l1)
            return rows
        except Exception as exc:
            save_raw(config.raw_dir, label, attempt, payload or str(exc))
            print(f'[warn] {label} attempt {attempt}/{config.max_retries} failed: {exc}', file=sys.stderr)
            time.sleep(config.pause * attempt)

    if len(keywords) == 1:
        keyword = keywords[0]
        print(f'[error] giving up on "{keyword}" -> marking as NONE', file=sys.stderr)
        return [KeywordRecord(keyword=keyword, l1='NONE', l2='NONE', l3='NONE')]

    midpoint = len(keywords) // 2
    left = process_chunk(client, keywords[:midpoint], f'{label}a', config)
    right = process_chunk(client, keywords[midpoint:], f'{label}b', config)
    return left + right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate target lexicon mappings with Gemini.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of keywords to process (testing).')
    parser.add_argument('--batch-size', type=int, default=1000, help='Primary batch size before adaptive splitting.')
    parser.add_argument('--max-retries', type=int, default=3, help='Retries before splitting a batch.')
    parser.add_argument('--pause', type=float, default=2.0, help='Seconds to wait between retries (exponential).')
    parser.add_argument('--taxonomy', type=Path, default=DEFAULT_TAXONOMY, help='Path to taxonomy CSV.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Path to output CSV.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = configure_client()

    print('Collecting keywords from data/...')
    keywords = collect_keywords(args.limit)
    if not keywords:
        raise SystemExit('No keywords collected from data directory.')
    print(f'Collected {len(keywords)} keywords')

    taxonomy_cfg = load_taxonomy(args.taxonomy)
    config = ProcessingConfig(
        l2_to_l1=taxonomy_cfg.l2_to_l1,
        l1_to_l2=taxonomy_cfg.l1_to_l2,
        max_retries=max(1, args.max_retries),
        pause=max(0.5, args.pause),
        raw_dir=RAW_DIR,
    )

    ensure_output(args.output)
    processed = load_existing(args.output)
    seen = set(processed.keys())
    remaining = [kw for kw in keywords if kw not in seen]

    print(f'{len(seen)} keywords already processed; {len(remaining)} left')

    progress = tqdm(total=len(remaining), desc='Processing keywords', unit='kw') if tqdm else None

    total = len(seen)
    for batch_index, batch in enumerate(chunk_keywords(remaining, args.batch_size), start=1):
        label = f'batch{batch_index:05d}'
        rows = process_chunk(client, batch, label, config)
        fresh = [row for row in rows if row.keyword not in seen]
        append_records(args.output, fresh)
        seen.update(row.keyword for row in fresh)
        total += len(fresh)
        if progress:
            progress.update(len(fresh))
        else:
            print(f'Processed {total}/{len(keywords)} keywords', end='\r')

    if progress:
        progress.close()

    print('\nCompleted processing. Total keywords written:', len(seen))


if __name__ == '__main__':
    main()