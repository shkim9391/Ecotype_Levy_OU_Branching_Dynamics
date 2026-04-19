from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TEXT_EXTS = {'.csv', '.tsv', '.txt', '.json', '.yaml', '.yml', '.rds.txt', '.log'}
BINARY_META_EXTS = {'.npy', '.npz', '.pkl', '.pickle', '.joblib'}

AXES = [
    'pc1',
    'pc2',
    'ilr_stem_vs_committed',
    'ilr_prog_vs_mature',
    'ilr_gmp_vs_monodc',
    'log_aux_clp',
    'log_aux_erybaso',
]

AXIS_ALIASES = {
    'pc1': ['pc1'],
    'pc2': ['pc2'],
    'ilr_stem_vs_committed': ['ilr_stem_vs_committed', 'stem_vs_committed', 'stem', 'committed'],
    'ilr_prog_vs_mature': ['ilr_prog_vs_mature', 'prog_vs_mature', 'prog', 'mature'],
    'ilr_gmp_vs_monodc': ['ilr_gmp_vs_monodc', 'gmp_vs_monodc', 'gmp', 'monodc'],
    'log_aux_clp': ['log_aux_clp', 'aux_clp', 'clp'],
    'log_aux_erybaso': ['log_aux_erybaso', 'aux_erybaso', 'erybaso'],
}

MODEL_TERMS = [
    'frozen', 'model', 'predict', 'projection', 'projected', 'ridge', 'coef',
    'coeff', 'coefficient', 'beta', 'weight', 'loading', 'intercept', 'malignant'
]
CALIBRATION_TERMS = [
    'calibration', 'calibrate', 'secondary', 'affine', 'intercept', 'slope',
    'comparison', 'projected', 'predicted', 'sample_summary'
]

HEADER_PATTERN = re.compile(r'[\t,]')


def normalize_text(text: str) -> str:
    return text.lower().replace('-', '_').replace(' ', '_')


def safe_read_head(path: Path, max_bytes: int = 400_000) -> str:
    try:
        with path.open('rb') as f:
            raw = f.read(max_bytes)
        return raw.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def inspect_binary(path: Path) -> str:
    if path.suffix not in BINARY_META_EXTS:
        return ''
    msg = f'binary:{path.suffix};size={path.stat().st_size}'
    if path.suffix in {'.npy', '.npz'}:
        try:
            import numpy as np
            arr = np.load(path, allow_pickle=True, mmap_mode='r')
            if hasattr(arr, 'shape'):
                msg += f';shape={tuple(arr.shape)};dtype={arr.dtype}'
            elif isinstance(arr, np.lib.npyio.NpzFile):
                msg += f';keys={list(arr.files)[:10]}'
        except Exception:
            pass
    return msg


def axis_hits_from_text(text: str) -> Dict[str, int]:
    norm = normalize_text(text)
    hits: Dict[str, int] = {}
    for axis, aliases in AXIS_ALIASES.items():
        count = 0
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in norm:
                count += norm.count(alias_norm)
        if count:
            hits[axis] = count
    return hits


def score_candidate(path: Path, preview: str, kind: str) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    path_text = normalize_text(str(path))
    text = normalize_text(preview)
    axis_path_hits = axis_hits_from_text(path_text)
    axis_text_hits = axis_hits_from_text(text)

    terms = MODEL_TERMS if kind == 'model' else CALIBRATION_TERMS
    score = 0

    # Base context score.
    if 'frozen_gse235063_model' in path_text:
        score += 18
    if 'derived_secondary_calibration' in path_text:
        score += 18
    if 'comparison_figure' in path_text:
        score += 4

    # Filename/path term hits.
    for term in terms:
        if term in path_text:
            score += 5
    for term in terms:
        if term in text:
            score += 3

    # Axis hits are the key signal.
    score += 8 * len(axis_path_hits)
    score += 12 * len(axis_text_hits)
    score += min(sum(axis_text_hits.values()), 20)

    # Prefer tabular/text files that likely contain usable columns.
    if path.suffix in {'.csv', '.tsv', '.txt', '.json'}:
        score += 3
    if 'gene' in text and kind == 'model':
        score += 4
    if ('coef' in text or 'beta' in text or 'weight' in text or 'loading' in text) and kind == 'model':
        score += 10
    if ('intercept' in text or 'slope' in text) and kind == 'calibration':
        score += 12
    if ('sample_id' in text and 'sample' in path_text):
        score += 2

    return score, axis_path_hits, axis_text_hits


def scan_root(root: Path, kind: str, max_preview_bytes: int) -> List[dict]:
    rows: List[dict] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            preview = ''
            binary_meta = ''
            if suffix in TEXT_EXTS or path.name.endswith('.csv.gz') or path.name.endswith('.tsv.gz'):
                # skip gz body reads; path name still scored
                if not path.name.endswith('.gz'):
                    preview = safe_read_head(path, max_bytes=max_preview_bytes)
            elif suffix in BINARY_META_EXTS:
                binary_meta = inspect_binary(path)

            score, axis_path_hits, axis_text_hits = score_candidate(path, preview + '\n' + binary_meta, kind)
            if score <= 0:
                continue

            rows.append({
                'path': str(path),
                'name': path.name,
                'score': score,
                'size_bytes': path.stat().st_size,
                'axis_path_hits': json.dumps(axis_path_hits, sort_keys=True),
                'axis_text_hits': json.dumps(axis_text_hits, sort_keys=True),
                'is_textlike': bool(preview),
                'binary_meta': binary_meta,
                'preview': preview[:4000].replace('\x00', ' '),
            })
    rows.sort(key=lambda x: (-x['score'], x['path']))
    return rows


def write_csv(rows: List[dict], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with outpath.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'name', 'score'])
        return
    fieldnames = list(rows[0].keys())
    with outpath.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_top_by_axis(rows: List[dict], n: int = 5) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        axis_text_hits = json.loads(row.get('axis_text_hits', '{}'))
        axis_path_hits = json.loads(row.get('axis_path_hits', '{}'))
        axes = set(axis_text_hits) | set(axis_path_hits)
        for axis in axes:
            grouped[axis].append({
                'path': row['path'],
                'name': row['name'],
                'score': row['score'],
                'axis_path_hits': axis_path_hits.get(axis, 0),
                'axis_text_hits': axis_text_hits.get(axis, 0),
            })
    for axis in list(grouped.keys()):
        grouped[axis] = sorted(grouped[axis], key=lambda x: (-x['score'], x['path']))[:n]
    return grouped


def write_report(rows: List[dict], by_axis: Dict[str, List[dict]], outpath: Path, title: str) -> None:
    with outpath.open('w') as f:
        f.write(title + '\n')
        f.write('=' * len(title) + '\n\n')
        f.write('Top candidates overall\n')
        f.write('----------------------\n')
        for item in rows[:25]:
            f.write(f"score={item['score']:>3}  {item['path']}\n")
            if item['axis_path_hits'] != '{}':
                f.write(f"  axis_path_hits: {item['axis_path_hits']}\n")
            if item['axis_text_hits'] != '{}':
                f.write(f"  axis_text_hits: {item['axis_text_hits']}\n")
            if item['binary_meta']:
                f.write(f"  binary_meta: {item['binary_meta']}\n")
            if item['preview']:
                lines = item['preview'].splitlines()[:8]
                for line in lines:
                    f.write(f"  > {line[:220]}\n")
            f.write('\n')

        f.write('\nTop candidates by axis\n')
        f.write('----------------------\n')
        for axis in AXES:
            f.write(f'\n[{axis}]\n')
            matches = by_axis.get(axis, [])
            if not matches:
                f.write('  No explicit hits found.\n')
                continue
            for item in matches:
                f.write(
                    f"  score={item['score']:>3} path_hits={item['axis_path_hits']} text_hits={item['axis_text_hits']}  {item['path']}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description='Targeted finder for malignant-axis transfer artifacts.')
    parser.add_argument('--gse235063-root', required=True)
    parser.add_argument('--gse235923-root', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--max-preview-bytes', type=int, default=400_000)
    args = parser.parse_args()

    gse235063_root = Path(args.gse235063_root)
    gse235923_root = Path(args.gse235923_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_rows = scan_root(gse235063_root, kind='model', max_preview_bytes=args.max_preview_bytes)
    calib_rows = scan_root(gse235923_root, kind='calibration', max_preview_bytes=args.max_preview_bytes)

    model_by_axis = summarize_top_by_axis(model_rows)
    calib_by_axis = summarize_top_by_axis(calib_rows)

    model_csv = outdir / 'gse235063_targeted_model_candidates.csv'
    calib_csv = outdir / 'gse235923_targeted_calibration_candidates.csv'
    model_report = outdir / 'gse235063_targeted_model_report.txt'
    calib_report = outdir / 'gse235923_targeted_calibration_report.txt'
    summary_json = outdir / 'targeted_axis_artifact_summary.json'

    write_csv(model_rows, model_csv)
    write_csv(calib_rows, calib_csv)
    write_report(model_rows, model_by_axis, model_report, 'GSE235063 targeted model candidates')
    write_report(calib_rows, calib_by_axis, calib_report, 'GSE235923 targeted calibration candidates')

    summary = {
        'gse235063_root': str(gse235063_root),
        'gse235923_root': str(gse235923_root),
        'n_model_candidates': len(model_rows),
        'n_calibration_candidates': len(calib_rows),
        'top_model_candidates': [
            {
                'path': x['path'],
                'name': x['name'],
                'score': x['score'],
                'axis_path_hits': json.loads(x['axis_path_hits']),
                'axis_text_hits': json.loads(x['axis_text_hits']),
            }
            for x in model_rows[:12]
        ],
        'top_calibration_candidates': [
            {
                'path': x['path'],
                'name': x['name'],
                'score': x['score'],
                'axis_path_hits': json.loads(x['axis_path_hits']),
                'axis_text_hits': json.loads(x['axis_text_hits']),
            }
            for x in calib_rows[:12]
        ],
        'top_model_by_axis': model_by_axis,
        'top_calibration_by_axis': calib_by_axis,
        'outputs': {
            'gse235063_targeted_model_candidates_csv': str(model_csv),
            'gse235923_targeted_calibration_candidates_csv': str(calib_csv),
            'gse235063_targeted_model_report_txt': str(model_report),
            'gse235923_targeted_calibration_report_txt': str(calib_report),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print('[OK] Targeted axis artifact finder complete.')
    print(f'Output: {outdir}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
