from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

AXIS_TOKENS = [
    "pc1",
    "pc2",
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "log_aux_clp",
    "log_aux_erybaso",
]

MODEL_HINTS = [
    "frozen", "model", "models", "ridge", "coef", "coeff", "coefficient",
    "coefficients", "weight", "weights", "beta", "predictor", "predictors",
    "joblib", "pkl", "pickle", "artifact", "artifacts", "fit", "fitted"
]

CALIBRATION_HINTS = [
    "calibration", "calibrate", "calibrated", "affine", "intercept", "slope",
    "secondary", "gse235923", "transfer", "mapping", "rescale", "rescaled",
    "a_k", "b_k", "raw", "pred", "predicted"
]

TEXTLIKE_EXTS = {".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log", ".md"}
BINARY_EXTS = {".pkl", ".pickle", ".joblib", ".npz", ".npy", ".pt", ".pth"}
ALLOWED_GZ_EXTS = {".txt.gz", ".csv.gz", ".tsv.gz", ".log.gz"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find frozen model and calibration artifacts.")
    p.add_argument(
        "--gse235063-root",
        type=Path,
        default=Path("/GSE235063"),
        help="Root directory for GSE235063 project.",
    )
    p.add_argument(
        "--gse235923-root",
        type=Path,
        default=Path("/GSE235923"),
        help="Root directory for GSE235923 project.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("./transfer_artifact_finder"),
        help="Output directory for candidate reports.",
    )
    p.add_argument(
        "--max-preview-bytes",
        type=int,
        default=1_000_000,
        help="Maximum file size for safe text previewing.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top candidates to retain per category.",
    )
    return p.parse_args()


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def normalized_name(path: Path) -> str:
    return str(path).lower().replace("-", "_")


def count_axis_hits(name: str) -> int:
    return sum(1 for tok in AXIS_TOKENS if tok in name)


def score_path(path: Path, category: str) -> int:
    name = normalized_name(path)
    score = 0

    if category == "model":
        for tok in MODEL_HINTS:
            if tok in name:
                score += 4
        if "gse235063" in name:
            score += 6
        if "frozen_gse235063_model" in name:
            score += 20
        if "train_gene_order" in name:
            score -= 5  # useful, but not a coefficient file
    elif category == "calibration":
        for tok in CALIBRATION_HINTS:
            if tok in name:
                score += 4
        if "gse235923" in name:
            score += 8
        if "secondary" in name:
            score += 4

    score += 3 * count_axis_hits(name)

    suffix = path.suffix.lower()
    suffix2 = "".join(path.suffixes[-2:]).lower() if len(path.suffixes) >= 2 else suffix
    if suffix in BINARY_EXTS:
        score += 6
    if suffix in TEXTLIKE_EXTS or suffix2 in ALLOWED_GZ_EXTS:
        score += 2
    if "derived" in name:
        score += 2
    if "archive" in name or "backup" in name:
        score -= 3

    return score


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune common irrelevant folders
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}
        ]
        for fn in filenames:
            yield Path(dirpath) / fn


def detect_textlike(path: Path) -> bool:
    suffix = path.suffix.lower()
    suffix2 = "".join(path.suffixes[-2:]).lower() if len(path.suffixes) >= 2 else suffix
    return suffix in TEXTLIKE_EXTS or suffix2 in ALLOWED_GZ_EXTS


def is_binary_candidate(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTS


def preview_text_file(path: Path, max_bytes: int = 1_000_000) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": safe_stat_size(path),
        "preview_type": None,
        "columns": None,
        "first_lines": None,
        "json_keys": None,
        "error": None,
    }

    if info["size_bytes"] > max_bytes:
        info["error"] = f"Skipped preview: file larger than {max_bytes} bytes"
        return info

    suffix = path.suffix.lower()
    suffix2 = "".join(path.suffixes[-2:]).lower() if len(path.suffixes) >= 2 else suffix

    try:
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            info["preview_type"] = "json"
            if isinstance(obj, dict):
                info["json_keys"] = list(obj.keys())[:50]
            else:
                info["json_keys"] = [f"type={type(obj).__name__}"]
            return info

        opener = gzip.open if suffix2 in ALLOWED_GZ_EXTS else open
        mode = "rt"
        with opener(path, mode, encoding="utf-8", errors="replace") as fh:
            text = "".join([fh.readline() for _ in range(8)])

        # try tabular parse if looks structured
        stripped = [ln for ln in text.splitlines() if ln.strip()]
        if not stripped:
            info["preview_type"] = "empty_text"
            info["first_lines"] = []
            return info

        sample = "\n".join(stripped[:5])
        dialect = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = None

        if dialect is not None:
            info["preview_type"] = "tabular"
            reader = csv.reader(io.StringIO("\n".join(stripped[:5])), dialect)
            rows = list(reader)
            info["columns"] = rows[0] if rows else []
            info["first_lines"] = [" | ".join(r[:12]) for r in rows[:4]]
        else:
            info["preview_type"] = "text"
            info["first_lines"] = stripped[:5]
        return info
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Preview failed: {exc}"
        return info


def classify_candidates(paths: Iterable[Path], category: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for path in paths:
        name = normalized_name(path)
        score = score_path(path, category)

        if category == "model":
            keep = (
                score >= 6
                or any(tok in name for tok in MODEL_HINTS)
                or count_axis_hits(name) > 0
            )
        else:
            keep = (
                score >= 6
                or any(tok in name for tok in CALIBRATION_HINTS)
                or ("gse235923" in name and detect_textlike(path))
            )

        if not keep:
            continue

        rec = {
            "path": str(path),
            "name": path.name,
            "score": score,
            "size_bytes": safe_stat_size(path),
            "is_textlike": detect_textlike(path),
            "is_binary_candidate": is_binary_candidate(path),
            "axis_hits": count_axis_hits(name),
        }
        candidates.append(rec)

    candidates.sort(key=lambda r: (-r["score"], r["path"]))
    return candidates


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("path,name,score,size_bytes,is_textlike,is_binary_candidate,axis_hits\n")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_candidates(candidates: List[Dict[str, Any]], max_preview_bytes: int) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for rec in candidates:
        item = dict(rec)
        path = Path(rec["path"])
        if rec["is_textlike"]:
            item["preview"] = preview_text_file(path, max_bytes=max_preview_bytes)
        else:
            item["preview"] = None
        summarized.append(item)
    return summarized


def write_human_report(path: Path, header: str, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        fh.write("=" * len(header) + "\n\n")
        if not rows:
            fh.write("No candidates found.\n")
            return
        for i, rec in enumerate(rows, start=1):
            fh.write(f"[{i}] score={rec['score']}  path={rec['path']}\n")
            fh.write(f"    size_bytes={rec['size_bytes']}  textlike={rec['is_textlike']}  binary={rec['is_binary_candidate']}  axis_hits={rec['axis_hits']}\n")
            prev = rec.get("preview")
            if prev:
                if prev.get("preview_type"):
                    fh.write(f"    preview_type={prev['preview_type']}\n")
                if prev.get("columns"):
                    fh.write(f"    columns={prev['columns'][:20]}\n")
                if prev.get("json_keys"):
                    fh.write(f"    json_keys={prev['json_keys'][:20]}\n")
                if prev.get("first_lines"):
                    fh.write("    first_lines:\n")
                    for ln in prev["first_lines"][:5]:
                        fh.write(f"      {ln}\n")
                if prev.get("error"):
                    fh.write(f"    preview_error={prev['error']}\n")
            fh.write("\n")


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    roots = {
        "gse235063": args.gse235063_root,
        "gse235923": args.gse235923_root,
    }

    all_files: Dict[str, List[Path]] = {name: list(iter_files(root)) for name, root in roots.items()}

    model_candidates = classify_candidates(all_files["gse235063"], category="model")[: args.top_n]
    calibration_candidates = classify_candidates(all_files["gse235923"], category="calibration")[: args.top_n]

    model_detailed = summarize_candidates(model_candidates, args.max_preview_bytes)
    calib_detailed = summarize_candidates(calibration_candidates, args.max_preview_bytes)

    write_csv(outdir / "candidate_frozen_model_files.csv", model_candidates)
    write_csv(outdir / "candidate_gse235923_calibration_files.csv", calibration_candidates)

    write_human_report(
        outdir / "candidate_frozen_model_report.txt",
        "Candidate frozen model files (GSE235063)",
        model_detailed,
    )
    write_human_report(
        outdir / "candidate_gse235923_calibration_report.txt",
        "Candidate GSE235923 calibration files",
        calib_detailed,
    )

    manifest = {
        "roots": {k: str(v) for k, v in roots.items()},
        "n_files_scanned": {k: len(v) for k, v in all_files.items()},
        "top_model_candidates": model_candidates[:10],
        "top_calibration_candidates": calibration_candidates[:10],
        "outputs": {
            "candidate_frozen_model_files_csv": str(outdir / "candidate_frozen_model_files.csv"),
            "candidate_gse235923_calibration_files_csv": str(outdir / "candidate_gse235923_calibration_files.csv"),
            "candidate_frozen_model_report_txt": str(outdir / "candidate_frozen_model_report.txt"),
            "candidate_gse235923_calibration_report_txt": str(outdir / "candidate_gse235923_calibration_report.txt"),
        },
    }

    with open(outdir / "transfer_artifact_finder_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("[OK] Transfer artifact finder complete.")
    print(f"Output: {outdir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
