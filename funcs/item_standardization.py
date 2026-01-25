"""Item standardization (fuzzy clustering) and JSON map creation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower()
    s = re.sub(r"[\/\\,;•·•]", " ", s)
    s = re.sub(r"[^\w\s\(\)\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_sort_key(s: str) -> str:
    tokens = re.findall(r"\w+", s)
    tokens.sort()
    return " ".join(tokens)

def fuzzy_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a_tok = token_sort_key(a)
    b_tok = token_sort_key(b)
    score_tok = SequenceMatcher(None, a_tok, b_tok).ratio()
    score_raw = SequenceMatcher(None, a, b).ratio()
    return 0.65 * score_tok + 0.35 * score_raw

def choose_representative_original(originals: List[str]) -> str:
    counter = Counter(originals)
    most_common = counter.most_common()
    rep = most_common[0][0]
    tied = [k for k, v in counter.items() if v == most_common[0][1]]
    if len(tied) > 1:
        rep = max(tied, key=lambda x: len(x))

    def title_keep_paren(text: str) -> str:
        m = re.match(r"^(.*?)(\s*\(.*\))?$", text.strip())
        if not m:
            return text.title()
        left, paren = m.group(1), m.group(2) or ""
        return left.title().strip() + (" " + paren if paren else "")
    return title_keep_paren(rep)

def make_items_auto_std(
    df: pd.DataFrame,
    output_file: str | Path,
    item_col: str,
    patient_col: str,
    similarity_threshold: float = 0.82,
    min_cluster_size: int = 1,
    index_base: int = 0,
) -> Dict[str, Dict[str, str]]:
    """Greedy clustering of item strings and save mapping to JSON.

    The saved JSON maps standardized item name -> { "<row_index>": "<patient_id>" }
    Row indices are based on df.index + index_base.
    """
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found.")
    if patient_col not in df.columns:
        raise ValueError(f"Column '{patient_col}' not found.")

    records: List[Tuple[int, str, str, str]] = []
    for idx, (val, pat) in zip(df.index, zip(df[item_col], df[patient_col])):
        orig = "" if pd.isna(val) else str(val).strip()
        norm = normalize_text(orig)
        pat_str = "" if pd.isna(pat) else str(pat).strip()
        records.append((int(idx), orig, norm, pat_str))

    n = len(records)
    assigned = [False] * n
    clusters: List[List[Tuple[int, str, str, str]]] = []

    for i in range(n):
        if assigned[i]:
            continue
        idx_i, orig_i, norm_i, pat_i = records[i]
        cluster = [(idx_i, orig_i, norm_i, pat_i)]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            idx_j, orig_j, norm_j, pat_j = records[j]
            sim = fuzzy_similarity(norm_i, norm_j)
            contains = False
            if norm_i and norm_j:
                contains = (norm_i in norm_j) or (norm_j in norm_i)
            if sim >= similarity_threshold or (contains and sim >= (similarity_threshold * 0.75)):
                cluster.append((idx_j, orig_j, norm_j, pat_j))
                assigned[j] = True
        clusters.append(cluster)

    result: Dict[str, Dict[str, str]] = {}
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        originals = [c[1] if c[1] != "" else "(blank)" for c in cluster]
        rep = choose_representative_original(originals)

        key = rep
        suffix = 1
        while key in result:
            key = f"{rep} ({suffix})"
            suffix += 1

        group_dict: Dict[str, str] = {}
        for idx, _, __, patient in cluster:
            group_dict[str(idx + index_base)] = patient
        result[key] = group_dict

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved item standardization map: %s", out_path)
    return result
