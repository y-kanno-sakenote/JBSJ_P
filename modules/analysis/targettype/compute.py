# modules/analysis/targettype_mod/compute.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools
from typing import List, Tuple
import pandas as pd
import streamlit as st
from .base import split_multi, prefer_title_column
try:
    from modules.common.filters import parse_taxonomy_pairs
except ImportError:
    # Fallback if module path varies
    def parse_taxonomy_pairs(s):
        if not s or pd.isna(s): return []
        res = []
        for p in str(s).split("|"):
            if "::" in p:
                parts = p.split("::", 1)
                res.append((parts[0].strip(), parts[1].strip()))
            else:
                res.append((p.strip(), ""))
        return res

@st.cache_data(ttl=600, show_spinner=False)
def count_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        # Fallback handling
        return pd.Series(dtype=int)
    bags: List[str] = []
    for v in df[col].fillna(""):
        bags += split_multi(v)
    if not bags:
        return pd.Series(dtype=int)
    s = pd.Series(bags)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def count_hierarchy(df: pd.DataFrame, col: str) -> dict:
    """
    戻り値:
    {
       "l1": pd.DataFrame(columns=[L1, count]),
       "l2": pd.DataFrame(columns=[L2, count]),
       "sunburst": pd.DataFrame(columns=[L1, L2, count])
    }
    """
    if col not in df.columns:
        return {}
    
    l1_list, l2_list, pairs_list = [], [], []
    for v in df[col].fillna(""):
        if not v: continue
        pp = parse_taxonomy_pairs(v)
        for val_l1, val_l2 in pp:
            if val_l1:
                l1_list.append(val_l1)
                if val_l2:
                    l2_list.append(val_l2)
                    pairs_list.append((val_l1, val_l2))
                else:
                    # L2がない場合でもサンバースト用に (L1, "") を追加するか、
                    # あるいは L1 only として扱うか。ここでは (L1, "Unspecified") とする手もあるが
                    # シンプルに (L1, L1) または (L1, None) 
                    # Plotly Sunburstで nan を扱うのは面倒なので、L2="" のままにする
                    pairs_list.append((val_l1, ""))
    
    res = {}
    if l1_list:
        res["l1"] = pd.Series(l1_list).value_counts().reset_index()
        res["l1"].columns = ["L1", "count"]
    else:
        res["l1"] = pd.DataFrame(columns=["L1", "count"])
        
    if l2_list:
        res["l2"] = pd.Series(l2_list).value_counts().reset_index()
        res["l2"].columns = ["L2", "count"]
    else:
        res["l2"] = pd.DataFrame(columns=["L2", "count"])

    if pairs_list:
        df_pairs = pd.DataFrame(pairs_list, columns=["L1", "L2"])
        # Group by L1, L2
        sb = df_pairs.groupby(["L1", "L2"]).size().reset_index(name="count")
        res["sunburst"] = sb
    else:
        res["sunburst"] = pd.DataFrame(columns=["L1", "L2", "count"])
        
    return res

@st.cache_data(ttl=600, show_spinner=False)
def cross_counts(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    # Legacy support
    if col_a not in df.columns or col_b not in df.columns:
        return pd.DataFrame(columns=["A", "B", "count"])
    rows = []
    for _, r in df.iterrows():
        As = list(dict.fromkeys(split_multi(r.get(col_a, ""))))
        Bs = list(dict.fromkeys(split_multi(r.get(col_b, ""))))
        for a in As:
            for b in Bs:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def cross_counts_hierarchical(df: pd.DataFrame, col_a: str, level_a: str, col_b: str, level_b: str) -> pd.DataFrame:
    """
    col_a, col_b: タクソナミー列名 (例: target_pairs_top5)
    level_a, level_b: 'L1' or 'L2'
    """
    rows = []
    for _, r in df.iterrows():
        # A側
        val_a = r.get(col_a, "")
        items_a = set()
        for p in parse_taxonomy_pairs(val_a):
            if level_a == "L1" and p[0]: items_a.add(p[0])
            elif level_a == "L2" and p[1]: items_a.add(p[1])
        
        # B側
        val_b = r.get(col_b, "")
        items_b = set()
        for p in parse_taxonomy_pairs(val_b):
            if level_b == "L1" and p[0]: items_b.add(p[0])
            elif level_b == "L2" and p[1]: items_b.add(p[1])
            
        for a in items_a:
            for b in items_b:
                rows.append((a, b))

    if not rows:
        return pd.DataFrame(columns=["A", "B", "count"])
    
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", col, "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        # 旧形式のデータは split_multi で処理
        items = list(dict.fromkeys(split_multi(r.get(col, ""))))
        for it in items:
            rows.append((int(y), it))
    if not rows:
        return pd.DataFrame(columns=["発行年", col, "count"])
    c = pd.DataFrame(rows, columns=["発行年", col]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_counts_hierarchical(df: pd.DataFrame, col: str, level: str) -> pd.DataFrame:
    """
    col: タクソナミー列 (e.g. target_pairs_top5)
    level: 'L1' or 'L2'
    戻り値: DataFrame(columns=["発行年", "item", "count"])
    """
    if "発行年" not in df.columns or col not in df.columns:
        return pd.DataFrame(columns=["発行年", "item", "count"])
    
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        y = int(y)
        
        val = r.get(col, "")
        items = set()
        for p in parse_taxonomy_pairs(val):
            # p = (l1, l2)
            if level == "L1" and p[0]:
                items.add(p[0])
            elif level == "L2" and p[1]:
                items.add(p[1])
        
        for it in items:
            rows.append((y, it))
            
    if not rows:
        return pd.DataFrame(columns=["発行年", "item", "count"])
        
    c = pd.DataFrame(rows, columns=["発行年", "item"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def build_cooccur_edges(df: pd.DataFrame, mode: str, min_edge: int) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        tg = list(dict.fromkeys(split_multi(r.get("対象物_top3", ""))))
        tp = list(dict.fromkeys(split_multi(r.get("研究タイプ_top3", ""))))
        if mode == "対象物のみ":
            items = tg
            pairs = itertools.combinations(sorted(items), 2)
        elif mode == "研究タイプのみ":
            items = tp
            pairs = itertools.combinations(sorted(items), 2)
        else:
            pairs = itertools.product(sorted(set(tg)), sorted(set(tp)))
        for a, b in pairs:
            if a and b and a != b:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

@st.cache_data(ttl=600, show_spinner=False)
def build_cooccur_edges_hierarchical(df: pd.DataFrame, mode: str, min_edge: int) -> pd.DataFrame:
    """
    mode: "Target L2 Only", "Research L2 Only", "Target L2 x Research L2"
    """
    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        # L2 items extraction
        tg_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("target_pairs_top5", "")) if p[1]}
        rp_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("research_pairs_top5", "")) if p[1]}
        
        pairs = []
        if mode == "Target L2 Only":
            pairs = itertools.combinations(sorted(list(tg_l2)), 2)
        elif mode == "Research L2 Only":
            pairs = itertools.combinations(sorted(list(rp_l2)), 2)
        else: # Cross
            pairs = itertools.product(sorted(list(tg_l2)), sorted(list(rp_l2)))
            
        for a, b in pairs:
            if a and b and a != b:
                rows.append((a, b))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
        
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def ordered_index_and_columns(piv: pd.DataFrame, target_order: list[str], type_order: list[str]) -> tuple[list[str], list[str]]:
    cols = list(piv.columns)
    idxs  = list(piv.index)
    cols_order = [x for x in target_order if x in cols] + sorted([x for x in cols if x not in target_order])
    idx_order  = [x for x in type_order   if x in idxs] + sorted([x for x in idxs if x not in type_order])
    return idx_order, cols_order

def example_titles_for_edge(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    title_col = prefer_title_column(df)
    if title_col is None:
        return []
    tg_col = "対象物_top3"; tp_col = "研究タイプ_top3"
    def _tokset(val: str) -> set[str]:
        return set(split_multi(val))
    rows = []
    for _, r in df.iterrows():
        tg = _tokset(r.get(tg_col, "")) if tg_col in df.columns else set()
        tp = _tokset(r.get(tp_col, "")) if tp_col in df.columns else set()
        ok = (a in tg and b in tg) if mode == "対象物のみ" else ((a in tp and b in tp) if mode == "研究タイプのみ" else (a in tg and b in tp))
        if ok:
            rows.append((r.get("発行年", None), str(r.get(title_col, ""))))
    if not rows:
        return []
    try:
        rows = sorted(rows, key=lambda x: (pd.to_numeric(x[0], errors="coerce") if x[0] is not None else -1), reverse=True)
    except Exception:
        pass
    return [t for _, t in rows[:limit]]

def example_titles_for_edge_hierarchical(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    # mode: "Target L2 Only", ...
    title_col = prefer_title_column(df)
    if not title_col: return []
    
    rows = []
    for _, r in df.iterrows():
        tg_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("target_pairs_top5", "")) if p[1]}
        rp_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("research_pairs_top5", "")) if p[1]}
        
        ok = False
        if mode == "Target L2 Only":
            ok = (a in tg_l2 and b in tg_l2)
        elif mode == "Research L2 Only":
            ok = (a in rp_l2 and b in rp_l2)
        else:
            ok = (a in tg_l2 and b in rp_l2)
            
        if ok:
            rows.append((r.get("発行年", None), str(r.get(title_col, ""))))
            
    try:
        rows = sorted(rows, key=lambda x: (pd.to_numeric(x[0], errors="coerce") if x[0] is not None else -1), reverse=True)
    except Exception:
        pass
    return [t for _, t in rows[:limit]]