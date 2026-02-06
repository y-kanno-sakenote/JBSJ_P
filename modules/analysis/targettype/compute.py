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

@st.cache_data(ttl=3600, show_spinner=False)
def count_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty: return pd.Series(dtype=int)
    if col == "対象物_top3" and "_target_pairs" in df.columns:
        s = df["_target_pairs"].apply(lambda x: [p[1] for p in x if p[1]]).explode()
    elif col == "研究分野_top3" and "_research_pairs" in df.columns:
        s = df["_research_pairs"].apply(lambda x: [p[1] for p in x if p[1]]).explode()
    elif col in df.columns:
        bags: List[str] = []
        for v in df[col].fillna(""):
            bags += split_multi(v)
        if not bags: return pd.Series(dtype=int)
        s = pd.Series(bags)
    else:
        return pd.Series(dtype=int)
    
    if s.empty: return pd.Series(dtype=int)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=3600, show_spinner=False)
def count_hierarchy(df: pd.DataFrame, col: str) -> dict:
    if df.empty: return {}
    
    pre_col = "_target_pairs" if col == "target_pairs_top5" else ("_research_pairs" if col == "research_pairs_top5" else None)
    
    if pre_col and pre_col in df.columns:
        pairs_series = df[pre_col].explode().dropna()
        if pairs_series.empty:
            return {
                "l1": pd.DataFrame(columns=["L1", "count"]),
                "l2": pd.DataFrame(columns=["L2", "count"]),
                "sunburst": pd.DataFrame(columns=["L1", "L2", "count"])
            }
        
        df_p = pd.DataFrame(pairs_series.tolist(), columns=["L1", "L2"])
        
        res = {}
        res["l1"] = df_p["L1"].value_counts().reset_index(); res["l1"].columns = ["L1", "count"]
        res["l2"] = df_p[df_p["L2"] != ""]["L2"].value_counts().reset_index(); res["l2"].columns = ["L2", "count"]
        res["sunburst"] = df_p.groupby(["L1", "L2"]).size().reset_index(name="count")
        return res

    # Fallback
    if col not in df.columns: return {}
    l1_list, l2_list, pairs_list = [], [], []
    for v in df[col].fillna(""):
        if not v: continue
        for val_l1, val_l2 in parse_taxonomy_pairs(v):
            if val_l1:
                l1_list.append(val_l1)
                if val_l2:
                    l2_list.append(val_l2); pairs_list.append((val_l1, val_l2))
                else:
                    pairs_list.append((val_l1, ""))
    res = {}
    res["l1"] = pd.Series(l1_list).value_counts().reset_index().rename(columns={"index":"L1", 0:"count"})
    res["l2"] = pd.Series(l2_list).value_counts().reset_index().rename(columns={"index":"L2", 0:"count"})
    res["sunburst"] = pd.DataFrame(pairs_list, columns=["L1", "L2"]).groupby(["L1", "L2"]).size().reset_index(name="count")
    return res

@st.cache_data(ttl=3600, show_spinner=False)
def cross_counts(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["A", "B", "count"])
    
    # Check for pre-computed
    pre_a = "_target_pairs" if col_a == "対象物_top3" else ("_research_pairs" if col_a == "研究タイプ_top3" else None)
    pre_b = "_target_pairs" if col_b == "対象物_top3" else ("_research_pairs" if col_b == "研究タイプ_top3" else None)
    
    if pre_a and pre_b and pre_a in df.columns and pre_b in df.columns:
        def get_l2(pairs): return list(dict.fromkeys([p[1] for p in pairs if p[1]]))
        def get_cross(row):
            As, Bs = get_l2(row[pre_a]), get_l2(row[pre_b])
            return list(itertools.product(As, Bs))
            
        cross_series = df.apply(get_cross, axis=1).explode()
        if cross_series.empty or cross_series.isna().all(): return pd.DataFrame(columns=["A", "B", "count"])
        
        c_df = pd.DataFrame(cross_series.dropna().tolist(), columns=["A", "B"])
        c = c_df.groupby(["A", "B"]).size().reset_index(name="count")
        return c.sort_values("count", ascending=False).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        As = list(dict.fromkeys(split_multi(r.get(col_a, ""))))
        Bs = list(dict.fromkeys(split_multi(r.get(col_b, ""))))
        for a in As:
            for b in Bs: rows.append((a, b))
    if not rows: return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cross_counts_hierarchical(df: pd.DataFrame, col_a: str, level_a: str, col_b: str, level_b: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["A", "B", "count"])
    
    pre_a = "_target_pairs" if col_a == "target_pairs_top5" else ("_research_pairs" if col_a == "research_pairs_top5" else None)
    pre_b = "_target_pairs" if col_b == "target_pairs_top5" else ("_research_pairs" if col_b == "research_pairs_top5" else None)
    
    if pre_a and pre_b and pre_a in df.columns and pre_b in df.columns:
        idx_a = 0 if level_a == "L1" else 1
        idx_b = 0 if level_b == "L1" else 1
        def get_items(pairs, idx): return list(dict.fromkeys([p[idx] for p in pairs if p[idx]]))
        def get_cross(row):
            As, Bs = get_items(row[pre_a], idx_a), get_items(row[pre_b], idx_b)
            return list(itertools.product(As, Bs))
        
        cross_series = df.apply(get_cross, axis=1).explode()
        if cross_series.empty or cross_series.isna().all(): return pd.DataFrame(columns=["A", "B", "count"])
        
        c_df = pd.DataFrame(cross_series.dropna().tolist(), columns=["A", "B"])
        c = c_df.groupby(["A", "B"]).size().reset_index(name="count")
        return c.sort_values("count", ascending=False).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        items_a = {p[0] if level_a=="L1" else p[1] for p in parse_taxonomy_pairs(r.get(col_a,"")) if (p[0] if level_a=="L1" else p[1])}
        items_b = {p[0] if level_b=="L1" else p[1] for p in parse_taxonomy_pairs(r.get(col_b,"")) if (p[0] if level_b=="L1" else p[1])}
        for a in items_a:
            for b in items_b: rows.append((a, b))
    if not rows: return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def yearly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or "発行年" not in df.columns: return pd.DataFrame(columns=["発行年", col, "count"])
    
    pre_col = "_target_pairs" if col == "対象物_top3" else ("_research_pairs" if col == "研究タイプ_top3" else None)
    if pre_col and pre_col in df.columns:
        df_y = df[["発行年", pre_col]].copy()
        df_y["発行年"] = pd.to_numeric(df_y["発行年"], errors="coerce")
        df_y = df_y.dropna(subset=["発行年"])
        df_y[col] = df_y[pre_col].apply(lambda x: list(dict.fromkeys([p[1] for p in x if p[1]])))
        df_y = df_y.explode(col).dropna(subset=[col])
        if df_y.empty: return pd.DataFrame(columns=["発行年", col, "count"])
        c = df_y.groupby(["発行年", col]).size().reset_index(name="count")
        return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        items = list(dict.fromkeys(split_multi(r.get(col, ""))))
        for it in items: rows.append((int(y), it))
    if not rows: return pd.DataFrame(columns=["発行年", col, "count"])
    c = pd.DataFrame(rows, columns=["発行年", col]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def yearly_counts_hierarchical(df: pd.DataFrame, col: str, level: str) -> pd.DataFrame:
    if df.empty or "発行年" not in df.columns: return pd.DataFrame(columns=["発行年", "item", "count"])
    
    pre_col = "_target_pairs" if col == "target_pairs_top5" else ("_research_pairs" if col == "research_pairs_top5" else None)
    if pre_col and pre_col in df.columns:
        df_y = df[["発行年", pre_col]].copy()
        df_y["発行年"] = pd.to_numeric(df_y["発行年"], errors="coerce")
        df_y = df_y.dropna(subset=["発行年"])
        idx = 0 if level=="L1" else 1
        df_y["item"] = df_y[pre_col].apply(lambda x: list(dict.fromkeys([p[idx] for p in x if p[idx]])))
        df_y = df_y.explode("item").dropna(subset=["item"])
        if df_y.empty: return pd.DataFrame(columns=["発行年", "item", "count"])
        c = df_y.groupby(["発行年", "item"]).size().reset_index(name="count")
        return c.sort_values(["発行年", "item"], ascending=[True, False]).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        items = {p[0] if level=="L1" else p[1] for p in parse_taxonomy_pairs(r.get(col,"")) if (p[0] if level=="L1" else p[1])}
        for it in items: rows.append((int(y), it))
    if not rows: return pd.DataFrame(columns=["発行年", "item", "count"])
    c = pd.DataFrame(rows, columns=["発行年", "item"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "item"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def build_cooccur_edges(df: pd.DataFrame, mode: str, min_edge: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["src", "dst", "weight"])
    
    # Check for pre-computed
    if "_target_pairs" in df.columns and "_research_pairs" in df.columns:
        def get_l2(row, col): return sorted(set(p[1] for p in row[col] if p[1]))
        def get_pairs(row):
            tg = get_l2(row, "_target_pairs")
            tp = get_l2(row, "_research_pairs")
            if mode == "対象物のみ": return list(itertools.combinations(tg, 2))
            if mode == "研究分野のみ": return list(itertools.combinations(tp, 2))
            return list(itertools.product(tg, tp))
            
        pairs_series = df.apply(get_pairs, axis=1).explode()
        if pairs_series.empty or pairs_series.isna().all(): return pd.DataFrame(columns=["src", "dst", "weight"])
        
        df_pairs = pd.DataFrame(pairs_series.dropna().tolist(), columns=["src", "dst"])
        edges = df_pairs.groupby(["src", "dst"]).size().reset_index(name="weight")
        edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
        return edges

    # Fallback
    rows = []
    for _, r in df.iterrows():
        tg = list(dict.fromkeys(split_multi(r.get("対象物_top3", ""))))
        tp = list(dict.fromkeys(split_multi(r.get("研究タイプ_top3", ""))))
        pairs = itertools.combinations(sorted(tg), 2) if mode=="対象物のみ" else (itertools.combinations(sorted(tp), 2) if mode=="研究分野のみ" else itertools.product(sorted(set(tg)), sorted(set(tp))))
        for a, b in pairs:
            if a and b and a != b: rows.append((a, b))
    if not rows: return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    return edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def build_cooccur_edges_hierarchical(df: pd.DataFrame, mode: str, min_edge: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["src", "dst", "weight"])
    
    if "_target_pairs" in df.columns and "_research_pairs" in df.columns:
        def get_l2(row, col): return sorted(set(p[1] for p in row[col] if p[1]))
        def get_pairs(row):
            tg = get_l2(row, "_target_pairs")
            tp = get_l2(row, "_research_pairs")
            if mode == "Target L2 Only": return list(itertools.combinations(tg, 2))
            if mode == "Research L2 Only": return list(itertools.combinations(tp, 2))
            return list(itertools.product(tg, tp))
            
        pairs_series = df.apply(get_pairs, axis=1).explode()
        if pairs_series.empty or pairs_series.isna().all(): return pd.DataFrame(columns=["src", "dst", "weight"])
        df_pairs = pd.DataFrame(pairs_series.dropna().tolist(), columns=["src", "dst"])
        edges = df_pairs.groupby(["src", "dst"]).size().reset_index(name="weight")
        return edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        tg_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("target_pairs_top5", "")) if p[1]}
        rp_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("research_pairs_top5", "")) if p[1]}
        pairs = itertools.combinations(sorted(list(tg_l2)), 2) if mode=="Target L2 Only" else (itertools.combinations(sorted(list(rp_l2)), 2) if mode=="Research L2 Only" else itertools.product(sorted(list(tg_l2)), sorted(list(rp_l2))))
        for a, b in pairs:
            if a and b and a != b: rows.append((a, b))
    if not rows: return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    return edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)

def ordered_index_and_columns(piv: pd.DataFrame, target_order: list[str], type_order: list[str]) -> tuple[list[str], list[str]]:
    cols, idxs = list(piv.columns), list(piv.index)
    cols_order = [x for x in target_order if x in cols] + sorted([x for x in cols if x not in target_order])
    idx_order  = [x for x in type_order   if x in idxs] + sorted([x for x in idxs if x not in type_order])
    return idx_order, cols_order

def example_titles_for_edge(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    title_col = prefer_title_column(df)
    if not title_col: return []
    
    if "_target_pairs" in df.columns and "_research_pairs" in df.columns:
        def check(row):
            tg = {p[1] for p in row["_target_pairs"]}; tp = {p[1] for p in row["_research_pairs"]}
            return (a in tg and b in tg) if mode=="対象物のみ" else ((a in tp and b in tp) if mode=="研究分野のみ" else (a in tg and b in tp))
        matches = df[df.apply(check, axis=1)]
    else:
        matches = df # Fallback to slower logic or just empty? Let's use old logic
        def old_check(r):
            tg = set(split_multi(r.get("対象物_top3", ""))); tp = set(split_multi(r.get("研究タイプ_top3", "")))
            return (a in tg and b in tg) if mode=="対象物のみ" else ((a in tp and b in tp) if mode=="研究分野のみ" else (a in tg and b in tp))
        matches = df[df.apply(old_check, axis=1)]

    res = matches[[title_col, "発行年"]].sort_values("発行年", ascending=False).head(limit)
    return res[title_col].tolist()

def example_titles_for_edge_hierarchical(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    title_col = prefer_title_column(df)
    if not title_col: return []
    if "_target_pairs" in df.columns and "_research_pairs" in df.columns:
        def check(row):
            tg = {p[1] for p in row["_target_pairs"]}; tp = {p[1] for p in row["_research_pairs"]}
            return (a in tg and b in tg) if mode=="Target L2 Only" else ((a in tp and b in tp) if mode=="Research L2 Only" else (a in tg and b in tp))
        matches = df[df.apply(check, axis=1)]
    else:
        def old_check(r):
            tg_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("target_pairs_top5", "")) if p[1]}
            rp_l2 = {p[1] for p in parse_taxonomy_pairs(r.get("research_pairs_top5", "")) if p[1]}
            return (a in tg_l2 and b in tg_l2) if mode=="Target L2 Only" else ((a in rp_l2 and b in rp_l2) if mode=="Research L2 Only" else (a in tg_l2 and b in rp_l2))
        matches = df[df.apply(old_check, axis=1)]
        
    res = matches[[title_col, "発行年"]].sort_values("発行年", ascending=False).head(limit)
    return res[title_col].tolist()