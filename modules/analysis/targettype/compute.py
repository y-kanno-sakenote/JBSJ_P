# modules/analysis/targettype_mod/compute.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools
from typing import List, Tuple
import pandas as pd
import streamlit as st
from .base import split_multi, prefer_title_column

try:
    from scipy.stats import chi2_contingency
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from modules.common.filters import parse_taxonomy_pairs
except ImportError:
    # Fallback if module path varies
    def parse_taxonomy_pairs(s):
        if not s or pd.isna(s): return []
        res = []
        pairs = str(s).split("|")
        for p in pairs:
            p = p.strip()
            if not p: continue
            if "::" in p:
                parts = p.split("::", 1)
                res.append((parts[0].strip(), parts[1].strip()))
            else:
                prefix = p.split(".")[0]
                if "-" in prefix:
                    res.append(("", p))
                else:
                    res.append((p, ""))
        return res

def count_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty: return pd.Series(dtype=int)
    is_tax = col in ["対象物_top3", "研究分野_top3", "研究タイプ_top3"]
    if is_tax and "_assigned_pairs_list" in df.columns:
        if col in ["研究分野_top3", "研究タイプ_top3"]:
            # L1 を抽出
            s = df["_assigned_pairs_list"].apply(lambda x: [p[0] for p in x if p[0]]).explode()
        else:
            # L2 (対象物_top3 等)
            s = df["_assigned_pairs_list"].apply(lambda x: [p[1] for p in x if p[1]]).explode()
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
    
    pre_col = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    
    if pre_col:
        pairs_series = df[pre_col].explode().dropna()
        if pairs_series.empty:
            return {
                "l1": pd.DataFrame(columns=["L1", "count"]),
                "l2": pd.DataFrame(columns=["L2", "count"]),
                "sunburst": pd.DataFrame(columns=["L1", "L2", "count"])
            }
        
        # pairs_series は (L1, L2) のタプル
        df_p = pd.DataFrame(pairs_series.tolist(), columns=["L1", "L2"])
        
        res = {}
        res["l1"] = df_p[df_p["L1"] != ""]["L1"].value_counts().reset_index(); res["l1"].columns = ["L1", "count"]
        res["l2"] = df_p[df_p["L2"] != ""]["L2"].value_counts().reset_index(); res["l2"].columns = ["L2", "count"]
        # サンバースト用: L1・L2 ともに空でない行のみ使用（空行が白余白の原因になる）
        df_sb = df_p[(df_p["L1"] != "") & (df_p["L2"] != "")]
        res["sunburst"] = df_sb.groupby(["L1", "L2"]).size().reset_index(name="count")
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
    res["l1"] = pd.Series([x for x in l1_list if x]).value_counts().reset_index().rename(columns={"index":"L1", 0:"count"})
    res["l2"] = pd.Series([x for x in l2_list if x]).value_counts().reset_index().rename(columns={"index":"L2", 0:"count"})
    # サンバースト用: L1・L2 ともに存在するペアのみ
    valid_pairs = [(l1, l2) for l1, l2 in pairs_list if l1 and l2]
    res["sunburst"] = pd.DataFrame(valid_pairs, columns=["L1", "L2"]).groupby(["L1", "L2"]).size().reset_index(name="count") if valid_pairs else pd.DataFrame(columns=["L1", "L2", "count"])
    return res

@st.cache_data(ttl=3600, show_spinner=False)
def cross_counts(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["A", "B", "count"])
    
    # Check for pre-computed
    is_tax = col_a in ["対象物_top3", "研究分野_top3"] or col_b in ["対象物_top3", "研究分野_top3"]
    pre = "_assigned_pairs_list" if ("_assigned_pairs_list" in df.columns and is_tax) else None
    
    if pre:
        def get_items(pairs, idx): return list(dict.fromkeys([p[idx] for p in pairs if p[idx]]))
        def get_cross(row):
            # L1 vs L2 の組み合わせ
            l1s = get_items(row[pre], 0)
            l2s = get_items(row[pre], 1)
            return list(itertools.product(l1s, l2s))
            
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
    
    pre = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    
    if pre:
        idx_a = 0 if level_a == "L1" else 1
        idx_b = 0 if level_b == "L1" else 1
        def get_items(pairs, idx): return list(dict.fromkeys([p[idx] for p in pairs if p[idx]]))
        def get_cross(row):
            # 同じリスト内の組み合わせ
            items_a = get_items(row[pre], idx_a)
            items_b = get_items(row[pre], idx_b)
            if idx_a == idx_b:
                return list(itertools.combinations(items_a, 2))
            else:
                return list(itertools.product(items_a, items_b))
        
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
        if level_a == level_b and col_a == col_b:
            for a, b in itertools.combinations(sorted(list(items_a)), 2):
                rows.append((a, b))
        else:
            for a in items_a:
                for b in items_b: rows.append((a, b))
    if not rows: return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def yearly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or "発行年" not in df.columns: return pd.DataFrame(columns=["発行年", col, "count"])
    
    is_tax = col in ["対象物_top3", "研究分野_top3", "研究タイプ_top3"]
    pre_col = "_assigned_pairs_list" if ("_assigned_pairs_list" in df.columns and is_tax) else None
    if pre_col:
        df_y = df[["発行年", pre_col]].copy()
        df_y["発行年"] = pd.to_numeric(df_y["発行年"], errors="coerce")
        df_y = df_y.dropna(subset=["発行年"])
        # L1かL2かを判定 (旧研究分野/タイプならL1, それ以外ならL2)
        idx = 0 if col in ["研究分野_top3", "研究タイプ_top3"] else 1
        df_y["item"] = df_y[pre_col].apply(lambda x: list(dict.fromkeys([p[idx] for p in x if p[idx]])))
        df_y = df_y.explode("item").dropna(subset=["item"])
        if df_y.empty: return pd.DataFrame(columns=["発行年", "item", "count"])
        c = df_y.groupby(["発行年", "item"]).size().reset_index(name="count")
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
    
    pre_col = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    if pre_col:
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
    pre = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    if pre:
        def get_l2(row): return sorted(set(p[1] for p in row[pre] if p[1]))
        def get_pairs(row):
            l2 = get_l2(row)
            if mode == "対象物のみ": return list(itertools.combinations(l2, 2))
            if mode == "研究分野のみ": return list(itertools.combinations(l2, 2))
            return list(itertools.combinations(l2, 2)) # 全部 L2 同士
            
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
    
    pre = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    
    if pre:
        def get_items(row, idx): return sorted(set(p[idx] for p in row[pre] if p[idx]))
        def get_pairs(row):
            l1 = get_items(row, 0)
            l2 = get_items(row, 1)
            if mode == "L2 Only": return list(itertools.combinations(l2, 2))
            if mode == "L1 Only": return list(itertools.combinations(l1, 2))
            return list(itertools.product(l1, l2))
            
        pairs_series = df.apply(get_pairs, axis=1).explode()
        if pairs_series.empty or pairs_series.isna().all(): return pd.DataFrame(columns=["src", "dst", "weight"])
        df_pairs = pd.DataFrame(pairs_series.dropna().tolist(), columns=["src", "dst"])
        edges = df_pairs.groupby(["src", "dst"]).size().reset_index(name="weight")
        return edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)

    # Fallback
    rows = []
    col = "assigned_pairs" if "assigned_pairs" in df.columns else "target_pairs_top5"
    for _, r in df.iterrows():
        pairs = parse_taxonomy_pairs(r.get(col, ""))
        l1 = {p[0] for p in pairs if p[0]}
        l2 = {p[1] for p in pairs if p[1]}
        if mode == "L2 Only":
            comb = itertools.combinations(sorted(list(l2)), 2)
        elif mode == "L1 Only":
            comb = itertools.combinations(sorted(list(l1)), 2)
        else:
            comb = itertools.product(sorted(list(l1)), sorted(list(l2)))
        for a, b in comb:
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
    
    pre = "_assigned_pairs_list" if "_assigned_pairs_list" in df.columns else None
    if pre:
        def check(row):
            l2 = {p[1] for p in row[pre]}
            return (a in l2 and b in l2) # 基本 L2 同士を想定
        matches = df[df.apply(check, axis=1)]
    elif "_target_pairs" in df.columns and "_research_pairs" in df.columns:
        def check(row):
            tg = {p[1] for p in row["_target_pairs"]}; tp = {p[1] for p in row["_research_pairs"]}
            return (a in tg and b in tg) if mode=="対象物のみ" else ((a in tp and b in tp) if mode=="研究分野のみ" else (a in tg and b in tp))
        matches = df[df.apply(check, axis=1)]
    else:
        matches = df # Fallback to slower logic or just empty? Let's use old logic
        col = "assigned_pairs" if "assigned_pairs" in df.columns else "対象物_top3"
        def old_check(r):
            items = set(p[1] for p in parse_taxonomy_pairs(r.get(col, ""))) if col == "assigned_pairs" else set(split_multi(r.get(col, "")))
            return (a in items and b in items)
        matches = df[df.apply(old_check, axis=1)]

    res = matches[[title_col, "発行年"]].sort_values("発行年", ascending=False).head(limit)
    return res[title_col].astype(str).tolist()

def example_titles_for_edge_hierarchical(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    title_col = prefer_title_column(df)
    if not title_col: return []
    if "_assigned_pairs_list" in df.columns:
        def check(row):
            l1 = {p[0] for p in row["_assigned_pairs_list"]}
            l2 = {p[1] for p in row["_assigned_pairs_list"]}
            if mode == "L2 Only": return (a in l2 and b in l2)
            if mode == "L1 Only": return (a in l1 and b in l1)
            return (a in l1 and b in l2)
        matches = df[df.apply(check, axis=1)]
    else:
        def old_check(r):
            pairs = parse_taxonomy_pairs(r.get("assigned_pairs", ""))
            l1 = {p[0] for p in pairs}; l2 = {p[1] for p in pairs}
            if mode == "L2 Only": return (a in l2 and b in l2)
            if mode == "L1 Only": return (a in l1 and b in l1)
            return (a in l1 and b in l2)
        matches = df[df.apply(old_check, axis=1)]
        
    res = matches[[title_col, "発行年"]].sort_values("発行年", ascending=False).head(limit)
    return res[title_col].tolist()

@st.cache_data(ttl=3600, show_spinner=False)
def compute_chi2_and_residuals(df: pd.DataFrame, periods: list[tuple[str, int, int]], target_col: str, level: str = "L1") -> dict:
    if df.empty or "発行年" not in df.columns or not HAS_SCIPY:
        return {}

    # 1. データの準備
    df_work = df.copy()
    df_work["発行年_num"] = pd.to_numeric(df_work["発行年"], errors="coerce")
    df_work = df_work.dropna(subset=["発行年_num"])
    
    def get_period(y):
        for label, start, end in periods:
            if start <= y <= end:
                return label
        return None
        
    df_work["period"] = df_work["発行年_num"].apply(get_period)
    df_work = df_work.dropna(subset=["period"])
    
    if df_work.empty:
        return {}
        
    if target_col == "assigned_pairs" and "_assigned_pairs_list" in df_work.columns:
        idx = 0 if level == "L1" else 1
        df_y = df_work[["period", "_assigned_pairs_list"]].copy()
        df_y["item"] = df_y["_assigned_pairs_list"].apply(lambda x: list(dict.fromkeys([p[idx] for p in x if p[idx]])))
        df_y = df_y.explode("item").dropna(subset=["item"])
    else:
        df_y = df_work[["period", target_col]].copy()
        df_y["item"] = df_y[target_col].apply(lambda x: list(dict.fromkeys(split_multi(str(x)))) if pd.notna(x) else [])
        df_y = df_y.explode("item").dropna(subset=["item"])
        
    if df_y.empty:
        return {}

    # 2. クロス集計
    period_labels = [p[0] for p in periods]
    df_y["period"] = pd.Categorical(df_y["period"], categories=period_labels, ordered=True)
    crosstab = pd.crosstab(df_y["period"], df_y["item"]).fillna(0).astype(int)
    
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return {"error": "比較できる期間またはカテゴリが不足しています。"}

    # 3. カイ二乗検定と残差計算
    try:
        chi2, p_val, dof, expected = chi2_contingency(crosstab)
        
        row_totals = crosstab.sum(axis=1).values
        col_totals = crosstab.sum(axis=0).values
        total = crosstab.values.sum()
        
        adj_residuals = pd.DataFrame(index=crosstab.index, columns=crosstab.columns, dtype=float)
        
        for i in range(crosstab.shape[0]):
            for j in range(crosstab.shape[1]):
                o = crosstab.iloc[i, j]
                e = expected[i, j]
                if e == 0 or total == 0:
                    adj_residuals.iloc[i, j] = 0.0
                else:
                    v = e * (1.0 - row_totals[i]/total) * (1.0 - col_totals[j]/total)
                    if v <= 0:
                        adj_residuals.iloc[i, j] = 0.0
                    else:
                        adj_residuals.iloc[i, j] = (o - e) / (v ** 0.5)

        return {
            "crosstab": crosstab,
            "expected": pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns),
            "adj_residuals": adj_residuals,
            "chi2": float(chi2),
            "p_value": float(p_val),
            "dof": int(dof)
        }
    except Exception as e:
        return {"error": f"検定エラー: {e}"}