# modules/analysis/coauthor/compute.py
from __future__ import annotations
import itertools
import pandas as pd
import streamlit as st

# Optional deps
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

from .filters_adapter import split_authors

@st.cache_data(ttl=3600, show_spinner=False)
def author_total_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "著者" not in df.columns: return pd.Series(dtype=int)
    
    if "_split_authors" in df.columns:
        s = df["_split_authors"].explode()
        if s.empty: return pd.Series(dtype=int)
        return s.value_counts().sort_values(ascending=False)
    
    # Fallback
    bags = []
    for a in df["著者"].fillna(""):
        bags += list(dict.fromkeys(split_authors(a)))
    if not bags: return pd.Series(dtype=int)
    return pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)

@st.cache_data(ttl=3600, show_spinner=False)
def yearly_author_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "著者" not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "著者", "count"])
    
    if "_split_authors" in df.columns:
        df_y = df[["発行年", "_split_authors"]].copy()
        df_y["発行年"] = pd.to_numeric(df_y["発行年"], errors="coerce")
        df_y = df_y.dropna(subset=["発行年"])
        df_y["著者"] = df_y["_split_authors"]
        df_y = df_y.explode("著者").dropna(subset=["著者"])
        if df_y.empty: return pd.DataFrame(columns=["発行年", "著者", "count"])
        c = df_y.groupby(["発行年", "著者"]).size().reset_index(name="count")
        return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        names = list(dict.fromkeys(split_authors(r.get("著者", ""))))
        for n in names: rows.append((int(y), n))
    if not rows: return pd.DataFrame(columns=["発行年", "著者", "count"])
    c = pd.DataFrame(rows, columns=["発行年","著者"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["src", "dst", "weight"])
    
    if "_split_authors" in df.columns:
        def get_pairs(authors):
            names = sorted(set(authors))
            return list(itertools.combinations(names, 2))
            
        pairs_series = df["_split_authors"].apply(get_pairs).explode()
        if pairs_series.empty or pairs_series.isna().all():
            return pd.DataFrame(columns=["src", "dst", "weight"])
            
        df_pairs = pd.DataFrame(pairs_series.dropna().tolist(), columns=["src", "dst"])
        if df_pairs.empty: return pd.DataFrame(columns=["src", "dst", "weight"])
        
        edges = df_pairs.groupby(["src", "dst"]).size().reset_index(name="weight")
        return edges.sort_values("weight", ascending=False).reset_index(drop=True)

    # Fallback
    rows = []
    for a in df.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(split_authors(a)))
        for s, t in itertools.combinations(names, 2): rows.append((s, t))
    if not rows: return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    return edges.sort_values("weight", ascending=False).reset_index(drop=True)

def centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    if edges.empty: return pd.DataFrame(columns=["著者", "共著数", "つながりスコア"])
    
    deg_simple = pd.concat([
        edges.groupby("src")["weight"].sum(),
        edges.groupby("dst")["weight"].sum(),
    ], axis=1).fillna(0)
    deg_simple["coauth_count"] = deg_simple.sum(axis=1)
    deg_simple = deg_simple["coauth_count"].reset_index().rename(columns={"index": "著者", "coauth_count": "共著数"})

    if not HAS_NX:
        out = deg_simple.copy()
        out["つながりスコア"] = out["共著数"]
        return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)

    import networkx as nx
    G = nx.Graph()
    # Vectorized edge adding might be harder, but this loop is on edges (fewer than rows)
    for r in edges.itertuples():
        G.add_edge(str(r.src), str(r.dst), weight=float(r.weight))

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try: cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception: cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    cen_df = pd.Series(cen, name="つながりスコア").reset_index().rename(columns={"index": "著者"})
    out = pd.merge(cen_df, deg_simple, on="著者", how="left")
    out["共著数"] = out["共著数"].fillna(0).round().astype(int)
    return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)