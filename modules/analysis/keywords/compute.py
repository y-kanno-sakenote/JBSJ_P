import math
import itertools
from typing import List, Dict
import pandas as pd
import streamlit as st
from .base import split_multi, prefer_title_column
from .stopwords import clean_token

KEY_COLS = ["llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
            "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
            "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10"]



@st.cache_data(ttl=3600, show_spinner=False)
def keyword_freq_by_mode(df: pd.DataFrame, mode: str = "df") -> pd.Series:
    if df.empty: return pd.Series(dtype=int)
    if "_clean_keywords" in df.columns:
        if mode == "df":
            # 同一論文内の重複を除去してからカウント (Document Frequency)
            s = df["_clean_keywords"].apply(lambda x: list(set(x))).explode()
        else:
            # 全ての出現をカウント (Term Frequency)
            s = df["_clean_keywords"].explode()
        if s.empty: return pd.Series(dtype=int)
        return s.value_counts().sort_values(ascending=False)
    
    # Fallback (slow iterrows)
    bags = []
    for _, r in df.iterrows():
        kws = _extract_keywords_from_row(r)
        if mode == "df": kws = list(set(kws))
        bags.extend(kws)
    if not bags: return pd.Series(dtype=int)
    return pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)

@st.cache_data(ttl=3600, show_spinner=False)
def get_global_df(df_all: pd.DataFrame) -> pd.Series:
    """全データベースにおける各単語の Document Frequency を計算"""
    if "_clean_keywords" in df_all.columns:
        return df_all["_clean_keywords"].apply(lambda x: list(set(x))).explode().value_counts()
    
    # Fallback
    counts = {}
    for _, r in df_all.iterrows():
        kws = set(_extract_keywords_from_row(r))
        for k in kws: counts[k] = counts.get(k, 0) + 1
    return pd.Series(counts, dtype=int)

@st.cache_data(ttl=3600, show_spinner=False)
def keyword_tfidf(df_subset: pd.DataFrame, df_all: pd.DataFrame, use_domain_stop: bool = False, power: float = 2.0) -> pd.Series:
    if df_subset.empty: return pd.Series(dtype=float)
    
    tf_series = keyword_freq_by_mode(df_subset, mode="df")
    global_df = get_global_df(df_all)
    n_total = len(df_all)
    domain_stop_threshold = max(20, int(n_total * 0.05))
    
    # Vectorized calculation
    df_scores = tf_series.to_frame(name="tf")
    df_scores["df_global"] = global_df.reindex(df_scores.index).fillna(0)
    
    if use_domain_stop:
        df_scores = df_scores[df_scores["df_global"] <= domain_stop_threshold]
    
    # IDF = (log(N/df) + 1)^power
    df_scores["idf"] = ( (n_total / (df_scores["df_global"] + 1)).apply(math.log) + 1.0 ).pow(power)
    df_scores["score"] = df_scores["tf"] * df_scores["idf"]
    
    return df_scores["score"].sort_values(ascending=False)

@st.cache_data(ttl=3600, show_spinner=False)
def yearly_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    
    if "_clean_keywords" in df.columns:
        # Vectorized year extraction
        df_y = df[["発行年", "_clean_keywords"]].copy()
        df_y["発行年"] = pd.to_numeric(df_y["発行年"], errors="coerce")
        df_y = df_y.dropna(subset=["発行年"])
        df_y["発行年"] = df_y["発行年"].astype(int)
        
        # Unique keywords per row
        df_y["keyword"] = df_y["_clean_keywords"].apply(lambda x: list(set(x)))
        df_y = df_y.explode("keyword").dropna(subset=["keyword"])
        
        if df_y.empty: return pd.DataFrame(columns=["発行年", "keyword", "count"])
        
        c = df_y.groupby(["発行年", "keyword"]).size().reset_index(name="count")
        return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

    # Fallback
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        kws = list(set(_extract_keywords_from_row(r)))
        for k in kws: rows.append((int(y), k))
    if not rows: return pd.DataFrame(columns=["発行年", "keyword", "count"])
    c = pd.DataFrame(rows, columns=["発行年","keyword"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def build_keyword_cooccur_edges(df: pd.DataFrame, min_edge: int) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["src","dst","weight"])
    
    if "_clean_keywords" in df.columns:
        def get_pairs(kws):
            kws = sorted(set(kws))
            return list(itertools.combinations(kws, 2))
        
        pairs_series = df["_clean_keywords"].apply(get_pairs).explode()
        if pairs_series.empty or pairs_series.isna().all():
            return pd.DataFrame(columns=["src","dst","weight"])
            
        df_pairs = pd.DataFrame(pairs_series.dropna().tolist(), columns=["src", "dst"])
        if df_pairs.empty: return pd.DataFrame(columns=["src","dst","weight"])
        
        edges = df_pairs.groupby(["src", "dst"]).size().reset_index(name="weight")
        edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
        return edges

    # Fallback
    rows = []
    for _, r in df.iterrows():
        kws = sorted(set(_extract_keywords_from_row(r)))
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                rows.append((kws[i], kws[j]))
    if not rows: return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    words: List[str] = []
    for c in KEY_COLS:
        val = row.get(c)
        if val and pd.notna(val):
            for w in split_multi(val):
                cw = clean_token(w)
                if cw: words.append(cw)
    return words

def collect_keywords(df: pd.DataFrame) -> pd.Series:
    if "_clean_keywords" in df.columns:
        return df["_clean_keywords"].explode()
    bags: List[str] = []
    for _, r in df.iterrows(): bags += _extract_keywords_from_row(r)
    return pd.Series(bags, dtype="object")