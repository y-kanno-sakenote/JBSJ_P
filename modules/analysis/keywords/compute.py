import math
from typing import List, Dict
import pandas as pd
import streamlit as st
from .base import split_multi
from .stopwords import clean_token

KEY_COLS = ["llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
            "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
            "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10"]

def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    words: List[str] = []
    for c in KEY_COLS:
        if c in row and pd.notna(row[c]):
            for w in split_multi(row[c]):
                cw = clean_token(w)
                if cw: words.append(cw)
    return words

@st.cache_data(ttl=600, show_spinner=False)
def collect_keywords(df: pd.DataFrame) -> pd.Series:
    bags: List[str] = []
    for _, r in df.iterrows(): bags += _extract_keywords_from_row(r)
    return pd.Series(bags, dtype="object")

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq(df: pd.DataFrame) -> pd.Series:
    s = collect_keywords(df)
    if s.empty: return pd.Series(dtype=int)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq_by_mode(df: pd.DataFrame, mode: str = "df") -> pd.Series:
    if mode == "df":
        bags: list[str] = []
        for _, r in df.iterrows():
            kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
            bags.extend(kws)
        if not bags: return pd.Series(dtype=int)
        return pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)
    return keyword_freq(df)

@st.cache_data(ttl=3600, show_spinner=False)
def get_global_df(df_all: pd.DataFrame) -> Dict[str, int]:
    """全データベースにおける各単語の Document Frequency (出現論文数) を計算"""
    counts: Dict[str, int] = {}
    for _, r in df_all.iterrows():
        # 同一論文内の重複を除いて集計
        kws = set(_extract_keywords_from_row(r))
        for k in kws:
            counts[k] = counts.get(k, 0) + 1
    return counts

@st.cache_data(ttl=600, show_spinner=False)
def keyword_tfidf(df_subset: pd.DataFrame, df_all: pd.DataFrame, use_domain_stop: bool = False, power: float = 2.0) -> pd.Series:
    """
    TF-IDF に基づくキーワードの特徴度を計算する。
    TF: subset 内での出現論文数 (DF 方式を採用)
    IDF: (log( 全論文数 / (全論文での出現数 + 1) ) + 1) ^ power
    
    use_domain_stop=True の場合、全体で非常に高頻度な語(例: DF > 5%)を抑制する。
    """
    if df_subset.empty: return pd.Series(dtype=float)
    
    # Subset の統計 (TF)
    tf_series = keyword_freq_by_mode(df_subset, mode="df")
    
    # 全文書の統計 (IDF用)
    global_df = get_global_df(df_all)
    n_total = len(df_all)
    
    # ドメインストップワードの閾値 (全体論文の 5% 以上に出る語)
    domain_stop_threshold = max(20, int(n_total * 0.05))
    
    scores = {}
    for word, tf in tf_series.items():
        df_global = global_df.get(word, 0)
        
        # ドメイン固有ストップワードの処理
        if use_domain_stop and df_global > domain_stop_threshold:
            continue
            
        # IDF の計算 (power で強調)
        # log(N/df) は、df が N に近いほど 0 に近づく。
        idf_base = math.log(n_total / (df_global + 1))
        # 0.0 になると消えてしまうので +1.0 して底上げしつつ、power で累乗
        idf = math.pow(idf_base + 1.0, power)
        
        scores[word] = tf * idf
        
    res = pd.Series(scores, dtype=float).sort_values(ascending=False)
    return res

@st.cache_data(ttl=600, show_spinner=False)
def yearly_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
        for k in kws: rows.append((int(y), k))
    if not rows: return pd.DataFrame(columns=["発行年", "keyword", "count"])
    c = pd.DataFrame(rows, columns=["発行年","keyword"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def build_keyword_cooccur_edges(df: pd.DataFrame, min_edge: int) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        kws = sorted(set(_extract_keywords_from_row(r)))
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                rows.append((kws[i], kws[j]))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges