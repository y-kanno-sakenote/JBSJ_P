# modules/common/filters.py
# -*- coding: utf-8 -*-
"""
共通フィルターバー（年 / 対象物 / 研究分野）
- 単体 DataFrame を返す（呼び出し側の後方互換重視）
"""


import re
import pandas as pd
import streamlit as st

# ========= 並び順（temporal.py と統一） & 補助ソート関数 =========
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス",
    "酵母・微生物","アミノ酸・タンパク質","その他"
]

TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究分野）"
]

GENRE_ORDER = [
    "清酒", "ビール", "ワイン（果実酒）", "焼酎・泡盛", "蒸留酒（その他）",
    "混成酒", "発酵調味料", "その他食品", "未特定"
]

def order_options(all_options: list[str], preferred: list[str]) -> list[str]:
    s = set(all_options)
    head = [x for x in preferred if x in s]
    tail = sorted([x for x in all_options if x not in preferred])
    return head + tail

# ========= ヘルパー関数 =========
def get_taxonomy_hierarchy(series):
    """L1::L2|L1::L2 形式の列から L1 一覧と L1->L2 マッピングを返す"""
    l1_set = set()
    l1_to_l2 = {}
    for entry in series.fillna(""):
        if not entry: continue
        for l1, l2 in parse_taxonomy_pairs(entry):
            if l1:
                l1_set.add(l1)
                if l1 not in l1_to_l2: l1_to_l2[l1] = set()
                if l2: l1_to_l2[l1].add(l2)
            elif l2:
                # L1が空でL2がある場合は、孤立したL2として扱うか無視するかだが
                # ここでは L1 リストに含めないことで、ユーザーの「L1にL2を混ぜない」要望に応える
                pass
    return sorted(l1_set), {k: sorted(v) for k, v in l1_to_l2.items()}

def parse_taxonomy_pairs(s):
    """L1::L2|L1::L2 形式の文字列を [(L1, L2), ...] のリストにパースする"""
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
            # :: がない場合、ハイフンの有無でL1/L2を簡易判定
            # L1: "1. " など, L2: "1-1. " など
            prefix = p.split(".")[0]
            if "-" in prefix:
                res.append(("", p)) # L2として扱う
            else:
                res.append((p, "")) # L1として扱う
    return res

def apply_hierarchical_filters(df, genre_sel=None, l1_sel=None, l2_sel=None):
    df2 = df.copy()
    
    # 引数の正規化（空文字やNoneを除外）
    def _norm_sel(s):
        if s is None: return []
        return [str(x).strip() for x in s if x and str(x).strip()]
    
    genre_sel = _norm_sel(genre_sel)
    l1_sel = _norm_sel(l1_sel)
    l2_sel = _norm_sel(l2_sel)

    if genre_sel and "product_L0_top3" in df2.columns:
        df2 = df2[df2["product_L0_top3"].apply(lambda v: any(g in [x.strip() for x in str(v).split("|")] for g in genre_sel))]

    def hit_hierarchy(val, l1_sel, l2_sel):
        if not l1_sel and not l2_sel: return True
        pairs = parse_taxonomy_pairs(val)
        if not pairs: return False
        
        l1_f = l1_sel
        l2_f = l2_sel
        
        for p1, p2 in pairs:
            m1 = (p1 in l1_f) if l1_f else True
            m2 = (p2 in l2_f) if l2_f else True
            if m1 and m2: return True
        return False

    if (l1_sel or l2_sel) and "assigned_pairs" in df2.columns:
        df2 = df2[df2["assigned_pairs"].apply(lambda v: hit_hierarchy(v, l1_sel, l2_sel))]
    
    return df2

# ---- メインUI -----------------------------------------------------------
def render_filter_bar(df,
                      key_prefix="flt",
                      target_order=None,
                      type_order=None,
                      show_caption=False,
                      show_reset=False):
    """
    更新版共通フィルターUI。
    """
    if df is None or df.empty:
        st.info("フィルター対象データがありません。")
        return df

    # 1. 基本（年）
    y = pd.to_numeric(df.get("発行年", pd.Series(dtype=float)), errors="coerce")
    ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)

    if show_caption:
        st.caption("年・ジャンル・研究分野・専門領域で絞り込みできます。")

    # 新しいタクソナミー列がある場合は階層型フィルタを表示
    has_wider = all(c in df.columns for c in ["product_L0_top3", "assigned_pairs"])

    genre_sel, l1_sel, l2_sel = [], [], []

    if has_wider:
        # 1. 基本（年・ジャンル）
        row1_y, row1_g = st.columns([1, 1])
        with row1_y:
            y_val = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key=f"{key_prefix}_year")
            y_from, y_to = y_val
        with row1_g:
            genre_all = {g.strip() for v in df["product_L0_top3"].fillna("") for g in v.split("|") if g.strip()}
            genre_all = order_options(list(genre_all), GENRE_ORDER)
            genre_sel = st.multiselect("ジャンル", genre_all, default=[], key=f"{key_prefix}_genre")
        
        # 2. 研究分野 (L1) ・ 専門領域 (L2)
        row2_l1, row2_l2 = st.columns([1, 1])
        l1_all, l1_to_l2 = get_taxonomy_hierarchy(df["assigned_pairs"])
        with row2_l1:
            l1_sel = st.multiselect("研究分野L1", l1_all, default=[], key=f"{key_prefix}_l1")
        
        with row2_l2:
            l1_missing = len(l1_sel) == 0
            l2_cand = sorted({l2 for l1 in l1_sel for l2 in l1_to_l2.get(l1, [])}) if not l1_missing else []
            l2_sel = st.multiselect(
                "専門領域L2", 
                l2_cand, 
                default=[], 
                key=f"{key_prefix}_l2",
                disabled=l1_missing,
                help="研究分野L1を選択すると、詳細な専門領域を選べるようになります。" if l1_missing else None
            )

        # フィルタ適用
        use = df.copy()
        if "発行年" in use.columns:
            yy = pd.to_numeric(use["発行年"], errors="coerce")
            use = use[(yy >= y_from) & (yy <= y_to) | yy.isna()]
        
        use = apply_hierarchical_filters(use, genre_sel, l1_sel, l2_sel)

        return {
            "df": use,
            "year": (y_from, y_to),
            "genre": genre_sel,
            "l1": l1_sel,
            "l2": l2_sel,
        }
    else:
        # フォールバック
        y_val = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key=f"{key_prefix}_year")
        y_from, y_to = y_val
        use = df.copy()
        if "発行年" in use.columns:
            yy = pd.to_numeric(use["発行年"], errors="coerce")
            use = use[(yy >= y_from) & (yy <= y_to) | yy.isna()]
        return {"df": use, "year": (y_from, y_to)}