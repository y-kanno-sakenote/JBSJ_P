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
        for p in str(entry).split("|"):
            if "::" in p:
                l1, l2 = p.split("::", 1)
                l1, l2 = l1.strip(), l2.strip()
                if l1:
                    l1_set.add(l1)
                    if l1 not in l1_to_l2: l1_to_l2[l1] = set()
                    if l2: l1_to_l2[l1].add(l2)
            else:
                l1 = p.strip()
                if l1: l1_set.add(l1)
    return sorted(l1_set), {k: sorted(v) for k, v in l1_to_l2.items()}

def parse_taxonomy_pairs(s):
    """L1::L2|L1::L2 形式の文字列を [(L1, L2), ...] のリストにパースする"""
    if not s or pd.isna(s): return []
    res = []
    pairs = str(s).split("|")
    for p in pairs:
        if "::" in p:
            parts = p.split("::", 1)
            res.append((parts[0].strip(), parts[1].strip()))
        else:
            res.append((p.strip(), ""))
    return res

def apply_hierarchical_filters(df, genre_sel=None, t_l1_sel=None, t_l2_sel=None, r_l1_sel=None, r_l2_sel=None):
    df2 = df.copy()
    
    # 引数の正規化（空文字やNoneを除外）
    def _norm_sel(s):
        if s is None: return []
        return [str(x).strip() for x in s if x and str(x).strip()]
    
    genre_sel = _norm_sel(genre_sel)
    t_l1_sel = _norm_sel(t_l1_sel)
    t_l2_sel = _norm_sel(t_l2_sel)
    r_l1_sel = _norm_sel(r_l1_sel)
    r_l2_sel = _norm_sel(r_l2_sel)

    if genre_sel and "product_L0_top3" in df2.columns:
        # L0_top3 は "ジャンル1|ジャンル2" 形式。データ側の空白にも耐えられるよう strip() する。
        df2 = df2[df2["product_L0_top3"].apply(lambda v: any(g in [x.strip() for x in str(v).split("|")] for g in genre_sel))]

    def is_l1(s):
        # タクソナミーのL1は「① 原料」「① 官能評価」のように丸数字で始まる。
        if not s: return False
        return s[0] in "①②③④⑤⑥⑦⑧⑨⑩"

    def hit_hierarchy(val, l1_sel, l2_sel):
        if not l1_sel and not l2_sel: return True
        pairs = parse_taxonomy_pairs(val)
        if not pairs: return False
        
        # フィルタリストのクリーンアップ: 
        # 分析タブ等から「全選択リスト」が l1_sel, l2_sel 両方に渡されるケースに対応。
        # L1判定にはL1ラベルのみ、L2判定にはL2ラベルのみを使用することで、
        # 「(L1条件) AND (L2条件)」が正しく機能するようにする。
        l1_f = [x for x in l1_sel if is_l1(x)]
        l2_f = [x for x in l2_sel if not is_l1(x)]
        
        # 論文がいずれかのペアで条件を満たせばヒットとする。
        # 各ペアにおいて、「L1が選択されているならL1にヒット」かつ「L2が選択されているならL2にヒット」を判定。
        for p1, p2 in pairs:
            m1 = (p1 in l1_f) if l1_f else True
            m2 = (p2 in l2_f) if l2_f else True
            if m1 and m2: return True
        return False

    if (t_l1_sel or t_l2_sel) and "target_pairs_top5" in df2.columns:
        df2 = df2[df2["target_pairs_top5"].apply(lambda v: hit_hierarchy(v, t_l1_sel, t_l2_sel))]
    
    if (r_l1_sel or r_l2_sel) and "research_pairs_top5" in df2.columns:
        df2 = df2[df2["research_pairs_top5"].apply(lambda v: hit_hierarchy(v, r_l1_sel, r_l2_sel))]
    
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
    新しいタクソナミー列がある場合は階層型フィルタを表示。
    """
    if df is None or df.empty:
        st.info("フィルター対象データがありません。")
        return df

    # 1. 基本（年）
    y = pd.to_numeric(df.get("発行年", pd.Series(dtype=float)), errors="coerce")
    ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)

    if show_caption:
        st.caption("年・ジャンル・対象領域・研究手法で絞り込みできます。")

    # 新しいタクソナミー列（ジャンル、対象、研究手法）の存在確認
    has_wider = all(c in df.columns for c in ["product_L0_top3", "target_pairs_top5", "research_pairs_top5"])

    genre_sel, t_l1_sel, t_l2_sel, r_l1_sel, r_l2_sel = [], [], [], [], []
    tg_legacy, tp_legacy = [], []

    if has_wider:
        # --- 新しい階層型フィルタ UI ---
        # 1. 基本（年・ジャンル）
        row1_y, row1_g = st.columns([1, 1])
        with row1_y:
            y_val = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key=f"{key_prefix}_year")
            y_from, y_to = y_val
        with row1_g:
            genre_all = {g.strip() for v in df["product_L0_top3"].fillna("") for g in v.split("|") if g.strip()}
            genre_all = order_options(list(genre_all), GENRE_ORDER)
            genre_sel = st.multiselect("ジャンル", genre_all, default=[], key=f"{key_prefix}_genre")
        
        # 2. 対象領域・対象物
        row2_t1, row2_t2 = st.columns([1, 1])
        t_l1_all, t_l1_to_l2 = get_taxonomy_hierarchy(df["target_pairs_top5"])
        with row2_t1:
            t_l1_sel = st.multiselect("対象領域 (L1)", t_l1_all, default=[], key=f"{key_prefix}_t_l1")
        
        with row2_t2:
            t_l1_missing = len(t_l1_sel) == 0
            t2_cand = sorted({l2 for l1 in t_l1_sel for l2 in t_l1_to_l2.get(l1, [])}) if not t_l1_missing else []
            t_l2_sel = st.multiselect(
                "対象物 (L2)", 
                t2_cand, 
                default=[], 
                key=f"{key_prefix}_t_l2",
                disabled=t_l1_missing,
                help="対象領域 (L1) を選択すると、詳細な対象物を選べるようになります。" if t_l1_missing else None
            )

        # 3. 研究分野・具体的なテーマ
        row3_r1, row3_r2 = st.columns([1, 1])
        r_l1_all, r_l1_to_l2 = get_taxonomy_hierarchy(df["research_pairs_top5"])
        with row3_r1:
            r_l1_sel = st.multiselect("研究分野", r_l1_all, default=[], key=f"{key_prefix}_r_l1")
        
        with row3_r2:
            r_l1_missing = len(r_l1_sel) == 0
            r2_cand = sorted({l2 for l1 in r_l1_sel for l2 in r_l1_to_l2.get(l1, [])}) if not r_l1_missing else []
            r_l2_sel = st.multiselect(
                "具体的なテーマ", 
                r2_cand, 
                default=[], 
                key=f"{key_prefix}_r_l2",
                disabled=r_l1_missing,
                help="研究分野を選択すると、具体的なテーマを選べるようになります。" if r_l1_missing else None
            )
    else:
        # 年スライダー（フォールバック用）
        y_val = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key=f"{key_prefix}_year")
        y_from, y_to = y_val

        # --- 従来型フィルタ UI (フォールバック) ---
        targets_all = list({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for w in _split_multi(v) if w})
        types_all   = list({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for w in _split_multi(v) if w})
        targets_all = _order_options(targets_all, target_order or TARGET_ORDER)
        types_all   = _order_options(types_all, type_order or TYPE_ORDER)

        c2, c3 = st.columns([1, 1])
        with c2:
            tg_legacy = st.multiselect("対象物で絞り込み", options=targets_all, default=[], key=f"{key_prefix}_tg")
        with c3:
            tp_legacy = st.multiselect("研究分野で絞り込み", options=types_all, default=[], key=f"{key_prefix}_tp")

    # フィルタ適用
    use = df.copy()
    if "発行年" in use.columns:
        yy = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(yy >= y_from) & (yy <= y_to) | yy.isna()]
    
    if has_wider:
        use = apply_hierarchical_filters(use, genre_sel, t_l1_sel, t_l2_sel, r_l1_sel, r_l2_sel)
    else:
        if tg_legacy and "対象物_top3" in use.columns:
            use = use[_contains_any(use["対象物_top3"], tg_legacy)]
        if tp_legacy and "研究タイプ_top3" in use.columns:
            use = use[_contains_any(use["研究タイプ_top3"], tp_legacy)]

    # 戻り値：後方互換性と情報提供のために辞書を返す
    return {
        "df": use,
        "year": (y_from, y_to),
        "genre": genre_sel,
        "target_l1": t_l1_sel,
        "target_l2": t_l2_sel,
        "research_l1": r_l1_sel,
        "research_l2": r_l2_sel,
        "targets": tg_legacy if not has_wider else (t_l1_sel + t_l2_sel),
        "types": tp_legacy if not has_wider else (r_l1_sel + r_l2_sel)
    }