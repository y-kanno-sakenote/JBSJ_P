# modules/analysis/targettype_mod/ui_trend.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import yearly_counts, yearly_counts_hierarchical
from .base import TARGET_ORDER, TYPE_ORDER, split_multi
from .filters import summary_global_filters

def render_trend_block(df: pd.DataFrame, y_from: int, y_to: int, genre_sel: list[str], tg_sel: list[str], tp_sel: list[str]) -> None:
    has_wider = "target_pairs_top5" in df.columns and "research_pairs_top5" in df.columns
    
    # 1. 分析軸の選択
    c1, c2, c3, c4 = st.columns([2.0, 1.6, 6.1, 1.5])
    
    yearly = pd.DataFrame()
    target_mode_label = ""
    item_col = "item" # 統一カラム名

    with c1:
        if has_wider:
            options = ["ジャンル", "対象領域 (L1)", "対象物 (L2)", "研究分野", "具体的なテーマ"]
            if "product_L0_top3" not in df.columns:
                options = [o for o in options if o != "ジャンル"]
            
            trend_axis = st.selectbox("分析軸", options, index=1, key="obj_trend_axis") # Default to Target L1
            target_mode_label = trend_axis
            
            if trend_axis == "ジャンル":
                # product_L0_top3 は | 区切りだが split_multi で処理される yearly_counts を利用
                raw = yearly_counts(df, "product_L0_top3")
                # yearly_counts returns [発行年, col_name, count]
                if not raw.empty:
                    yearly = raw.rename(columns={"product_L0_top3": "item"})
            
            elif trend_axis == "対象領域 (L1)":
                yearly = yearly_counts_hierarchical(df, "target_pairs_top5", "L1")
            elif trend_axis == "対象物 (L2)":
                yearly = yearly_counts_hierarchical(df, "target_pairs_top5", "L2")
            elif trend_axis == "研究分野":
                yearly = yearly_counts_hierarchical(df, "research_pairs_top5", "L1")
            elif trend_axis == "具体的なテーマ":
                yearly = yearly_counts_hierarchical(df, "research_pairs_top5", "L2")
                
        else:
            # Legacy fallback
            target_mode = st.selectbox(
                "対象",
                ["対象物_top3", "研究タイプ_top3"],
                index=0,
                key="obj_trend_mode_legacy",
                format_func=lambda x: "対象物" if x == "対象物_top3" else ("研究分野" if x == "研究タイプ_top3" else str(x))
            )
            target_mode_label = "対象物" if target_mode == "対象物_top3" else "研究分野"
            raw = yearly_counts(df, target_mode)
            if not raw.empty:
                yearly = raw.rename(columns={target_mode: "item"})

    if yearly.empty:
        st.info("データがありません。")
        return

    # 2. 自動選択ロジック
    latest_year = int(yearly["発行年"].max()) if not yearly.empty else None
    auto_top: List[str] = []
    if latest_year is not None:
        auto_top = yearly[yearly["発行年"] == latest_year].sort_values("count", ascending=False)["item"].head(5).tolist()

    with c2:
        st.markdown('<div style="height:36px;"></div>', unsafe_allow_html=True)
        auto_top5 = st.checkbox("最新年Top5を自動選択", value=False, key="obj_trend_auto5")
        if "obj_trend_items" not in st.session_state:
            st.session_state["obj_trend_items"] = []

    if auto_top5 and auto_top:
        # 軸が変わった時などにリセットしたいが、簡易的に autoset フラグで管理
        # 軸変更検知が難しいので、latest_yearの変動をトリガーにする既存ロジックを維持
        if st.session_state.get("_obj_trend_autoset_val") != f"{latest_year}_{target_mode_label}":
            st.session_state["obj_trend_items"] = auto_top
            st.session_state["_obj_trend_autoset_val"] = f"{latest_year}_{target_mode_label}"

    # 3. 項目の絞り込み (multiselect)
    all_items_raw = sorted(yearly["item"].unique())
    # 並び順: TARGET_ORDER/TYPE_ORDER にあるものを優先表示（レガシー互換のため）
    # 新軸の場合は単純ソートでよいが、既存ロジックを流用して損はない
    all_items = [x for x in (TARGET_ORDER + TYPE_ORDER) if x in all_items_raw] + sorted([x for x in all_items_raw if x not in (TARGET_ORDER + TYPE_ORDER)])
    
    # セッションステート内の選択項目が、現在の全項目に含まれていなければ除外
    current_sel = st.session_state.get("obj_trend_items", [])
    valid_sel = [x for x in current_sel if x in all_items]
    if len(valid_sel) != len(current_sel):
        st.session_state["obj_trend_items"] = valid_sel

    with c3:
        sel = st.multiselect("表示する項目（複数可）", options=all_items[:1000], key="obj_trend_items")

    with c4:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="obj_trend_ma", help="年ごとのノイズをならします。")

    # 4. Pivot & Plot
    piv = yearly.pivot_table(index="発行年", columns="item", values="count", aggfunc="sum").fillna(0).sort_index()
    if sel:
        piv = piv[[c for c in sel if c in piv.columns]]
    
    if piv.shape[1] == 0:
        st.info("表示対象がありません。リストから1つ以上選んでください。")
        return

    if ma > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    _sel_key = ",".join(sel) if sel else "__ALL__"
    _uniq_key = f"obj_trend_plot|{target_mode_label}|{_sel_key}|ma{ma}"

    if HAS_PX:
        # Plotly Express Line Chart
        plot_df = piv.reset_index().melt(id_vars="発行年", var_name="項目", value_name="件数")
        fig = px.line(plot_df, x="発行年", y="件数", color="項目", markers=True)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
    else:
        st.line_chart(piv, key=_uniq_key)

    _shown_n = piv.shape[1]
    st.caption("条件：" + f"分析軸：{target_mode_label} ｜ 表示項目数：{_shown_n} ｜ 移動平均：{int(ma)}年 ｜ " + summary_global_filters(y_from, y_to, genre_sel, tg_sel, tp_sel))

    # 5. データダウンロード
    with st.expander("📊 表データを表示（トレンド）", expanded=False):
        try:
            tbl = piv.copy().reset_index()
            # 年の整形
            if "発行年" in tbl.columns:
                tbl["発行年"] = tbl["発行年"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
            
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            fname = f"trend_{target_mode_label}.csv"
            st.download_button(
                "📥 表をCSVで保存",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name=fname,
                mime="text/csv",
                key=f"dl_obj_trend_table_{_uniq_key}",
            )
        except Exception as _e:
            st.caption(f"表の表示に失敗しました: {_e!s}")

