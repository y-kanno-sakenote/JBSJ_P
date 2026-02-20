# modules/analysis/targettype_mod/ui_cross.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import cross_counts, cross_counts_hierarchical
from .base import TARGET_ORDER, TYPE_ORDER
from .filters import summary_global_filters

def render_cross_block(df: pd.DataFrame, y_from: int, y_to: int, genre_sel: list[str], l1_sel: list[str], l2_sel: list[str]) -> None:
    st.markdown('<div style="font-weight=600; font-size:1.1rem; margin:0 0 0.25rem;">対象領域 × 研究分野（クロスヒートマップ）</div>', unsafe_allow_html=True)
    
    has_wider = "assigned_pairs" in df.columns
    cross = pd.DataFrame()
    x_label, y_label = "", ""

    if has_wider:
        # --- 新UI: 軸のレベル選択 ---
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            y_level = st.radio("縦軸の粒度", ["L1（研究分野）", "L2（専門領域）"], horizontal=True, index=0, key="cross_y_level")
        with c_opt2:
            x_level = st.radio("横軸の粒度", ["L1（研究分野）", "L2（専門領域）"], horizontal=True, index=1, key="cross_x_level")
        
        y_lvl_code = "L1" if "L1" in y_level else "L2"
        x_lvl_code = "L1" if "L1" in x_level else "L2"
        
        y_label = y_level
        x_label = x_level
        
        # 集計 (assigned_pairs を想定)
        cross = cross_counts_hierarchical(df, "assigned_pairs", y_lvl_code, "assigned_pairs", x_lvl_code)
        
    else:
        # --- 旧UI (フォールバック) ---
        x_label = "研究分野"
        y_label = "対象物"
        cross = cross_counts(df, "対象物_top3", "研究タイプ_top3")

    if cross.empty:
        st.info("クロス集計できるデータがありません。")
        return

    # Pivot: A=縦(Target), B=横(Research) を想定していたが、compute側は A, B なので
    # cross_counts_hierarchical は A=Target, B=Research として返している想定
    # cross columns: A, B, count
    piv = cross.pivot(index="A", columns="B", values="count").fillna(0).astype(int)
    piv.index.name = y_label
    piv.columns.name = x_label
    
    # 並び順の制御（レガシーデータの場合のみ既存のORDERを適用、新データは件数順）
    if not has_wider:
        idx_order = [x for x in TARGET_ORDER if x in piv.index] + sorted([x for x in piv.index if x not in TARGET_ORDER])
        cols_order = [x for x in TYPE_ORDER if x in piv.columns] + sorted([x for x in piv.columns if x not in TYPE_ORDER])
        piv = piv.reindex(index=idx_order, columns=cols_order)
    else:
        # 件数が多い順に並べ替え（index, columnsともに）
        # 行の合計
        row_sums = piv.sum(axis=1).sort_values(ascending=False)
        col_sums = piv.sum(axis=0).sort_values(ascending=False)
        piv = piv.reindex(index=row_sums.index, columns=col_sums.index)

    show_values = bool(st.session_state.get("obj_cross_show_values", False))

    if HAS_PX:
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", labels=dict(color="件数"))
        fig.update_xaxes(tickangle=45, automargin=True)
        fig.update_yaxes(automargin=True)
        
        # ホバーテンプレート
        fig.update_traces(hovertemplate=f"{y_label}=%{{y}}<br>{x_label}=%{{x}}<br>件数=%{{z}}<extra></extra>")
        if show_values:
            fig.update_traces(text=piv.values, texttemplate="%{text}")
            
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), coloraxis_colorbar_title="件数")
        st.plotly_chart(fig, use_container_width=True)
        
        rb_spacer, rb_cb = st.columns([6, 1])
        with rb_cb:
            st.checkbox("セルの値を表示", value=show_values, key="obj_cross_show_values")
    else:
        st.dataframe(piv, use_container_width=True)

    st.caption("条件：" + summary_global_filters(y_from, y_to, genre_sel, l1_sel, l2_sel))

    with st.expander("📋 ヒートマップ表データを表示", expanded=False):
        try:
            st.dataframe(piv, use_container_width=True)
            st.download_button(
                "📥 表をCSVで保存",
                data=piv.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="cross_heatmap_table.csv",
                mime="text/csv",
                key="dl_cross_piv_csv",
            )
        except Exception as _e:
            st.caption(f"表の表示に失敗しました: {_e!s}")