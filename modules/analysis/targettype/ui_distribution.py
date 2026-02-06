# modules/analysis/targettype_mod/ui_distribution.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import count_series, count_hierarchy
from .filters import summary_global_filters

def _px_bar(df_xy: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str = None):
    if not HAS_PX or df_xy.empty:
        return None
    try:
        fig = px.bar(df_xy, x=x_col, y=y_col, text_auto=True, title=title, color=color_col)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=400, yaxis_title=y_col)
        fig.update_xaxes(tickangle=45, automargin=True)
        return fig
    except Exception:
        return None

def _px_sunburst(df_sb: pd.DataFrame, path: list[str], val_col: str, title: str):
    if not HAS_PX or df_sb.empty:
        return None
    try:
        fig = px.sunburst(df_sb, path=path, values=val_col, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=500)
        return fig
    except Exception:
        return None

def render_distribution_block(df: pd.DataFrame, y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str]) -> None:
    st.markdown("<style>.subttl{font-size:0.95rem; opacity:0.75; margin:0 0 0.25rem;}</style>", unsafe_allow_html=True)
    st.caption("条件：" + summary_global_filters(y_from, y_to, tg_sel, tp_sel))

    has_wider = "target_pairs_top5" in df.columns and "research_pairs_top5" in df.columns

    # --- 1. ジャンル分布 (もしあれば) ---
    if "product_L0_top3" in df.columns:
        g_df = count_series(df, "product_L0_top3").reset_index()
        g_df.columns = ["ジャンル", "件数"]
        if not g_df.empty:
            st.markdown("#### ▼ ジャンル分布")
            fig_g = _px_bar(g_df, "ジャンル", "件数", f"ジャンル別構成比（合計: {g_df['件数'].sum():,}件）", color_col="ジャンル")
            if fig_g: st.plotly_chart(fig_g, use_container_width=True)
            else: st.bar_chart(g_df.set_index("ジャンル")["件数"])

    st.markdown("---")

    if has_wider:
        # --- 新タクソナミー: サンバースト & L1分布 ---
        
        # 集計実行
        t_data = count_hierarchy(df, "target_pairs_top5")
        r_data = count_hierarchy(df, "research_pairs_top5")

        t_sb = t_data.get("sunburst", pd.DataFrame())
        r_sb = r_data.get("sunburst", pd.DataFrame())
        
        t_l1 = t_data.get("l1", pd.DataFrame())
        r_l1 = r_data.get("l1", pd.DataFrame())
        
        # 表示エリア
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### ▼ 対象領域 (Target)")
            if not t_sb.empty:
                st.markdown("**階層構造 (L1 → L2)**")
                fig_t_sb = _px_sunburst(t_sb, ["L1", "L2"], "count", "対象領域の階層分布")
                if fig_t_sb: st.plotly_chart(fig_t_sb, use_container_width=True)
                
                st.markdown("**L1（大分類）構成比**")
                fig_t_l1 = _px_bar(t_l1, "L1", "count", "対象領域(L1) 上位", color_col="L1")
                if fig_t_l1: st.plotly_chart(fig_t_l1, use_container_width=True)
            else:
                st.info("データがありません")

        with c2:
            st.markdown("#### ▼ 研究分野")
            if not r_sb.empty:
                st.markdown("**階層構造 (L1 → L2)**")
                fig_r_sb = _px_sunburst(r_sb, ["L1", "L2"], "count", "研究分野の階層分布")
                if fig_r_sb: st.plotly_chart(fig_r_sb, use_container_width=True)
                
                st.markdown("**L1（大分類）構成比**")
                fig_r_l1 = _px_bar(r_l1, "L1", "count", "研究分野(L1) 上位", color_col="L1")
                if fig_r_l1: st.plotly_chart(fig_r_l1, use_container_width=True)
            else:
                st.info("データがありません")
        
        # データダウンロード用 Expander
        with st.expander("📋 詳細データテーブルを表示"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.write("対象領域 (L1)")
                st.dataframe(t_l1, hide_index=True)
                st.write("対象物 (L2)")
                st.dataframe(t_data.get("l2", pd.DataFrame()), hide_index=True)
            with ec2:
                st.write("研究分野 (L1)")
                st.dataframe(r_l1, hide_index=True)
                st.write("具体的なテーマ (L2)")
                st.dataframe(r_data.get("l2", pd.DataFrame()), hide_index=True)

    else:
        # --- 旧来のロジック (フォールバック) ---
        tg_df = count_series(df, "対象物_top3").reset_index()
        tg_df.columns = ["対象物", "件数"]
        tp_df = count_series(df, "研究タイプ_top3").reset_index()
        tp_df.columns = ["研究分野", "件数"]

        c1, c2 = st.columns(2)
        with c1:
            if not tg_df.empty:
                fig = _px_bar(tg_df, "対象物", "件数", "対象物の出現件数")
                if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("データなし")
        with c2:
            if not tp_df.empty:
                fig2 = _px_bar(tp_df, "研究分野", "件数", "研究分野の出現件数")
                if fig2: st.plotly_chart(fig2, use_container_width=True)
            else: st.info("データなし")