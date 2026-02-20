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
from .compute import yearly_counts, yearly_counts_hierarchical, compute_chi2_and_residuals, HAS_SCIPY
from .base import TARGET_ORDER, TYPE_ORDER, split_multi
from .filters import summary_global_filters

def render_trend_block(df: pd.DataFrame, y_from: int, y_to: int, genre_sel: list[str], l1_sel: list[str], l2_sel: list[str]) -> None:
    has_wider = "assigned_pairs" in df.columns
    
    # 1. 分析軸の選択
    c1, c2, c3, c4 = st.columns([2.0, 1.6, 6.1, 1.5])
    
    yearly = pd.DataFrame()
    target_mode_label = ""
    item_col = "item" # 統一カラム名

    with c1:
        if has_wider:
            options = ["ジャンル", "研究分野 (L1)", "専門領域 (L2)"]
            if "product_L0_top3" not in df.columns:
                options = [o for o in options if o != "ジャンル"]
            
            trend_axis = st.selectbox("分析軸", options, index=1, key="obj_trend_axis")
            target_mode_label = trend_axis
            
            with c4:
                metric = st.radio("指標", ["件数", "比率 (%)"], index=0, horizontal=True, key="obj_trend_metric")

            if trend_axis == "ジャンル":
                raw = yearly_counts(df, "product_L0_top3")
                if not raw.empty:
                    yearly = raw.rename(columns={"product_L0_top3": "item"})
            
            elif trend_axis == "研究分野 (L1)":
                yearly = yearly_counts_hierarchical(df, "assigned_pairs", "L1")
            elif trend_axis == "専門領域 (L2)":
                yearly = yearly_counts_hierarchical(df, "assigned_pairs", "L2")
                
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
            
            with c4:
                metric = st.radio("指標", ["件数", "比率 (%)"], index=0, horizontal=True, key="obj_trend_metric_legacy")

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

    if metric.startswith("比率"):
        # 各年ごとの総件数を取得して比率を計算
        # yearly には軸に含まれる全アイテムのカウントが入っている
        yearly_total = yearly.groupby("発行年")["count"].sum().replace(0, 1)
        piv = (piv.T / yearly_total).T * 100.0

    if ma > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    _sel_key = ",".join(sel) if sel else "__ALL__"
    _metric_key = "ratio" if metric.startswith("比率") else "count"
    _uniq_key = f"obj_trend_plot|{target_mode_label}|{_sel_key}|ma{ma}|{_metric_key}"

    if HAS_PX:
        # Plotly Express Line Chart
        y_label = "比率 (%)" if metric.startswith("比率") else "件数"
        plot_df = piv.reset_index().melt(id_vars="発行年", var_name="項目", value_name=y_label)
        fig = px.line(plot_df, x="発行年", y=y_label, color="項目", markers=True)
        if metric.startswith("比率"):
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
    else:
        st.line_chart(piv, key=_uniq_key)

    _shown_n = piv.shape[1]
    st.caption("条件：" + f"分析軸：{target_mode_label} ｜ 指標：{metric} ｜ 表示項目数：{_shown_n} ｜ 移動平均：{int(ma)}年             " + summary_global_filters(y_from, y_to, genre_sel, l1_sel, l2_sel))

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

    # 6. カイ二乗検定と残差分析
    with st.expander("⚖️ 時期別の偏り検定（カイ二乗検定・残差分析）", expanded=False):
        st.write("時期（初期・中期・後期など）と分類カテゴリの間に偏りがあるか（どの時期にどの分野が多い/少ないか）を検定します。")
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            p1_name = st.text_input("第1期 名前", value="初期", key="chi2_p1_name")
            p1_start = st.number_input("開始年", value=1980, max_value=2050, key="chi2_p1_start")
            p1_end = st.number_input("終了年", value=1999, max_value=2050, key="chi2_p1_end")
        with c_p2:
            p2_name = st.text_input("第2期 名前", value="中期", key="chi2_p2_name")
            p2_start = st.number_input("開始年", value=2000, max_value=2050, key="chi2_p2_start")
            p2_end = st.number_input("終了年", value=2010, max_value=2050, key="chi2_p2_end")
        with c_p3:
            p3_name = st.text_input("第3期 名前", value="後期", key="chi2_p3_name")
            p3_start = st.number_input("開始年", value=2011, max_value=2050, key="chi2_p3_start")
            p3_end = st.number_input("終了年", value=2024, max_value=2050, key="chi2_p3_end")
            
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            use_p3 = st.checkbox("第3期を使用する", value=True, key="chi2_use_p3")
        with col_check2:
            use_bonferroni = st.checkbox("Bonferroni補正を適用", value=True, help="セル数（比較回数）が多い場合の多重比較問題を補正し、より厳密に評価します。")
        
        if st.button("🚀 検定を実行", key="chi2_run"):
            periods = []
            if p1_name: periods.append((p1_name, int(p1_start), int(p1_end)))
            if p2_name: periods.append((p2_name, int(p2_start), int(p2_end)))
            if use_p3 and p3_name: periods.append((p3_name, int(p3_start), int(p3_end)))
            
            if len(periods) < 2:
                st.error("比較するには2つ以上の期間が必要です。")
            else:
                target_col = "assigned_pairs" if has_wider else ("研究タイプ_top3" if target_mode_label == "研究分野" else "対象物_top3")
                level = "L1"
                if target_mode_label == "ジャンル":
                    target_col = "product_L0_top3"
                elif target_mode_label == "専門領域 (L2)" or target_mode_label == "対象物":
                    level = "L2"
                    
                res = compute_chi2_and_residuals(df, periods=periods, target_col=target_col, level=level)
                
                if "error" in res:
                    st.error(res["error"])
                elif not res:
                    st.warning("検定に必要なデータが不足しています。")
                else:
                    st.markdown(f"**カイ二乗検定の結果** (対象: {target_mode_label})")
                    p_val = res['p_value']
                    chi2 = res['chi2']
                    sig = "★ 有意な偏りがあります" if p_val < 0.05 else "有意な偏りは認められません"
                    st.write(f"p-value: **{p_val:.4e}** ({sig}) / χ² = {chi2:.2f} (df={res['dof']})")
                    
                    if p_val < 0.05:
                        st.markdown("**調整済み残差 (Adjusted Residuals)**")
                        
                        adj_res = res["adj_residuals"]
                        num_cells = adj_res.size
                        
                        import scipy.stats as st_stats
                        if use_bonferroni and HAS_SCIPY:
                            alpha_adj = 0.05 / num_cells
                            threshold = abs(st_stats.norm.ppf(alpha_adj / 2))
                            st.caption(f"※ Bonferroni補正適用（α' = 0.05 / {num_cells} = {alpha_adj:.5f}）: **±{threshold:.2f}以上**で有意と判定")
                        else:
                            threshold = 1.96
                            st.caption("※ 補正なし（α = 0.05）: **±1.96以上**で有意と判定")
                        
                        st.caption("赤色はポジティブ（期待値より有意に多い）、青色はネガティブ（期待値より有意に少ない）を示します。")
                        
                        def highlight_residuals(val):
                            if pd.isna(val): return ''
                            if val >= threshold: return 'background-color: rgba(255, 99, 132, 0.4); font-weight: bold;'
                            if val <= -threshold: return 'background-color: rgba(54, 162, 235, 0.4); font-weight: bold;'
                            return ''
                            
                        st.dataframe(adj_res.style.map(highlight_residuals).format("{:.2f}"))
                        
                        # サマリーテキスト
                        st.markdown(f"**時期別の特徴（絶対値が {threshold:.2f} 以上の分野）**")
                        for p_name in adj_res.index:
                            row = adj_res.loc[p_name]
                            increased = row[row >= threshold].index.tolist()
                            decreased = row[row <= -threshold].index.tolist()
                            
                            inc_text = f"↗️ **特化（多）**: {', '.join(increased)}" if increased else ""
                            dec_text = f"↘️ **過小（少）**: {', '.join(decreased)}" if decreased else ""
                            
                            if inc_text or dec_text:
                                st.write(f"- **{p_name}**: {' / '.join(filter(None, [inc_text, dec_text]))}")
                            else:
                                st.write(f"- **{p_name}**: 特筆すべき増減なし")
