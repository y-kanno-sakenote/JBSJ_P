from __future__ import annotations
import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    from wordcloud import WordCloud  # type: ignore
    HAS_WC = True
except Exception:
    HAS_WC = False

from .compute import keyword_freq_by_mode, keyword_tfidf
from .images import get_japanese_font_path, safe_show_image
from .base import short_preview, get_banner_filters
from .copyui import expander as copy_expander

def _freq_to_df(freq: pd.Series, topn: int, value_label: str = "件数") -> pd.DataFrame:
    if freq.empty: return pd.DataFrame(columns=["キーワード", value_label])
    df = freq.head(int(topn)).reset_index()
    df.columns = ["キーワード", value_label]
    return df

def _build_caption(df_use: pd.DataFrame, topn: int, min_total: int, mode: str) -> str:
    y_from, y_to, tg_sel, tp_sel = get_banner_filters(prefix="kw")
    if y_from is not None and y_to is not None:
        period = f"{int(y_from)}–{int(y_to)}"
    else:
        period = "—"

    parts = [
        f"条件：表示件数：{int(topn)}",
        f"最低回数≧{int(min_total)}",
        "DF（登場論文数）" if mode=="df" else "TF（総出現回数）" if mode=="tf" else "特徴度（TF-IDF）",
        f"期間：{period}",
    ]
    tg = short_preview(tg_sel or [])
    tp = short_preview(tp_sel or [])
    if tg:
        parts.append(f"対象物：{tg}")
    if tp:
        parts.append(f"研究タイプ：{tp}")
    return " ｜ ".join(parts)

def render_freq_block(df_use: pd.DataFrame, df_all: pd.DataFrame | None = None) -> None:
    c1, c2, c3 = st.columns([1, 1, 2.0])
    with c1:
        topn = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5, key="kw_freq_topn")
    with c2:
        min_total = st.number_input("最低総出現回数", min_value=1, max_value=100, value=3, step=1, key="kw_freq_min_total")
    with c3:
        label = st.radio("カウント方式", ["登場論文数（DF）", "総出現回数（TF）", "特徴度（TF-IDF）"], index=0, horizontal=True, key="kw_freq_countmode")
        if "DF" in label: mode = "df"
        elif "TF-IDF" in label: mode = "tfidf"
        else: mode = "tf"

    if mode == "tfidf" and df_all is not None:
        freq = keyword_tfidf(df_use, df_all)
        value_label = "特徴スコア"
        title_suffix = "（特徴度：TF-IDF）"
    else:
        freq = keyword_freq_by_mode(df_use, mode=("df" if mode=="df" else "tf"))
        value_label = "件数"
        title_suffix = "（登場論文数）" if mode == "df" else "（出現回数）"

    if freq.empty:
        st.info("条件に合うキーワードが見つかりませんでした。"); return
        
    # 特徴度（tfidf）の場合は出現回数での足切りを別途行う（スコアが低くても出現回数が多いものを除外したくない場合もあるが、ノイズ除去のため）
    if int(min_total) > 1:
        # tfidf の場合でも、頻度ベースの series を取得してフィルタリング
        df_count = keyword_freq_by_mode(df_use, mode="df")
        freq = freq[freq.index.isin(df_count[df_count >= int(min_total)].index)]

    freq_df = _freq_to_df(freq, int(topn), value_label=value_label)
    if freq_df.empty:
        st.info("（フィルタで該当なし）条件を緩めてください。"); return

    # 左に表、右にグラフ
    left, right = st.columns([1.0, 1.1])
    display_height = 420
    with left:
        st.dataframe(freq_df, use_container_width=True, hide_index=True, height=display_height)

    with right:
        if HAS_PX:
            try:
                df_chart = freq_df.sort_values(value_label, ascending=False).head(10)
                df_plot = df_chart.sort_values(value_label, ascending=True)
                fig = px.bar(df_plot, x=value_label, y="キーワード", orientation='h', text_auto='.2f' if mode=="tfidf" else True, title=f"キーワード分析（Top10）{title_suffix}")
                fig.update_layout(margin=dict(l=6, r=6, t=40, b=6), height=display_height, bargap=0.20, bargroupgap=0.06)
                fig.update_yaxes(automargin=True)
                fig.update_traces(marker_line_width=0)
                fig.update_layout(xaxis_title="")
                fig.update_layout(annotations=[
                    dict(
                        x=1.0, y=-0.02, xref='paper', yref='paper',
                        text=value_label, showarrow=False, xanchor='right', yanchor='bottom',
                    )
                ])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(freq_df.set_index("キーワード")[value_label].sort_values(ascending=False).head(10))
        else:
            st.bar_chart(freq_df.set_index("キーワード")[value_label].sort_values(ascending=False).head(10))

    if mode == "tfidf":
        st.info("💡 **特徴度 (TF-IDF)**: データベース全体で「ありふれた語」を抑制し、現在の検索結果に特有の単語を際立たせる指標です。")

    st.caption(_build_caption(df_use, topn, min_total, mode))
    copy_expander("📋 キーワードをすぐコピー", freq_df["キーワード"].astype(str).tolist())

    with st.expander("☁ WordCloud", expanded=False):
        if HAS_WC and st.button("生成する", key="kw_wc_btn"):
            textfreq = {row["キーワード"]: int(row["件数"]) for _, row in freq_df.iterrows()}
            wc = WordCloud(width=900, height=450, background_color="white",
                           collocations=False, prefer_horizontal=1.0,
                           font_path=get_japanese_font_path() or None)
            img = wc.generate_from_frequencies(textfreq).to_image()
            safe_show_image(img)
        elif not HAS_WC:
            st.caption("※ wordcloud が未導入のため非表示です。")