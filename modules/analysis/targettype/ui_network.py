# modules/analysis/targettype_mod/ui_network.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import pandas as pd
import streamlit as st
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

from .compute import build_cooccur_edges, example_titles_for_edge, build_cooccur_edges_hierarchical, example_titles_for_edge_hierarchical
from .base import node_options_for_mode
from .filters import summary_global_filters, parse_taxonomy_pairs

# ... (Helper functions remain unchanged) ...

def _color_square_data_uri(color_hex: str, size_px: int = 12) -> str:
    color = str(color_hex or "#999999"); size = max(6, int(size_px))
    try:
        from PIL import Image; import io, base64
        img = Image.new("RGBA", (size, size), color); buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii"); return f"data:image/png;base64,{b64}"
    except Exception:
        import base64
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"><rect width="{size}" height="{size}" fill="{color}"/></svg>'
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii"); return "data:image/svg+xml;base64," + b64

def _compute_communities_from_edges(edges: pd.DataFrame):
    if edges.empty or not HAS_NX: return {}, {}
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1)))
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_id = {str(n): i for i, cset in enumerate(comms) for n in cset}
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}
    base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
                   "#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd",
                   "#e6550d","#31a354","#756bb1","#636363","#9ecae1","#fdae6b","#74c476","#bcbddc","#bdbdbd"]
    palette = {i: base_colors[i % len(base_colors)] for i in set(comm_id.values())}
    return comm_id, palette

def _render_cluster_legend_counts(palette: dict[int, str], comm_id: dict[str, int]) -> None:
    if not palette: return
    counts = {}
    for _, cid in (comm_id or {}).items(): counts[cid] = counts.get(cid, 0) + 1
    items = sorted(palette.items(), key=lambda kv: kv[0])
    html = ['<div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:6px 0 10px 0;">',
            '<span style="font-weight:700; margin-right:4px;">クラスター凡例</span>']
    for cid, color in items:
        cnt = counts.get(cid, 0)
        square = f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{color};border:1px solid #999;margin:0 6px 0 2px;"></span>'
        label = f'C{int(cid)+1}（{cnt}語）'
        html.append(f'<span style="display:inline-flex; align-items:center; gap:4px;">{square}<span style="font-size:12.5px;opacity:0.9;">{label}</span></span>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 680, fixed_layout: bool = False, node_colors: dict[str,str] | None = None, footer_caption: str | None = None) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。"); return
    if edges.empty:
        st.warning("エッジがありません。"); return

    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), float(r.get("weight", 1))
        if G.has_edge(s, t): G[s][t]["weight"] += w
        else: G.add_edge(s, t, weight=w)

    strength = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_id = {n: i for i, cset in enumerate(comms) for n in cset}
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}
    base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
                   "#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd",
                   "#e6550d","#31a354","#756bb1","#636363","#9ecae1","#fdae6b","#74c476","#bcbddc","#bdbdbd"]
    palette = {i: base_colors[i % len(base_colors)] for i in set(comm_id.values())}
    def node_color(n: str) -> str:
        if node_colors and n in node_colors: return node_colors[n]
        cid = int(comm_id.get(n, 0)); return palette.get(cid, "#999999")

    from pyvis.network import Network
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    if fixed_layout:
        net.set_options('{"interaction":{"hover":true,"tooltipDelay":200,"zoomView":true,"dragView":true},"physics":{"enabled":false},"layout":{"improvedLayout":true,"randomSeed":42},"nodes":{"shape":"dot"},"edges":{"smooth":{"type":"dynamic"}}}')
    else:
        net.set_options('{"interaction":{"hover":true,"tooltipDelay":200,"zoomView":true,"dragView":true},"physics":{"stabilization":{"enabled":true,"iterations":200},"barnesHut":{"gravitationalConstant":-25000,"centralGravity":0.2,"springLength":140,"springConstant":0.025,"damping":0.4,"avoidOverlap":0.5}},"nodes":{"shape":"dot"},"edges":{"smooth":{"type":"dynamic"}}}')

    def size_for(n): s = max(1.0, float(strength.get(n, 1))); return max(6.0, min(28.0, 6.0 + 4.0 * math.log1p(s)))
    for n in G.nodes():
        lbl = n if n in set(sorted(G.nodes(), key=lambda x: strength.get(x,0), reverse=True)[:40]) else ""
        title = f"{n}<br>総共起重み: {strength.get(n,0):,.0f}"
        net.add_node(n, label=lbl, title=title, value=strength.get(n,0), size=size_for(n), color=node_color(n))

    def width_for(w): return max(1.0, min(10.0, 1.0 + 2.0 * math.log1p(float(w))))
    for s, t, d in G.edges(data=True):
        w = d.get("weight", 1); net.add_edge(s, t, value=float(w), width=width_for(w), title=f"共起: {int(w)} 回")

    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)
    if footer_caption: st.caption(footer_caption)
    st.download_button("📥 ネットワークHTML", data=html.encode("utf-8"), file_name="cooccurrence_network.html", mime="text/html", key="dl_pyvis_html")

def _get_l2_nodes(df: pd.DataFrame, col: str) -> list[str]:
    nodes = set()
    for v in df[col].fillna(""):
        for p in parse_taxonomy_pairs(v):
            if p[1]: nodes.add(p[1])
    return sorted(list(nodes))

def render_cooccurrence_block(df_use: pd.DataFrame, y_from: int, y_to: int, genre_sel: list[str], l1_sel: list[str], l2_sel: list[str]) -> None:
    has_wider = "assigned_pairs" in df_use.columns
    
    c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1.0, 1.6, 1.6])
    
    mode_map = {}
    if has_wider:
        mode_map = {
            "専門領域(L2)同士": "L2 Only",
            "研究分野(L1)同士": "L1 Only",
            "研究分野(L1)×専門領域(L2)": "L1 x L2"
        }
    else:
        # Legacy
        mode_map = {
            "対象物のみ": "対象物のみ",
            "研究分野のみ": "研究分野のみ",
            "対象物×研究分野": "対象物×研究分野"
        }
        
    with c1:
        mode_label = st.selectbox("ネットワークの種類", list(mode_map.keys()), index=0, key="obj_net_mode")
        mode_val = mode_map[mode_label]
        
    # ノード候補の取得
    node_options = []
    if has_wider:
        def _get_nodes(df, idx):
            nodes = set()
            for v in df["assigned_pairs"].fillna(""):
                for p in parse_taxonomy_pairs(v):
                    if p[idx]: nodes.add(p[idx])
            return sorted(list(nodes))

        if mode_val == "L2 Only":
            node_options = _get_nodes(df_use, 1)
        elif mode_val == "L1 Only":
            node_options = _get_nodes(df_use, 0)
        else:
            n1 = _get_nodes(df_use, 0)
            n2 = _get_nodes(df_use, 1)
            node_options = sorted(list(set(n1 + n2)))
    else:
        node_options = node_options_for_mode(df_use, mode_val)

    with c2:
        topN = st.number_input("表示するノード数（多い順）", min_value=30, max_value=300, value=120, step=10, key="obj_net_topn")
    with c3:
        min_edge = st.number_input("最低共起数（同時出現）", min_value=1, max_value=50, value=3, step=1, key="obj_net_minw")
        
    with c4:
        include_terms = st.multiselect("必須（選択式）", options=node_options, default=[], key="obj_net_include_sel")
    with c5:
        exclude_terms = st.multiselect("除外（選択式）", options=node_options, default=[], key="obj_net_exclude_sel")

    # エッジ構築
    edges = pd.DataFrame()
    if has_wider:
         edges = build_cooccur_edges_hierarchical(df_use, mode_val, int(min_edge))
    else:
         edges = build_cooccur_edges(df_use, mode_val, int(min_edge))

    if not edges.empty and (include_terms or exclude_terms):
        e = edges.copy()
        if include_terms:
            incl = set(include_terms); e = e[(e["src"].isin(incl)) | (e["dst"].isin(incl))]
        if exclude_terms:
            excl = set(exclude_terms); e = e[~(e["src"].isin(excl) | e["dst"].isin(excl))]
        edges = e.reset_index(drop=True)
    if not edges.empty and int(topN) > 0:
        deg = pd.concat([edges.groupby("src")["weight"].sum(), edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    # 1. 転置インデックス作成 (高速化)
    title_col = "論文タイトル" # or preferred
    from .compute import prefer_title_column
    title_col = prefer_title_column(df_use) or "論文タイトル"
    
    df_titles = df_use[df_use[title_col].notna()].copy()
    inv_idx_tg = {}
    inv_idx_rp = {}
    
    has_pre = "_target_pairs" in df_titles.columns and "_research_pairs" in df_titles.columns
    if has_pre:
        for idx, row in df_titles.iterrows():
            for p in row["_target_pairs"]:
                if p[1]: inv_idx_tg.setdefault(p[1], set()).add(idx)
            for p in row["_research_pairs"]:
                if p[1]: inv_idx_rp.setdefault(p[1], set()).add(idx)
    
    comm_id, palette = _compute_communities_from_edges(edges)
    edge_clusters, edge_colors, ex_titles = [], [], []
    
    for _, r in edges.iterrows():
        a, b = str(r["src"]), str(r["dst"])
        ca, cb = comm_id.get(a, 0), comm_id.get(b, 0)
        c_use = ca if ca == cb else ca
        edge_clusters.append(c_use)
        edge_colors.append(palette.get(c_use, "#999999"))
        
        # タイトル例取得
        if has_pre:
            if mode_val in ("Target L2 Only", "対象物のみ"):
                matches = inv_idx_tg.get(a, set()) & inv_idx_tg.get(b, set())
            elif mode_val in ("Research L2 Only", "研究分野のみ"):
                matches = inv_idx_rp.get(a, set()) & inv_idx_rp.get(b, set())
            else: # Cross
                matches = inv_idx_tg.get(a, set()) & inv_idx_rp.get(b, set())
            
            if matches:
                titles = df_titles.loc[list(matches)[:3], title_col].tolist()
                ex_titles.append(" / ".join(map(str, titles)))
            else:
                ex_titles.append("")
        else:
            # Fallback
            if has_wider:
                titles = example_titles_for_edge_hierarchical(df_use, mode_val, a, b, limit=3)
            else:
                titles = example_titles_for_edge(df_use, mode_val, a, b, limit=3)
            ex_titles.append(" / ".join(map(str, titles)))

    edges = edges.copy()
    edges["cluster_id"] = edge_clusters
    edges["cluster_color"] = edge_colors
    edges["example_titles"] = ex_titles

    st.caption(f"エッジ数: {len(edges)}")

    _cond = ("条件："
             + f"ネットワーク：{mode_label} ｜ 表示するノード数：{int(topN)} ｜ 最低共起数≧{int(min_edge)} ｜ "
             + (f"必須：{len(include_terms)}件 ｜ " if include_terms else "必須：0件 ｜ ")
             + (f"除外：{len(exclude_terms)}件 ｜ " if exclude_terms else "除外：0件 ｜ ")
             + summary_global_filters(y_from, y_to, genre_sel, l1_sel, l2_sel))

    if mode_val in ("対象物のみ", "Target L2 Only"):
        col_a, col_b = "対象物A", "対象物B"
    elif mode_val in ("研究分野のみ", "Research L2 Only"):
        col_a, col_b = "具体的なテーマA", "具体的なテーマB"
    else:
        col_a, col_b = "対象物", "具体的なテーマ"

    disp = edges.rename(columns={"src": col_a, "dst": col_b, "weight": "共起回数"}).copy()
    disp["cluster_img"] = disp["cluster_color"].map(lambda c: _color_square_data_uri(c, 12))
    disp_view = disp[["cluster_img", col_a, col_b, "共起回数", "example_titles"]].head(200)
    try:
        st.dataframe(
            disp_view, use_container_width=True, hide_index=True,
            column_config={
                "cluster_img": st.column_config.ImageColumn("cluster", width="small", help="クラスタ（表・ネットワークと色連動）"),
                col_a: st.column_config.TextColumn(col_a, width="medium"),
                col_b: st.column_config.TextColumn(col_b, width="medium"),
                "共起回数": st.column_config.NumberColumn("共起回数", format="%d", width="small"),
                "example_titles": st.column_config.TextColumn("example_titles", width="large", help="そのペアが同時に登場する論文タイトル（最大3件）"),
            },
        )
    except Exception:
        st.dataframe(disp_view, use_container_width=True, hide_index=True)

    if palette: _render_cluster_legend_counts(palette, comm_id)
    st.caption(_cond)

    with st.expander("🕸️ ネットワークを可視化", expanded=False):
        fix_layout = st.checkbox("レイアウトを固定", value=False, key="obj_net_fix_layout")
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="obj_net_draw"):
                node_colors = {}
                for _, r in edges.iterrows():
                    node_colors[str(r["src"])] = r.get("cluster_color", "#999999")
                    node_colors[str(r["dst"])] = r.get("cluster_color", "#999999")
                _foot = ("条件："
                         + f"ネットワーク：{mode_label} ｜ 表示するノード数：{int(topN)} ｜ 最低共起数≧{int(min_edge)} ｜ "
                         + (f"必須：{len(include_terms)}件 ｜ " if include_terms else "必須：0件 ｜ ")
                         + (f"除外：{len(exclude_terms)}件 ｜ " if exclude_terms else "除外：0件 ｜ ")
                         + summary_global_filters(y_from, y_to, genre_sel, l1_sel, l2_sel))
                _draw_pyvis_from_edges(edges, height_px=680, fixed_layout=fix_layout, node_colors=node_colors, footer_caption=_foot)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")