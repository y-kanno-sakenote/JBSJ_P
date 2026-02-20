from __future__ import annotations
import pandas as pd
import streamlit as st
import random
import numpy as np

try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network  # type: ignore
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

try:
    from networkx.algorithms.community import greedy_modularity_communities as _greedy_comms  # type: ignore
    from networkx.algorithms.community.quality import modularity as _nx_modularity
    HAS_COMMUNITY = True
except Exception:
    HAS_COMMUNITY = False

try:
    from networkx.algorithms.centrality import betweenness_centrality as _betweenness
    from networkx.algorithms.cluster import average_clustering as _avg_clustering
    HAS_ALGO = True
except Exception:
    HAS_ALGO = False

from .base import PALETTE

@st.cache_data(ttl=3600, show_spinner=False)
def compute_node_communities_from_edges(edges: pd.DataFrame) -> dict[str, int]:
    if edges is None or edges.empty or not HAS_NX or not HAS_COMMUNITY: return {}
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1.0)))
    try:
        comms = list(_greedy_comms(G, weight="weight"))
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for gi, nodes in enumerate(comms):
        for n in nodes: mapping[str(n)] = gi
    return mapping

@st.cache_data(ttl=3600, show_spinner=False)
def build_pyvis_html(edges: pd.DataFrame, height_px: int = 650, freeze_layout: bool = False, font_size: int = 16) -> tuple[str, str]:
    if not (HAS_NX and HAS_PYVIS): return ("","")
    if edges is None or edges.empty: return ("","")
    import pandas as _pd
    import networkx as _nx
    from pyvis.network import Network as _Network

    G = _nx.Graph()
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"]); w = float(r["weight"])
        if G.has_edge(s, t): G[s][t]["weight"] += w
        else: G.add_edge(s, t, weight=w)
    if G.number_of_nodes()==0: return ("","")

    deg = dict(G.degree())
    deg_w = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        deg_w[u] += w; deg_w[v] += w

    legend_html: str = ""
    if HAS_COMMUNITY:
        from networkx.algorithms.community import greedy_modularity_communities as _greedy
        comms = list(_greedy(G, weight="weight"))
        node_group: dict[str, int] = {}
        for gi, nodes in enumerate(comms):
            for n in nodes: node_group[str(n)] = gi
        chips = []
        for i, nodes in enumerate(comms):
            col = PALETTE[i % len(PALETTE)]
            chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> C{i+1}（{len(nodes)}語）")
        legend_html = " ".join(chips)
    else:
        node_group = {}

    net = _Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-25000, central_gravity=0.15, spring_length=140, spring_strength=0.03, damping=0.18)

    def _scale(v, vmin, vmax, out_min=8, out_max=36):
        if vmax==vmin: return (out_min+out_max)/2
        r = (v - vmin) / (vmax - vmin)
        return out_min + r * (out_max - out_min)

    w_values = list(deg_w.values())
    wmin, wmax = (min(w_values), max(w_values)) if w_values else (0.0, 1.0)

    for n in G.nodes():
        wsum = float(deg_w.get(n, 0.0)); d = int(deg.get(n, 1))
        size = _scale(wsum if wsum>0 else d, wmin if wmin>0 else 0.0, wmax if wmax>0 else 1.0)
        g = node_group.get(str(n))
        if g is not None:
            gi = int(g)
            G.nodes[n]["group"] = gi
            G.nodes[n]["color"] = PALETTE[gi % len(PALETTE)]
        label = str(n)
        label_short = label if len(label)<=18 else (label[:16] + "…")
        G.nodes[n]["label"] = label_short
        G.nodes[n]["title"] = f"{label}&lt;br&gt;重み合計: {wsum:.0f} / 度数: {d}"
        G.nodes[n]["value"] = size

    e_w = [float(d.get("weight", 1.0)) for _,_,d in G.edges(data=True)]
    ew_min, ew_max = (min(e_w), max(e_w)) if e_w else (1.0, 1.0)
    def _ew_scale(w): 
        if ew_max==ew_min: return 1.5
        return 1.0 + 4.0 * (w - ew_min) / (ew_max - ew_min)
    for u,v,d in G.edges(data=True):
        d["width"] = _ew_scale(float(d.get("weight", 1.0)))

    net.from_nx(G)
    try:
        net.set_options(f"""{{"interaction":{{"hover":true,"navigationButtons":false,"multiselect":true,"tooltipDelay":120}},
                            "nodes":{{"shape":"dot","shadow":true,"scaling":{{"min":8,"max":36}},
                                     "font":{{"size":{font_size},"face":"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial"}},
                                     "borderWidth":1}},
                            "edges":{{"smooth":{{"type":"dynamic"}},"color":{{"opacity":0.45}}}},
                            "physics":{{"stabilization":{{"enabled":true,"iterations":220}},
                                       "barnesHut":{{"avoidOverlap":0.25,"springLength":140,"springConstant":0.03,"damping":0.18}},
                                       "minVelocity":0.75}}}}""")
    except Exception:
        pass

    html = net.generate_html(notebook=False)
    if freeze_layout:
        html = html.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\nnetwork.once('stabilizationIterationsDone', function(){ network.setOptions({ physics:false }); });"
        )
    return (html, legend_html)

def draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650, freeze_layout: bool = False, font_size: int = 16) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("networkx / pyvis が未導入のため、表のみ表示しています。"); return
    if edges.empty:
        st.warning("対象条件でエッジがありません。"); return
    html, legend_html = build_pyvis_html(edges, height_px=height_px, freeze_layout=freeze_layout, font_size=font_size)
    if not html:
        st.info("ネットワークを生成できませんでした。条件を見直してください。"); return
    st.components.v1.html(html, height=height_px, scrolling=True)
    if legend_html:
        st.markdown("**クラスタ凡例**&nbsp;&nbsp;" + legend_html, unsafe_allow_html=True)
    st.download_button("📥 ネットワークHTMLを保存", data=html.encode("utf-8"),
                       file_name="keyword_cooccurrence_network.html", mime="text/html",
                       key="dl_kw_pyvis_html_cached")

@st.cache_data(ttl=3600, show_spinner=False)
def compute_network_metrics(edges: pd.DataFrame) -> dict[str, float | dict[str, float]]:
    """
    ネットワークの指標を算出する
    - Density: 密度
    - Modularity: モジュラリティ
    - Average clustering: 平均クラスリング係数
    - Betweenness centrality: 媒介中心性
    """
    if edges is None or edges.empty or not HAS_NX:
        return {}

    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1.0)))

    if G.number_of_nodes() == 0:
        return {}

    metrics = {}

    # 1. Density
    metrics["density"] = nx.density(G)

    # 2. Modularity
    if HAS_COMMUNITY:
        try:
            comms = list(_greedy_comms(G, weight="weight"))
            metrics["modularity"] = _nx_modularity(G, comms, weight="weight")
        except Exception:
            metrics["modularity"] = 0.0

    # 3. Average clustering
    if HAS_ALGO:
        try:
            metrics["avg_clustering"] = _avg_clustering(G, weight="weight")
        except Exception:
            metrics["avg_clustering"] = 0.0

    # 4. Betweenness centrality
    if HAS_ALGO:
        try:
            # 重みを考慮した媒介中心性（距離として扱うため 1/weight とする）
            G_dist = G.copy()
            for u, v, d in G_dist.edges(data=True):
                w = d.get("weight", 1.0)
                d["distance"] = 1.0 / w if w > 0 else 1.0
            
            bc = _betweenness(G_dist, weight="distance", normalized=True)
            # 上位10件などの辞書として保持
            metrics["betweenness"] = dict(sorted(bc.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            metrics["betweenness"] = {}

    return metrics

def run_permutation_test(df_target: pd.DataFrame, n_perms: int = 100, min_edge: int = 3, top_n: int = 120):
    """
    指定されたデータフレームのネットワークに対するランダムモデル（帰無モデル）を生成し、
    ネットワーク指標の差の有意性を置換検定で算出する。
    """
    from .compute import build_keyword_cooccur_edges
    
    def _get_metrics(df):
        edges = build_keyword_cooccur_edges(df, min_edge)
        # topN 処理 (ui_cooccur.py のロジックを模倣)
        if not edges.empty and top_n > 0:
            deg = pd.concat([edges.groupby("src")["weight"].sum(), edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
            keep_nodes = set(deg.sort_values(ascending=False).head(int(top_n)).index.tolist())
            edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)
        return compute_network_metrics(edges)

    obs = _get_metrics(df_target)
    
    keys = ["density", "modularity", "avg_clustering"]
    obs_vals = {k: obs.get(k, 0.0) for k in keys}
    
    count_geq = {k: 0 for k in keys}
    count_leq = {k: 0 for k in keys}
    
    kws_lens = []
    all_kws = []
    if "_clean_keywords" in df_target.columns:
        # 重複を排除したリストを作成してから全て集める
        clean_lists = df_target["_clean_keywords"].apply(lambda x: list(set(x)) if isinstance(x, list) else [])
        kws_lens = clean_lists.apply(len).tolist()
        all_kws = clean_lists.explode().dropna().tolist()
    
    progress_bar = st.progress(0, text="ランダムモデルでの Permutation Test 実行中...")
    rand_vals_history = {k: [] for k in keys}
    
    for i in range(n_perms):
        if all_kws:
            random.shuffle(all_kws)
            new_kws = []
            idx = 0
            for L in kws_lens:
                chunk = all_kws[idx:idx+L]
                new_kws.append(list(set(chunk)))
                idx += L
                
            df_rand = df_target.copy()
            df_rand["_clean_keywords"] = pd.Series(new_kws, index=df_rand.index)
        else:
            df_rand = df_target.copy() # fallback
            
        m_rand = _get_metrics(df_rand)
        
        for k in keys:
            v_rand = m_rand.get(k, 0.0)
            rand_vals_history[k].append(v_rand)
            if v_rand >= obs_vals[k]:
                count_geq[k] += 1
            if v_rand <= obs_vals[k]:
                count_leq[k] += 1
        
        progress_bar.progress((i + 1) / n_perms, text=f"ランダムモデルでの Permutation Test 実行中... ({i+1}/{n_perms})")
    
    progress_bar.empty()
    
    p_values = {}
    for k in keys:
        p_geq = count_geq[k] / n_perms
        p_leq = count_leq[k] / n_perms
        p_values[k] = min(1.0, 2 * min(p_geq, p_leq)) # 両側検定
        
    random_means = {k: float(np.mean(rand_vals_history[k])) if rand_vals_history[k] else 0.0 for k in keys}
    
    return {
        "obs": obs,
        "random_means": random_means,
        "p_values": p_values
    }