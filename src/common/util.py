import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from typing import Tuple, List, Optional, Dict, Set
from pathlib import Path
from sklearn.neighbors import BallTree
from tqdm import tqdm
import folium

gtfs_stops_gdf = None
# ----------------------------
# Utilidades geoespaciais
# ----------------------------

def ensure_crs_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Projeta para UTM adequada via OSMnx (usa centroid para escolher zona)."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    try:
        utm = ox.projection.project_gdf(gdf)
        return utm
    except Exception:
        # Fallback: usa UTM 22S (SIRGAS 2000) que cobre Londrina/PR
        return gdf.to_crs(31982)


def buffer_meters(gdf: gpd.GeoDataFrame, meters: float) -> gpd.GeoDataFrame:
    gdf_utm = ensure_crs_utm(gdf)
    gdf_utm["geometry"] = gdf_utm.geometry.buffer(meters)
    return gdf_utm.to_crs(4326)


def haversine_series(points_a: gpd.GeoSeries, points_b: gpd.GeoSeries) -> np.ndarray:
    """Distância haversine entre pares (em metros). Ambos em WGS84."""
    # Vetorizado para arrays do mesmo tamanho
    R = 6371000.0
    lat1 = np.radians(points_a.y.values)
    lon1 = np.radians(points_a.x.values)
    lat2 = np.radians(points_b.y.values)
    lon2 = np.radians(points_b.x.values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c




# ----------------------------
# Saídas
# ----------------------------

def export_outputs(out_dir: str,
                   pipename: str,
                   candidates: gpd.GeoDataFrame,
                   chosen_ids: List[str],
                   universe: gpd.GeoDataFrame,
                   cover_sets: Dict[str, Set[str]],
                   radius_m: float,
                   place: str):
    
    out_dir = Path(out_dir) / pipename 
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in out_dir.iterdir():
        if f.is_file():
            f.unlink()

    selected = candidates[candidates["cand_id"].isin(chosen_ids)].copy()
    selected["selected"] = 1
    selected.to_file(out_dir/"selected_sensors.geojson", driver="GeoJSON")

    candidates.to_file(out_dir/"candidates.geojson", driver="GeoJSON")
    universe.to_file(out_dir/"universe.geojson", driver="GeoJSON")

    # Ranking por ganho marginal (na ordem da escolha)
    rows = []
    covered = set()
    for rank, cid in enumerate(chosen_ids, start=1):
        gain_set = cover_sets[cid] - covered
        rows.append({"rank": rank, "cand_id": cid, "marginal_gain": len(gain_set)})
        covered |= cover_sets[cid]
    pd.DataFrame(rows).to_csv(out_dir/"selection_ranking.csv", index=False)

    # Mapa Folium
    print("[Export] Gerando mapa Folium…")
    center = candidates.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")

    # Adiciona candidatos (cinzas)
    for _, r in candidates.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, tooltip=r.get("cand_id","cand")).add_to(m)

    # Adiciona selecionados (círculo de cobertura)
    sel = selected.copy()
    sel_buf = buffer_meters(sel[["geometry"]].copy(), radius_m)
    for _, r in sel.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6, tooltip=f"{r['cand_id']} ({r['source']})").add_to(m)
    for _, r in sel_buf.iterrows():
        folium.GeoJson(r.geometry.__geo_interface__, style_function=lambda x: {"fillOpacity":0.05, "weight":1}).add_to(m)

    m.save(str(out_dir/"sensor_plan_map.html"))
    print(f"[Export] Arquivos salvos em: {out_dir.resolve()}\n - selected_sensors.geojson\n - candidates.geojson\n - universe.geojson\n - selection_ranking.csv\n - sensor_plan_map.html")


def compute_centralities(G: nx.MultiDiGraph, k_sample: int = 600) -> pd.DataFrame:
    print("[Graph] Calculando centralidades (degree, betweenness~k, closeness)…")
    G_und = nx.Graph(G)  # não-direcionado para métricas globais
    deg = dict(G_und.degree())
    # Betweenness com amostragem de nós (k) para escalar
    btw = nx.betweenness_centrality(G_und, k=min(k_sample, G_und.number_of_nodes()), normalized=True, seed=42)
    clo = nx.closeness_centrality(G_und)
    df = pd.DataFrame({"node": list(deg.keys()),
                       "degree": list(deg.values()),
                       "betweenness": [btw.get(n,0.0) for n in deg.keys()],
                       "closeness": [clo.get(n,0.0) for n in deg.keys()]})
    return df


# ----------------------------
# Seleção de candidatos
# ----------------------------

def rank_candidates(gdf_nodes: gpd.GeoDataFrame,
                    centralities: pd.DataFrame,
                    gdf_pois: Optional[gpd.GeoDataFrame] = None,
                    gtfs_stops: Optional[gpd.GeoDataFrame] = None,
                    top_n_central: int = 10,
                    min_separation_m: int = 100) -> gpd.GeoDataFrame:
    """Cria candidatos a sensores: top-N por score + POIs + paradas GTFS.
    Faz deduplicação espacial (~min_separation_m) para evitar pontos muito próximos.
    """
    df = centralities.copy()
    # Score combinado simples (normalização min-max)
    for c in ["degree","betweenness","closeness"]:
        m, M = df[c].min(), df[c].max()
        df[c+"_n"] = 0.0 if M==m else (df[c]-m)/(M-m)
    df["score"] = 0.25*df["degree_n"] + 0.5*df["betweenness_n"] + 0.25*df["closeness_n"]
    df = df.sort_values("score", ascending=False).head(top_n_central)

    base = gdf_nodes.loc[df["node"].values].copy()
    base = base[["geometry"]].reset_index().rename(columns={"osmid":"node"})
    base["source"] = "centrality"

    frames = [base]
    if gdf_pois is not None and len(gdf_pois) > 0:
        p = gdf_pois[["geometry","poi_type"]].copy()
        p = p.rename(columns={"poi_type":"source"})
        frames.append(p)
    if gtfs_stops is not None and len(gtfs_stops) > 0:
        s = gtfs_stops[["geometry"]].copy()
        s["source"] = "gtfs_stop"
        frames.append(s)
    cand = pd.concat(frames, ignore_index=True)

    # Deduplicação por distância mínima
    cand = cand.dropna(subset=["geometry"]).reset_index(drop=True)
    cand = cand.set_crs(4326)
    cand_utm = ensure_crs_utm(cand)
    chosen_idx = []
    tree = None
    xy = np.vstack([cand_utm.geometry.x.values, cand_utm.geometry.y.values]).T
    tree = BallTree(xy, leaf_size=32, metric='euclidean')

    if tree is not None:
        taken = np.zeros(len(cand), dtype=bool)
        for i in range(len(cand)):
            if taken[i]:
                continue
            chosen_idx.append(i)
            # Marca vizinhos dentro de min_separation_m
            ind = tree.query_radius(xy[i:i+1], r=min_separation_m)[0]
            taken[ind] = True
    else:
        # Fallback O(n^2)
        coords = np.vstack([cand_utm.geometry.x.values, cand_utm.geometry.y.values]).T
        chosen_idx = []
        for i in range(len(cand)):
            if len(chosen_idx)==0:
                chosen_idx.append(i)
                continue
            pi = coords[i]
            too_close = False
            for j in chosen_idx:
                pj = coords[j]
                if np.linalg.norm(pi-pj) < min_separation_m:
                    too_close = True
                    break
            if not too_close:
                chosen_idx.append(i)

    cand = cand.iloc[chosen_idx].reset_index(drop=True)
    cand["cand_id"] = [f"c{i:05d}" for i in range(len(cand))]
    return cand

# ----------------------------
# Universo de cobertura e pesos
# ----------------------------

def build_universe(gdf_nodes: gpd.GeoDataFrame,
                   gdf_pois: Optional[gpd.GeoDataFrame],
                   ibge_polys: Optional[gpd.GeoDataFrame],
                   ibge_pop_field: Optional[str]) -> gpd.GeoDataFrame:
    items = []
    # Nós da malha (peso 1 por padrão)
    n = gdf_nodes[["geometry"]].copy().reset_index(drop=True)
    n["u_type"] = "road_node"
    n["weight"] = 1.0
    items.append(n)
    # POIs (peso 3 por padrão)
    if gdf_pois is not None and len(gdf_pois) > 0:
        p = gdf_pois[["geometry"]].copy().reset_index(drop=True)
        p["u_type"] = "poi"
        p["weight"] = 3.0
        items.append(p)
    # IBGE: usa centroid e peso = população se disponível
    if ibge_polys is not None and len(ibge_polys) > 0:
        ib = ibge_polys.copy()
        if not all(ib.geometry.geom_type.isin(["Polygon","MultiPolygon"])):
            ib = ib[ib.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()
        ib = ib.to_crs(4326)
        cent = ib.copy()
        cent["geometry"] = cent.geometry.representative_point()
        cent["u_type"] = "ibge_centroid"
        if ibge_pop_field and ibge_pop_field in cent.columns:
            # Evita zeros extremos
            w = cent[ibge_pop_field].astype(float).clip(lower=0)
            cent["weight"] = w.replace({np.nan: 0.0}).values
        else:
            cent["weight"] = 2.0
        items.append(cent[["geometry","u_type","weight"]])
    U = pd.concat(items, ignore_index=True) if items else gpd.GeoDataFrame(columns=["geometry","u_type","weight"], crs=4326)
    U["u_id"] = [f"u{i:06d}" for i in range(len(U))]
    return U.set_crs(4326)


def precompute_cover_sets(candidates: gpd.GeoDataFrame,
                          universe: gpd.GeoDataFrame,
                          radius_m: float) -> Dict[str, Set[str]]:
    print(f"[Cover] Pré-computando conjuntos de cobertura (raio={radius_m} m)…")
    # Calcula vizinhança por buffer em UTM para precisão
    cand_utm = ensure_crs_utm(candidates)
    uni_utm = ensure_crs_utm(universe)

    cover_sets: Dict[str, Set[str]] = {}
    # Índice espacial do universo
    uni_sindex = uni_utm.sindex

    for idx, crow in tqdm(cand_utm.iterrows(), total=len(cand_utm)):
        geom = crow.geometry
        buf = geom.buffer(radius_m)
        possible = list(uni_sindex.intersection(buf.bounds))
        if not possible:
            cover_sets[candidates.loc[idx, "cand_id"]] = set()
            continue
        hits = uni_utm.iloc[possible][uni_utm.iloc[possible].intersects(buf)]
        cover_sets[candidates.loc[idx, "cand_id"]] = set(hits.index.map(lambda i: universe.index[i]).astype(str))
    return cover_sets

'''
def export_plan(out_dir: Path, candidates: gpd.GeoDataFrame, chosen_idx: List[int],
                universe: gpd.GeoDataFrame, radius_m: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = candidates.iloc[chosen_idx].copy()
    sel["selected"] = 1
    candidates.to_file(out_dir/"candidates.geojson", driver="GeoJSON")
    sel.to_file(out_dir/"selected_sensors.geojson", driver="GeoJSON")
    universe.to_file(out_dir/"universe.geojson", driver="GeoJSON")

    # ranking com ganhos marginais (reexecuta para relatório)
    rows = []
    covered = set()
    for rnk, idx in enumerate(chosen_idx, start=1):
        # recomputa ganho marginal a partir de buffers (para relatório)
        # aqui só reportamos distância a itens do universo (aprox pelo precompute não reusado para simplicidade)
        rows.append({"rank": rnk, "cand_id": candidates.iloc[idx].get("cand_id", f"c{idx:05d}"), "note": "selecionado pelo GRL"})
    pd.DataFrame(rows).to_csv(out_dir/"selection_ranking_grl.csv", index=False)

    # Mapa folium
    print("[Export] Gerando mapa Folium…")
    center = candidates.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron")
    for _, r in candidates.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3).add_to(m)
    sel_buf = buffer_meters(sel[["geometry"]].copy(), radius_m)
    for _, r in sel.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6).add_to(m)
    for _, r in sel_buf.iterrows():
        folium.GeoJson(r.geometry.__geo_interface__, style_function=lambda x: {"fillOpacity":0.05, "weight":1}).add_to(m)
    m.save(str(out_dir/"sensor_plan_map_grl.html"))
'''