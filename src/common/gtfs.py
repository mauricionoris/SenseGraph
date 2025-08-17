
import geopandas as gpd
import pandas as pd 

from pathlib import Path

from typing import Tuple, List, Optional, Dict
from shapely.geometry import Point, LineString, Polygon



def load_gtfs(gtfs_dir: Optional[str]) -> Optional[Dict[str, pd.DataFrame]]:
    if not gtfs_dir:
        return None
    gtfs_dir = Path(gtfs_dir)
    print(f"[GTFS] Lendo {gtfs_dir} …")
    req = ["stops.txt","trips.txt","stop_times.txt"]
    files = {f: gtfs_dir/f for f in req}
    if not all(fp.exists() for fp in files.values()):
        print("[GTFS] Arquivos obrigatórios não encontrados, ignorando GTFS…")
        return None
    data = {k: pd.read_csv(v) for k,v in files.items()}
    return data


def gtfs_to_gdf_nodes_edges(gtfs: Dict[str, pd.DataFrame]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Converte GTFS stops + sequências de viagem em pontos e linhas (aproximação).
    Cria arestas ligando paradas consecutivas por trip_id (ordem por stop_sequence).
    """
    stops = gtfs["stops.txt"].copy()
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
        crs=4326
    )
    # Conecta pares consecutivos de cada viagem
    st = gtfs["stop_times.txt"][["trip_id","stop_id","stop_sequence"]].copy()
    st = st.sort_values(["trip_id","stop_sequence"])  
    pairs = st.groupby("trip_id").apply(lambda df: list(zip(df.stop_id.values[:-1], df.stop_id.values[1:])))
    pairs = [p for sub in pairs for p in sub]
    # Cria linhas simples (reta entre pontos)
    stop_lookup = stops_gdf.set_index("stop_id").geometry
    rows = []
    for a,b in pairs:
        if a in stop_lookup.index and b in stop_lookup.index:
            rows.append({
                "u": a,
                "v": b,
                "geometry": LineString([stop_lookup[a], stop_lookup[b]])
            })
    edges_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return stops_gdf, edges_gdf

