import geopandas as gpd
from typing import Tuple, List, Optional


def load_ibge_polygons(path: Optional[str], layer: Optional[str]) -> Optional[gpd.GeoDataFrame]:
    if not path:
        return None
    print(f"[IBGE] Lendo {path} (layer={layer}) …")
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)