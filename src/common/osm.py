import geopandas as gpd
import networkx as nx
import osmnx as ox

from typing import Tuple, List, Optional

def load_osm(place: str,
             network_type: str = "drive",
             custom_pois: Optional[List[str]] = None) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    
    """
    Baixa grafo viário e POIs do OSM.
      - network_type: 'drive', 'walk', ou 'all'
      - custom_pois: lista de chaves de amenity/landuse/tourism/shop (valores OSM) para coletar
    Retorna: (G, gdf_nodes, gdf_pois)
    """
    
    print(f"[OSM] Baixando rede: {place} ({network_type}) …")
    G = ox.graph_from_place(place, network_type=network_type, simplify=True)
    # G = ox.simplify_graph(G)

    # Nodes como GeoDataFrame
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # POIs: buscar por múltiplos tags
    default_tags = {
        "amenity": [
            "university",
            "school",
            
             ],

            #"hospital",
            #"clinic",
            #"police",
            #"fire_station",
            #"pharmacy",
            #"library",
            #"marketplace",
            #"bus_station",
            #"toilets",
            #"community_centre",
            #"townhall"
       
        #"leisure": ["park","pitch","sports_centre"],
        #"tourism": ["attraction","museum"],
        #"shop": ["supermarket","mall"]
    }
    if custom_pois:
        # Interpreta como uma lista de valores em 'amenity' + tenta mapear aos outros
        default_tags = {"amenity": list(set(default_tags["amenity"] + custom_pois))}

    print(f"[OSM] Baixando POIs com tags: {list(default_tags.keys())} …")
    gdf_pois = ox.features_from_place(place, tags=default_tags)
    
    # Apenas pontos/centroides
    if not all(gdf_pois.geometry.geom_type.isin(["Point","MultiPoint"])):
        gdf_pois = gdf_pois.copy()
        gdf_pois["geometry"] = gdf_pois.geometry.representative_point()

    # Limpeza mínima
    gdf_pois = gdf_pois.reset_index(drop=True)
    gdf_pois["poi_type"] = gdf_pois[["amenity"]].bfill(axis=1).iloc[:,0] 
    
    #,"leisure","tourism","shop"

    return G, gdf_nodes.to_crs(4326), gdf_pois.to_crs(4326)


'''
it is a coordinate reference system (CRS) transformation in GeoPandas.

Breaking it down

 - gdf_nodes / gdf_pois: GeoDataFrames containing geographic data (points, lines, polygons).

  .to_crs(...): Method to reproject the geometries to another CRS.

4326: The EPSG code for WGS 84 latitude/longitude coordinates (the standard used by GPS and most web maps, like OpenStreetMap, Google Maps).

Why it matters

When you work with geospatial data, every dataset has a CRS — basically a mathematical model describing how the 3D Earth is projected onto a flat coordinate system.
If two layers have different CRSs, operations like measuring distances or overlaying points and polygons may produce wrong results.

'''