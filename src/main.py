import logging
import argparse
from pathlib import Path

from common import pipeline2 as pipeline


logger = logging.getLogger(__name__)
logging.basicConfig(filename='paper_1.log', encoding='utf-8', level=logging.INFO)

#local = 'Universidade Estadual de Londrina, Londrina, Brazil'
local = 'Londrina, Paraná, Brasil'
#local = 'Maringa, Paraná, Brazil'


def parse_and_setup():
    parser = argparse.ArgumentParser(description="Pipeline de grafos + posicionamento de sensores (máxima cobertura)")
    parser.add_argument('--place',          type=str,   default=local,                                          help='Consulta OSMnx para a área de estudo') 
    parser.add_argument('--network_type',   type=str,   default='drive', choices=['drive','walk','all'],        help='Tipo de rede viária do OSM')
    parser.add_argument('--pois',           type=str,   default='',                                             help='Lista de POIs (amenity/leisure/shop/tourism), separados por vírgula') 
    parser.add_argument('--ibge_path',      type=str,   default='./BR_Pais_2024.shp',                           help='Caminho para GPKG/SHP do IBGE (setores/bairros)')
    parser.add_argument('--ibge_layer',     type=str,   default=None,                                           help='Nome da layer (se GPKG)')
    parser.add_argument('--ibge_pop_field', type=str,   default=None,                                           help='Campo de população para pesos (opcional)')
    parser.add_argument('--gtfs_dir',       type=str,   default=None,                                           help='Diretório com arquivos GTFS')
    parser.add_argument('--k',              type=int,   default=100,                                             help='Orçamento de sensores (k)')
    parser.add_argument('--radius_m',       type=float, default=50.0,                                          help='Raio de cobertura em metros')
    parser.add_argument('--top_n_central',  type=int,   default=250,                                            help='Top-N nós por score de centralidade para virar candidato')
    parser.add_argument('--min_sep_m',      type=int,   default=20,                                             help='Separação mínima entre candidatos (m)')
    parser.add_argument('--betw_k',         type=int,   default=400,                                            help='Amostras k para betweenness (trade-off velocidade/qualidade)')
    parser.add_argument('--out_dir',        type=str,   default='./outputs',                                    help='Diretório de saída')
    parser.add_argument('--episodes',       type=int,   default=200)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--hidden',         type=int,   default=64)
    parser.add_argument('--gamma',          type=float, default=0.90)
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    # monta o dicionário conforme sua estrutura
    input_dict = {
        'pipename': None,
        'x': 5,
        'args': {
            'place': args.place,
            'network_type': args.network_type,
            'custom_pois': args.pois.split(",") if args.pois else [],
            'top_n_central': args.top_n_central,
            'min_sep_m': args.min_sep_m,
            'radius_m': args.radius_m,
            'k': args.k,
            'betw_k': args.betw_k,
            'episodes': args.episodes,
            'lr': args.lr,
            'hidden': args.hidden,
            'gamma': args.gamma,
            'out_dir': args.out_dir
        },
        'G': None,
        'gdf_nodes': None,
        'gdf_pois': None,
        'ibge': None,
        'gtfs': None,
        'cent': None,
        'candidates': None,
        'universe': None,
        'cover_sets': None,
        'weights': None,
        'chosen': {'greedy': None, 'reinfoce': None, 'a2c': None},
        'output': None
    }

    return input_dict



def main():

    data = parse_and_setup()
    #pipeline.exec(data) # versão single thread

    all_results = pipeline.exec_with_baseline(data, max_workers=8)
    #print(all_results)

main()




