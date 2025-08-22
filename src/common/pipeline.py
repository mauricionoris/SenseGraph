from typing import Callable, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import wandb as wb

from common import osm, util
from algo import greedy, reinforce, a2c, env
cache = False


setup = 3


def pipeline(data: Any, steps: List[Callable]) -> Any:

    for step in steps:
        #print(str(step))
        data = step(data)
    return data

def is_dataframe(obj):
    if cache == False:
        return False
    return isinstance(obj, pd.DataFrame) and not obj.empty

#Etapas (comuns)
def step_load_osm(input):
    if input['G'] != None and cache == True:
        return input
    
    input['G'], input['gdf_nodes'] , input['gdf_pois'] = osm.load_osm(input['args']['place'], input['args']['network_type'], input['args']['custom_pois'])
    return input

def step_compute_centralities(input):

    if is_dataframe(input['cent']):
        return input

    input['cent'] = util.compute_centralities(input['G'], k_sample=input['args']['betw_k'])
    return input

def step_rank_candidates(input):

    if is_dataframe(input['candidates']):
        return input



    input['candidates'] = util.rank_candidates(gdf_nodes        =input['gdf_nodes']
                                            , centralities      =input['cent']
                                            , gdf_pois          =input['gdf_pois']
                                            , gtfs_stops        =util.gtfs_stops_gdf
                                            , top_n_central     =input['args']['top_n_central']
                                            , min_separation_m  =input['args']['min_sep_m'])

    return input

def step_build_universe(input):
    if is_dataframe(input['universe']):
        return input

    input['universe'] = util.build_universe(input['gdf_nodes'], input['gdf_pois'], None, None)

    return input  

def step_cover_set(input):

    if is_dataframe(input['cover_sets']):
        return input
   
    #print('candidates', input['candidates'])
    #print('universe', input['universe'])

    input['cover_sets'] = util.precompute_cover_sets(input['candidates'],  input['universe'] , input['args']['radius_m'])
    return input

def step_weights(input):
    if is_dataframe(input['weights']):
        return input
    input['weights'] = input['universe'].set_index(input['universe'].index.astype(str))["weight"]
    return input

def step_export(input):
    util.export_outputs( input['args']['out_dir'], input['pipename'], input['candidates'], input['chosen'][input['pipename']]
                       , input['universe'], input['cover_sets'], input['args']['radius_m'], input['args']['place'])
    return True


## Etapas específicas 

#Etapa (especifica egreedy)
def step_greedy_coverage(input):

    r = wb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sensegraphteam",
        # Set the wandb project where this run will be logged.
        project="sensegraph",
        name=f"Greedy-{setup}",
        # Track hyperparameters and run metadata.
        config={
            "budget": input['args']['k'],
            "number_of_candidates": len(input['candidates']),
        },
    )


    input['chosen'][input['pipename']], r = greedy.greedy_max_coverage(input['cover_sets'], input['weights'],  input['args']['k'], r)
    r.finish()
    return input

#Etapa (especifica reinforce)
def step_setup_train_reinforce(input):


    r = wb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sensegraphteam",
        # Set the wandb project where this run will be logged.
        project="sensegraph",
        name=f"reinforce-{setup}",
        # Track hyperparameters and run metadata.
        config={
            "budget": input['args']['k'],
            "number_of_candidates": len(input['candidates']),
        },
    )

    X_static, A_hat = reinforce.build_node_features(input['candidates'], input['cent'])  # X_static shape [N, F_static]
    exec_env = env.SensorPlacementEnv(input['candidates'], input['universe'], input['cover_sets'], input['args']['k'], X_static, input['weights'])

    policy, returns_hist, best, r = reinforce.train(exec_env
                                               , A_hat
                                               , episodes=input['args']['episodes']
                                               , lr=input['args']['lr']
                                               , hidden=input['args']['hidden']
                                               , gamma=input['args']['gamma']
                                               , seed=42
                                               , run=r)

    input['chosen'][input['pipename']] = reinforce.greedy_env_placement(exec_env)
    
    r.finish()
    
    return input 


#Etapa (especifica a2c)
def step_setup_train_a2c(input):



    r = wb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sensegraphteam",
        # Set the wandb project where this run will be logged.
        project="sensegraph",
        name=f"a2c-{setup}",
        # Track hyperparameters and run metadata.
        config={
            "budget": input['args']['k'],
            "number_of_candidates": len(input['candidates']),
        },
    )

    X_static, A_hat = reinforce.build_node_features(input['candidates'], input['cent'])  # X_static shape [N, F_static]
    exec_env = env.SensorPlacementEnv(input['candidates'], input['universe'], input['cover_sets'], input['args']['k'], X_static, input['weights'])

    policy, returns_hist, best, r = a2c.train(exec_env
                                               , A_hat
                                               , episodes=input['args']['episodes']
                                               , lr=input['args']['lr']
                                               , hidden=input['args']['hidden']
                                               , gamma=input['args']['gamma']
                                               , seed=42
                                               , run=r)

    input['chosen'][input['pipename']] = reinforce.greedy_env_placement(exec_env)
    r.finish()
    return input 

# Definindo a pipelines



# =============================
# Definição dos pipelines
# =============================

common_steps = [
    step_load_osm,
    step_compute_centralities,
    step_rank_candidates,
    step_build_universe,
    step_cover_set,
    step_weights,
]

pipelines = {
    'greedy':    [step_greedy_coverage, step_export],
    'reinforce': [step_setup_train_reinforce, step_export],
    'a2c':       [step_setup_train_a2c, step_export],
}

# =============================
# Executor
# =============================

def run_single_pipeline(pipename, base_result):
    """Executa apenas os steps específicos do pipeline, a partir do resultado comum."""
    i = dict(base_result)   # cópia para não compartilhar estado
    i['pipename'] = pipename
    print("Pipe Name:", pipename)
    result = pipeline(i, pipelines[pipename])
    print("Resultado final:", result)
    return pipename, result


def exec_mt(input_dict, max_workers=4):
    # 1. Executa steps comuns uma única vez
    base_result = pipeline(dict(input_dict), common_steps)

    # 2. Executa steps específicos em paralelo
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for p in pipelines.keys():
            futures.append(
                executor.submit(run_single_pipeline, p, base_result)
            )

        results = {}
        for f in as_completed(futures):
            pipename, result = f.result()
            results[pipename] = result

    return results





















'''
pipelines = {}

pipelines['greedy']      = [step_load_osm, step_compute_centralities, step_rank_candidates, step_build_universe, step_cover_set, step_weights, step_greedy_coverage, step_export]
pipelines['reinforce']   = [step_load_osm, step_compute_centralities, step_rank_candidates, step_build_universe, step_cover_set, step_weights, step_setup_train_reinforce, step_export]
pipelines['a2c']         = [step_load_osm, step_compute_centralities, step_rank_candidates, step_build_universe, step_cover_set, step_weights, step_setup_train_a2c, step_export]




def run_single_pipeline(pipename, input_dict):
    i = dict(input_dict)   # evita compartilhar o mesmo dicionário entre threads
    i['pipename'] = pipename
    print("Pipe Name:", pipename)
    result = pipeline(i, pipelines[pipename])
    print("Resultado final:", result)
    return pipename, result

def exec_mt(input_dict, max_workers=4):
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for p in pipelines.keys():
            futures.append(
                executor.submit(run_single_pipeline, p, input_dict)
            )

        results = {}
        for f in as_completed(futures):
            pipename, result = f.result()
            results[pipename] = result

    return results




single thread
def exec(input):
    for p in pipelines.keys():
        i = input
        print("Pipe Name:", p)
        i['pipename'] = p
        # Executando
        result = pipeline(i, pipelines[p])
        print("Resultado final:", result)
'''



'''

                          
                          #TODO > Outros pipelines
#pipelines['a2c']         = [step1, step2, step5]

input = {  'pipename': None
         , 'x':5    
         , 'args': {'place': None, 'network_type': None, 'custom_pois': None
                     , 'top_n_central': None, 'min_sep_m': None, 'radius_m': None, 'k': None
                     , 'episodes': None, 'lr': None, 'hidden': None, 'gamma': None}
         , 'G': None
         , 'gdf_nodes': None
         , 'gdf_pois': None
         , 'ibge': None
         , 'gtfs': None
         , 'cent': None
         , 'candidates': None
         , 'universe': None
         , 'cover_sets': None
         , 'weights': None
         , 'chosen': {'greedy':None, 'reinfoce': None, 'a2c': None}
         , 'output': None
        }
                  
'''
