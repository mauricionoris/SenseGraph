from typing import Callable, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import wandb as wb

from common import osm, util
from algo import greedy, reinforce, a2c, dqn, sac
from algo import ppo2 as ppo

from algo import env2 as env

cache = False


setup = 1


def pipeline(data: Any, steps: List[Callable]) -> Any:
    for step in steps:
        data = step(data)
    return data


def is_dataframe(obj):
    if cache is False:
        return False
    return isinstance(obj, pd.DataFrame) and not obj.empty


# =============================
# Etapas comuns
# =============================

def step_load_osm(input):
    if input['G'] is not None and cache:
        return input
    input['G'], input['gdf_nodes'], input['gdf_pois'] = osm.load_osm(
        input['args']['place'],
        input['args']['network_type'],
        input['args']['custom_pois']
    )
    return input


def step_compute_centralities(input):
    if is_dataframe(input['cent']):
        return input
    input['cent'] = util.compute_centralities(input['G'], k_sample=input['args']['betw_k'])
    return input


def step_rank_candidates(input):
    if is_dataframe(input['candidates']):
        return input
    input['candidates'] = util.rank_candidates(
        gdf_nodes=input['gdf_nodes'],
        centralities=input['cent'],
        gdf_pois=input['gdf_pois'],
        gtfs_stops=util.gtfs_stops_gdf,
        top_n_central=input['args']['top_n_central'],
        min_separation_m=input['args']['min_sep_m']
    )
    return input


def step_build_universe(input):
    if is_dataframe(input['universe']):
        return input
    input['universe'] = util.build_universe(input['gdf_nodes'], input['gdf_pois'], None, None)
    return input


def step_cover_set(input):
    if is_dataframe(input['cover_sets']):
        return input
    input['cover_sets'] = util.precompute_cover_sets(input['candidates'], input['universe'], input['args']['radius_m'])
    return input


def step_weights(input):
    if is_dataframe(input['weights']):
        return input
    input['weights'] = input['universe'].set_index(input['universe'].index.astype(str))["weight"]
    return input


def step_export(input):
    util.export_outputs(
        input['args']['out_dir'],
        input['pipename'],
        input['candidates'],
        input['chosen'][input['pipename']],
        input['universe'],
        input['cover_sets'],
        input['args']['radius_m'],
        input['args']['place']
    )
    return True


# =============================
# Etapas específicas
# =============================

def step_greedy_coverage(input):
    r = wb.init(
        entity="sensegraphteam",
        project="sensegraph",
        name=f"Greedy-{setup}",
        config={
            "budget": input['args']['k'],
            "number_of_candidates": len(input['candidates']),
        },
    )
    input['chosen'][input['pipename']], r = greedy.greedy_max_coverage(
        input['cover_sets'], input['weights'], input['args']['k'], r
    )
    r.finish()
    # Guardar baseline para RLs
    # converter baseline em índices inteiros
    baseline_ids = input['chosen'][input['pipename']]
    id_to_idx = {cid: idx for idx, cid in enumerate(input['candidates']['cand_id'])}
    baseline_idx = [id_to_idx[cid] for cid in baseline_ids if cid in id_to_idx]

    input['baseline'] = baseline_idx
    return input


# Função auxiliar para treinar RLs

def _train_rl(input, algo, name):
    r = wb.init(
        entity="sensegraphteam",
        project="sensegraph",
        name=f"{name}-{setup}",
        config={
            "budget": input['args']['k'],
            "number_of_candidates": len(input['candidates']),
        },
    )

    X_static, A_hat = reinforce.build_node_features(input['candidates'], input['cent'])
    exec_env = env.SensorPlacementEnv(
        input['candidates'], input['universe'], input['cover_sets'],
        input['args']['k'], X_static, input['weights'],
        initial_solution=input.get('baseline')  # passa baseline
    )

    policy, returns_hist, best, r = algo.train(
        exec_env, A_hat,
        episodes=input['args']['episodes'],
        lr=input['args']['lr'],
        hidden=input['args']['hidden'],
        gamma=input['args']['gamma'],
        seed=42,
        run=r
    )

    cand_ids = list(input['candidates']['cand_id'])
    chosen_ids = [cand_ids[idx] for idx in exec_env.selected]

    input['chosen'][input['pipename']] = chosen_ids

    #input['chosen'][input['pipename']] = exec_env.selected #input['chosen'][input['pipename']] = reinforce.greedy_env_placement(exec_env)
    r.finish()
    return input


def step_setup_train_reinforce(input):
    return _train_rl(input, reinforce, "reinforce")


def step_setup_train_a2c(input):
    return _train_rl(input, a2c, "a2c")


def step_setup_train_dqn(input):
    return _train_rl(input, dqn, "dqn")


def step_setup_train_sac(input):
    return _train_rl(input, sac, "sac")


def step_setup_train_ppo(input):
    return _train_rl(input, ppo, "ppo")


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

baseline_pipeline = common_steps + [step_greedy_coverage]

rl_pipelines = {
    'reinforce': [step_setup_train_reinforce, step_export],
    'a2c':       [step_setup_train_a2c, step_export],
    'dqn':       [step_setup_train_dqn, step_export],
    'sac':       [step_setup_train_sac, step_export],
    'ppo':       [step_setup_train_ppo, step_export],
}


# =============================
# Executor
# =============================

def run_single_pipeline(pipename, base_result):
    i = dict(base_result)
    i['pipename'] = pipename
    print("Pipe Name:", pipename)
    result = pipeline(i, rl_pipelines[pipename])
    #print("Resultado final:", result)
    return pipename, result


def exec_with_baseline(input_dict, max_workers=4):
    # 1. Executa baseline greedy uma vez
    baseline_result = pipeline(dict(input_dict), baseline_pipeline)

    # 2. Executa RL pipelines a partir do baseline
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for p in rl_pipelines.keys():
            futures.append(executor.submit(run_single_pipeline, p, baseline_result))

        results = {}
        for f in as_completed(futures):
            pipename, result = f.result()
            results[pipename] = result

    return {"baseline": baseline_result, "rl_results": results}
