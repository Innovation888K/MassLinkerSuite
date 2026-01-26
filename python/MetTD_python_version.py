import joblib
import pandas as pd
from igraph import Graph, plot
import numpy as np
import torch
from collections import defaultdict
import random
from tqdm import tqdm
import copy
import threading
import multiprocessing as mp
import data
import matplotlib.pyplot as plt
import adjustText
from utils import plot_MetTD
from multiprocessing import Pool


def add_node_type(graph):
    temp = graph.vs['name']
    graph.vs["type"] = "e"
    for i in range(len(temp)):
        if '.' in temp[i]:
            continue
        else:
            graph.vs.find(name=temp[i])["type"] = "m"
    return graph


def calculate_node_scores(node_types, adj_matrix, fet_importance):
    adj_matrix_np = np.array(adj_matrix)
    scores = np.zeros(len(node_types))
    for i in range(len(node_types)):
        if node_types[i] == "e":
            mask = adj_matrix_np[i, :] != 0
            row_scores = np.zeros(len(fet_importance))
            row_scores[mask] = fet_importance[mask] / adj_matrix_np[i, mask]
            scores[i] = np.sum(row_scores)
        else:
            scores[i] = 0
    return scores


def single_worker(args):
    """单次permutation计算（优化版）"""
    seed, met_input, met_index, original_scores, node_types, adj_matrix = args
    np.random.seed(seed)
    random.seed(seed)
    shuffled_met = met_input.copy()
    random.shuffle(shuffled_met)
    fet_importance = np.array([shuffled_met[i] if i is not None else 0 for i in met_index])
    score = calculate_node_scores(node_types, adj_matrix, fet_importance)
    return (score > original_scores).astype(int)


def get_p(g, idx, permutation_time, met_index, adj_matrix, met_input, n_core=10):
    g.vs["fet_importance"] = [met_input[idx][i] if i is not None else 0 for i in met_index]
    g.vs["score"] = calculate_node_scores(g.vs["type"], adj_matrix, np.array(g.vs["fet_importance"]))
    original_scores = np.array(g.vs["score"])
    args_list = [
        (
            i,
            met_input[idx],
            met_index,
            original_scores,
            g.vs["type"],
            adj_matrix
        )
        for i in range(permutation_time)
    ]
    # for i in args_list:
    #     single_worker(i)
    with Pool(processes=n_core) as pool:
        results = list(tqdm(
            pool.imap(single_worker, args_list),
            total=permutation_time,
            desc="Permutation testing"
        ))
    p = np.zeros_like(original_scores, dtype=int)
    for result in results:
        p += result
    p_values = p / permutation_time
    e_mask = np.array(g.vs["type"]) == "e"
    e_p_values = p_values[e_mask]
    e_scores = original_scores[e_mask]
    e_names = np.array(g.vs["name"])[e_mask]
    return e_p_values, e_scores, e_names


# Single cpu version
# def get_p(g, idx, permutation_time, met_index):
#     g.vs["fet_importance"] = [met_input[idx][i] if i is not None else 0 for i in met_index]
#     g.vs["score"] = calculate_node_scores(g.vs["type"], adj_matrix, np.array(g.vs["fet_importance"]))
#     original_scores = np.array(g.vs["score"])
#     p = [0 for i in range(len(g.vs["score"]))]
#     for _ in tqdm(range(permutation_time)):
#         random.shuffle(met_input[idx])
#         g.vs["fet_importance"] = [met_input[idx][i] if i is not None else 0 for i in met_index]
#         score = calculate_node_scores(g.vs["type"], adj_matrix, np.array(g.vs["fet_importance"]))
#         p += (score > original_scores).astype(int)
#     p_values = np.array(p) / permutation_time
#     e_mask = np.array(g.vs["type"]) == "e"
#     e_p_values = p_values[e_mask]
#     e_scores = np.array(g.vs["score"])[e_mask]
#     e_names = np.array(g.vs["name"])[e_mask]
#     return e_p_values, e_scores, e_names


if __name__ == "__main__":
    network_info = pd.read_excel(r"links.xlsx")
    met_name = pd.read_csv("pathway_compound_detail.csv")["compound_names"].tolist()
    permutation_time = 1000
    g = Graph.DataFrame(
        network_info,
        directed=True,
        use_vids=False
    )
    adj_matrix = g.get_adjacency().data
    g = add_node_type(g)
    feature_importances = joblib.load("sample_feature_improtances.joblib")
    input_ori = [data.data_rev_transform(np.mean(abs(i), axis=1)) for i in feature_importances]
    met_input = [[np.mean(j) for j in i] for i in input_ori]
    met_index = [met_name.index(i) if i in met_name else None for i in g.vs['name']]
    idx = 0
    e_p_values, e_scores, e_names = get_p(g, idx, permutation_time, met_index, adj_matrix, met_input, n_core=20)
    # ###
    # e_p_values = joblib.load('e_p_value.joblib')
    # ###
    plot_MetTD(e_p_values, e_scores, e_names, permutation_time)
