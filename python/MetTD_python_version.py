"""
Python implementation of MetTD-style topology-based enzyme scoring.

This module provides graph-based utilities for propagating MassLinker-derived
metabolite feature importance values to enzyme nodes in a metabolic network.
The workflow assigns node types, calculates topology-weighted enzyme scores,
and estimates empirical p-values through permutation testing.

In the graph representation, metabolite nodes and enzyme nodes are distinguished
by their names. Nodes whose names contain a period are treated as enzyme nodes,
whereas other nodes are treated as metabolite nodes. Feature importance values
from MassLinker downstream analyses, such as SHAP-based interpretability or
distance-based differential analysis, can be mapped onto metabolite nodes and
then aggregated to enzyme-level topology scores.
"""

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
    """
    Assign metabolite or enzyme node types to graph vertices.

    Nodes whose names contain a period are treated as enzyme nodes and assigned
    type `e`. All other nodes are treated as metabolite nodes and assigned type
    `m`.

    Parameters
    ----------
    graph : igraph.Graph
        Metabolic network graph containing vertex names.

    Returns
    -------
    igraph.Graph
        The same graph object with a vertex attribute named `type`.
    """
    temp = graph.vs['name']

    # Initialize all nodes as enzyme nodes.
    graph.vs["type"] = "e"

    for i in range(len(temp)):
        # Node names containing a period are treated as enzyme identifiers.
        if '.' in temp[i]:
            continue
        else:
            graph.vs.find(name=temp[i])["type"] = "m"

    return graph


def calculate_node_scores(node_types, adj_matrix, fet_importance):
    """
    Calculate topology-weighted scores for enzyme nodes.

    For each enzyme node, the score is computed from neighboring metabolite
    feature importance values. Each connected feature importance value is
    weighted by the corresponding adjacency-matrix entry.

    Metabolite nodes receive a score of zero because this function is designed
    to aggregate metabolite-level MassLinker importance values to enzyme-level
    scores.

    Parameters
    ----------
    node_types : list
        Node type labels, where `e` denotes enzyme nodes and `m` denotes
        metabolite nodes.
    adj_matrix : array-like
        Graph adjacency matrix or topology-weight matrix.
    fet_importance : np.ndarray
        Feature importance values mapped to graph nodes.

    Returns
    -------
    np.ndarray
        Node scores with nonzero values assigned to enzyme nodes.
    """
    adj_matrix_np = np.array(adj_matrix)
    scores = np.zeros(len(node_types))

    for i in range(len(node_types)):
        if node_types[i] == "e":
            mask = adj_matrix_np[i, :] != 0
            row_scores = np.zeros(len(fet_importance))

            # Divide feature importance by the corresponding topology weight.
            row_scores[mask] = fet_importance[mask] / adj_matrix_np[i, mask]
            scores[i] = np.sum(row_scores)
        else:
            scores[i] = 0

    return scores


def single_worker(args):
    """
    Run one permutation replicate for empirical significance testing.

    The metabolite-level feature importance values are shuffled, remapped to
    graph nodes, converted into enzyme scores, and compared with the original
    enzyme scores.

    Parameters
    ----------
    args : tuple
        Tuple containing seed, met_input, met_index, original_scores,
        node_types, and adj_matrix.

    Returns
    -------
    np.ndarray
        Binary array indicating whether each permuted score is greater than
        the corresponding original score.
    """
    seed, met_input, met_index, original_scores, node_types, adj_matrix = args

    np.random.seed(seed)
    random.seed(seed)

    shuffled_met = met_input.copy()
    random.shuffle(shuffled_met)

    # Map shuffled metabolite feature importance values to graph node order.
    fet_importance = np.array([shuffled_met[i] if i is not None else 0 for i in met_index])

    score = calculate_node_scores(node_types, adj_matrix, fet_importance)

    return (score > original_scores).astype(int)


def get_p(g, idx, permutation_time, met_index, adj_matrix, met_input, n_core=10):
    """
    Estimate empirical p-values for topology-weighted enzyme scores.

    The function first maps observed MassLinker-derived metabolite importance
    values onto graph nodes and calculates original enzyme scores. It then
    performs permutation testing by repeatedly shuffling metabolite importance
    values and recalculating enzyme scores in parallel.

    The empirical p-value for each enzyme node is the proportion of permuted
    scores greater than the observed score.

    Parameters
    ----------
    g : igraph.Graph
        Metabolic network graph with vertex attributes including `type`.
    idx : int
        Index selecting one metabolite-importance vector from `met_input`.
    permutation_time : int
        Number of permutation replicates.
    met_index : list
        Mapping from graph node order to metabolite-feature index. Entries may
        be None for nodes without mapped metabolite features.
    adj_matrix : array-like
        Graph adjacency matrix or topology-weight matrix.
    met_input : list or np.ndarray
        Collection of MassLinker-derived metabolite importance vectors.
    n_core : int, optional
        Number of worker processes used for permutation testing.

    Returns
    -------
    tuple
        e_p_values : np.ndarray
            Empirical p-values for enzyme nodes.
        e_scores : np.ndarray
            Original topology-weighted scores for enzyme nodes.
        e_names : np.ndarray
            Names of enzyme nodes.
    """
    # Map observed metabolite feature importance values onto graph nodes.
    g.vs["fet_importance"] = [met_input[idx][i] if i is not None else 0 for i in met_index]

    # Calculate original topology-weighted node scores.
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

    # Run permutation testing in parallel.
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

    # Return results only for enzyme nodes.
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
