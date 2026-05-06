import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from transformer import transformer_language
from data import ExcelDataset
from tqdm import tqdm
import pandas as pd
from matplotlib.gridspec import GridSpec
import MetTD_python_version as MetTD
import utils
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
from matplotlib.colors import LinearSegmentedColormap


def plot_time_step_bar_chart(shap_values, n_test, top_n_time_steps, feature_labels=None, stacked=True, colors=None,
                             save_path=None):
    total_rows, n_features, n_classes = shap_values.shape
    seq_len = total_rows // n_test

    shap_reshaped = shap_values.reshape(n_test, seq_len, n_features, n_classes)
    importance_by_class = np.sum(np.abs(shap_reshaped), axis=2)
    mean_importance_by_class = np.mean(importance_by_class, axis=0)
    total_importance = np.sum(mean_importance_by_class, axis=1)
    order_index = np.argsort(total_importance)
    top_n_indices = order_index[-top_n_time_steps:].tolist()

    plot_data = mean_importance_by_class[top_n_indices, :]

    if feature_labels is not None:
        x_labels = [feature_labels[i] for i in top_n_indices]
    else:
        x_labels = [str(i) for i in top_n_indices]
    class_labels = [f'Class {i}' for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_width = 0.8 / n_classes if not stacked else 0.8
    plot_colors = colors if colors and len(colors) >= n_classes else [None] * n_classes

    if stacked:
        bottom = np.zeros(top_n_time_steps)
        for class_idx in range(n_classes):
            ax.bar(np.arange(top_n_time_steps), plot_data[:, class_idx],
                   width=0.8,
                   bottom=bottom,
                   label=class_labels[class_idx],
                   color=plot_colors[class_idx],
                   alpha=0.8)
            bottom += plot_data[:, class_idx]
        ax.set_title(f'Top {top_n_time_steps} Time Steps: Stacked Bar Chart of SHAP Importance by Class')
    else:
        x = np.arange(top_n_time_steps)
        for class_idx in range(n_classes):
            ax.bar(x + class_idx * bar_width - (n_classes - 1) * bar_width / 2,
                   plot_data[:, class_idx],
                   width=bar_width,
                   label=class_labels[class_idx],
                   color=plot_colors[class_idx],
                   alpha=0.8)
        ax.set_title(f'Top {top_n_time_steps} Time Steps: Grouped Bar Chart of SHAP Importance by Class')

    ax.set_xticks(np.arange(top_n_time_steps))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel('Original Sequence Index (Time Step)')
    ax.set_ylabel('Average Total Absolute SHAP Value (Sum of 60 Features)')
    ax.legend(title='Output Class', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    return order_index

def plot_aggregated_time_importance(shap_values, n_test, top_n_time_steps, feature_labels=None, colors=None,
                                    save_path=None):
    total_rows, n_features, n_classes = shap_values.shape
    seq_len = total_rows // n_test
    shap_reshaped = shap_values.reshape(n_test, seq_len, n_features, n_classes)
    importance_by_class = np.sum(np.abs(shap_reshaped), axis=2)
    mean_importance_by_class = np.mean(importance_by_class, axis=0)
    total_importance = np.sum(mean_importance_by_class, axis=1)
    order_index = np.argsort(total_importance)
    top_n_indices = order_index[-top_n_time_steps:].tolist()

    plot_data = mean_importance_by_class[top_n_indices, :]

    if feature_labels is not None:
        x_labels = [feature_labels[i] for i in top_n_indices]
    else:
        x_labels = [str(i) for i in top_n_indices]
    fig, ax = plt.subplots(figsize=(14, 6))
    class_labels = [f'Class {i}' for i in range(n_classes)]
    stack_colors = colors if colors and len(colors) >= n_classes else None
    ax.stackplot(np.arange(top_n_time_steps), plot_data.T,
                 labels=class_labels,
                 colors=stack_colors, 
                 alpha=0.8)
    ax.set_xticks(np.arange(top_n_time_steps))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title(f'Top {top_n_time_steps} Time Steps: Stacked SHAP Importance by Class')
    ax.set_xlabel('Original Sequence Index (Time Step)')
    ax.set_ylabel('Total Average Absolute SHAP Value (Sum of 60 Features)')
    ax.legend(title='Output Class', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')
    else:
        plt.show()
    return order_index
def plot_top_n_feature_shap_2d(shap_values, n_test, top_n_time_steps, target_class_index, feature_labels=None,
                               save_path=None, mode='ML', cmap=None):
    if mode == 'ML':
        data_raw = shap_values[:, target_class_index]
    else:
        data_raw = shap_values[:, :, target_class_index]
    total_rows = data_raw.shape[0]
    seq_len = total_rows // n_test
    if mode == 'ML':
        shap_reshaped = data_raw.reshape(n_test, seq_len // 60, 60)
    else:
        shap_reshaped = data_raw.reshape(n_test, seq_len, 60)
    NUM_MAIN_FEATURES = 3
    PARAMS_PER_FEATURE = 20
    main_feature_names = [f'Parameter of {i}' for i in ['Height', 'Position', 'Width']]
    y_labels = [str(i) for i in range(PARAMS_PER_FEATURE)]
    all_plot_shap = []
    all_feature_labels = []
    for test_time in range(n_test):
        shap_data = shap_reshaped[test_time, :, :]
        shap_abs = abs(shap_data)
        shap_sum = np.sum(shap_abs, axis=1)
        order_index = np.argsort(shap_sum)
        top_n_indices = order_index[-top_n_time_steps:][::-1]
        plot_index = top_n_indices.tolist()
        plot_shap = shap_data[plot_index]
        feature_label = [feature_labels[i] for i in plot_index]
        all_plot_shap.append(plot_shap)
        all_feature_labels.append(feature_label)
    mean_shap_raw = np.mean(shap_reshaped, axis=0)
    mean_shap_abs = np.abs(mean_shap_raw)
    mean_shap_sum = np.sum(mean_shap_abs, axis=1)
    mean_order_index = np.argsort(mean_shap_sum)
    mean_top_n_indices = mean_order_index[-top_n_time_steps:][::-1]
    mean_plot_index = mean_top_n_indices.tolist()
    mean_plot_shap = mean_shap_raw[mean_plot_index]
    mean_feature_label = [feature_labels[i] for i in mean_plot_index]
    all_plot_shap.append(mean_plot_shap)
    all_feature_labels.append(mean_feature_label)
    num_rows = n_test + 1
    fig = plt.figure(figsize=(15, 3.5 * num_rows))
    gs = GridSpec(num_rows, 6, width_ratios=[10, 1, 10, 1, 10, 1],
                  wspace=0.1, hspace=0.3)
    for row_idx in range(num_rows):
        plot_shap = all_plot_shap[row_idx]
        feature_label = all_feature_labels[row_idx]
        if row_idx < n_test:
            row_title = f'Sample {row_idx + 1} ({top_n_time_steps} Time Steps)'
        else:
            row_title = f'AVERAGE ({top_n_time_steps} Time Steps)'
        for col_idx in range(NUM_MAIN_FEATURES):
            plot_ax = fig.add_subplot(gs[row_idx, 2 * col_idx])
            start = col_idx * PARAMS_PER_FEATURE
            end = (col_idx + 1) * PARAMS_PER_FEATURE
            data_20 = plot_shap[:, start:end]
            sub_max_abs = np.max(np.abs(data_20))
            vmin, vmax = -sub_max_abs, sub_max_abs
            im = plot_ax.imshow(data_20.T, aspect='auto', cmap=cmap if cmap else 'seismic',
                                vmin=vmin, vmax=vmax, interpolation='nearest')
            cbar_ax = fig.add_subplot(gs[row_idx, 2 * col_idx + 1])
            cbar_label = f'SHAP (±{sub_max_abs:.2f})'
            fig.colorbar(im, cax=cbar_ax, label=cbar_label)
            if col_idx == 0:
                plot_ax.set_yticks(np.arange(PARAMS_PER_FEATURE))
                plot_ax.set_yticklabels(y_labels)
                plot_ax.set_ylabel(row_title, rotation=90, labelpad=10, fontsize=10, fontweight='bold')
            else:
                plot_ax.set_yticks([])
                plot_ax.set_yticklabels([])
            plot_ax.set_xticks(np.arange(top_n_time_steps))
            plot_ax.set_xticklabels(feature_label, rotation=90, ha='right')
            if row_idx == 0:
                plot_ax.set_title(main_feature_names[col_idx], fontsize=12)
            if row_idx == num_rows - 1:
                plot_ax.set_xlabel('Original Sequence Index (Time Step)')
                plot_ax.text(0.5, -0.9, main_feature_names[col_idx], ha='center', transform=plot_ax.transAxes,
                             fontsize=24, fontweight='bold')
            else:
                plot_ax.set_xlabel('')
    plt.suptitle(f'Top {top_n_time_steps} Important Time Steps Analysis (Class {target_class_index})', y=0.99,
                 fontsize=16)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()
def run_shap_analysis(model_path, data_path, device, n_background=50, n_test=3):
    dataset = joblib.load(data_path)
    x_raw = dataset.samples
    x_tensor = torch.tensor(x_raw, dtype=torch.float32).reshape(x_raw.shape[0], 5194, 60)
    total_idx = np.arange(len(x_tensor))
    bg_idx = np.random.choice(total_idx, n_background, replace=False)
    test_idx = np.random.choice(total_idx, n_test, replace=False)
    background = x_tensor[bg_idx].to(device)
    test_samples = x_tensor[test_idx]
    if hasattr(dataset, 'classes'):
        num_classes = len(set(dataset.classes))
    else:
        y = getattr(dataset, 'is_positive', None)
        num_classes = len(np.unique(y)) if y is not None else 2
    model = transformer_language(class_count=num_classes, embed_dim=512, n_heads=8, depth=8)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    del checkpoint
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    explainer = shap.DeepExplainer(model, background)
    all_shap_values = []
    for i in tqdm(range(len(test_samples)), desc=f"Calculating SHAP for {n_test} test samples"):
        single_test_sample = test_samples[i].unsqueeze(0).to(device)
        shap_values_for_sample = explainer.shap_values(single_test_sample)
        all_shap_values.append(shap_values_for_sample)
    if not all_shap_values:
        return [], test_samples.cpu().numpy(), num_classes
    num_classes_actual = len(all_shap_values[0])
    final_shap_values = []
    for c in range(num_classes_actual):
        class_shap_values = np.concatenate([all_shap_values[s][c] for s in range(len(all_shap_values))], axis=0)
        final_shap_values.append(class_shap_values)
    return final_shap_values, test_samples.cpu().numpy(), num_classes
