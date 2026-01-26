import os.path
import joblib
from transformer_shap import plot_top_n_feature_shap_2d
from ML_tools import addition_roc
from data import ExcelDataset, load_data, data_transform, gen_feature_names
import ML_tools
import os
import pandas as pd
import copy
import torch
from tqdm import tqdm
from utils import visualization_foldn_valid
import pathway_mask
from utils import metabo_dis, get_diff, kegg_enrichment, plot_enrichment, plot_peak_comp, plot_2d, plot_p_value
import joblib
import numpy as np
from transformer import transformer_language
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

def MassLinker_Anal(dataset_path, sele_sample=None, load_JS=None, load_was=None, use_JS=False, plot_peak_comp_dir=None,
                    plot_peak_topn=100, path_way_detail_file="pathway_compound_detail.csv", work_dir='./', save=True,
                    p_limit=0.05):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    dataset = joblib.load(dataset_path)
    if sele_sample is None:
        sele_sample = [i for i in range(len(dataset))]
    group = [dataset.is_positive[i].numpy().item() for i in sele_sample]
    params = [dataset.samples[i] for i in sele_sample]
    if load_JS is not None and load_was is not None:
        JS = joblib.load(load_JS)
        was = joblib.load(load_was)
    else:
        JS, was = metabo_dis(params)
    if save:
        joblib.dump(JS, os.path.join(work_dir, "JS.joblib"))
        joblib.dump(was, os.path.join(work_dir, "was.joblib"))
    was_arr = np.array(was)[0]
    JS_arr = np.array(JS)[0]
    if use_JS:
        p_values, diff = get_diff(JS_arr, group)
    else:
        p_values, diff = get_diff(was_arr, group)
    if save:
        joblib.dump(p_values, os.path.join(work_dir, "p_values.joblib"))
    met_name = pd.read_csv(path_way_detail_file)["compound_names"].tolist()
    mzs = pd.read_csv(path_way_detail_file)["mz"].tolist()
    plot_p_value(p_values, met_name, mzs=mzs, n=20, save_path=os.path.join(work_dir, "p_value.pdf"))
    if plot_peak_comp_dir is not None:
        plot_peak_comp(p_values, group, params, met_name, mzs, save_path=plot_peak_comp_dir, top_n=plot_peak_topn)
    pathway_df_full = pd.read_csv(path_way_detail_file, index_col=0)
    kegg_enriched = kegg_enrichment(pathway_df_full, p_values, diff)
    if save:
        joblib.dump(kegg_enriched, os.path.join(work_dir, "kegg_enriched.joblib"))
    plot_df = kegg_enriched.copy()
    plot_enrichment(plot_df, save_name=os.path.join(work_dir, "KEGG_Enriched1.pdf"))
    was1 = [[] for i in range(len(was[0]))]
    JS1 = [[] for i in range(len(was[0]))]
    for i in range(len(p_values)):
        if p_values[i] <= p_limit:
            for j in range(len(was[0])):
                was1[j].append(was[0][j][i])
                JS1[j].append(JS[0][j][i])
    plot_2d([JS1], [was1], group, save_path=work_dir)
    return JS, was


def fold_n_valid(data_path, task_name, save_dir, n_part=5, fit_mode='single', compare_model_path=None,
                 compare_models=None):
    loaded_data, split_idx = load_data(data_path, n_part=n_part)
    addition_roc = None
    if compare_model_path:
        addition_roc = ML_tools.addition_roc(paths=compare_model_path, model_names=compare_models)
    for fold in tqdm(range(n_part)):
        test_idx, train_idx = ML_tools.fold_n(split_idx, fold)
        models = ML_tools.ML_models(loaded_data[train_idx], loaded_data[train_idx], loaded_data[test_idx],
                                    loaded_data[test_idx])
        models.fit_models(fit_mode=fit_mode)
        models.prediction()
        models.validation(pos_label=2)
        ML_tools.save_model_results(models, fold, task_name, save_dir=save_dir)
        joblib.dump(models, os.path.join(save_dir, f"models_fold{fold}.joblib"))
        # if fit_mode == 'single':
        #     models.plot_combined_roc(save_path=os.path.join(save_dir, task_name + f"fold{fold}_ROC.pdf"),
        #                              addition_ROCs=addition_roc, titles=task_name)
    visualization_foldn_valid(os.path.join(save_dir, task_name), save_dir, ['SVM', 'XGB', 'RF', 'LGB'],
                              task_name + 'fold5_valid.pdf',
                              addition_ROCs=addition_roc, title=task_name)


def shap_ana(data_path, fit_mode='single', models=None, max_display=30, save_dir='./'):
    ret = []
    loaded_data, split_idx = load_data(data_path, n_part=9)
    if models is None:
        models = ML_tools.ML_models(loaded_data[[i for i in range(len(loaded_data))]],
                                    loaded_data[[i for i in range(len(loaded_data))]], None, None)
        models.fit_models(fit_mode=fit_mode)
    analyzers = [
        ML_tools.UniversalSHAPAnalyzer(
            model=models.RF,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.XGB,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.lgb,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
    ]
    for analyzer in range(len(analyzers)):
        # if fit_mode == 'single':
        #     analyzer.plot_waterfall()
        analyzers[analyzer].plot_importance(["RF", "XGB", "LGB"][analyzer], save_dir=save_dir)
        ret.append(analyzers[analyzer].get_feature_importance())
    return ret, models


if __name__ == '__main__':
    for names in ['3444','1122','1705','2521','6039','3838']:
        dataset = joblib.load(f"F:\质谱不鉴定数据\源数据\合并后joblib数据/{names}.joblib")
        seledata=[i for i in range(len(dataset))]
        MassLinker_Anal(f"F:\质谱不鉴定数据\源数据\合并后joblib数据/{names}.joblib", work_dir=f'F:\MassLinker_workspace\MassLinker_Anal/{names}',
                        load_JS=f'F:\MassLinker_workspace\MassLinker_Anal/{names}\JS.joblib',
                        load_was=f'F:\MassLinker_workspace\MassLinker_Anal/{names}\Was.joblib',
                        # sele_sample=seledata,
                        p_limit=0.05)
        # fold_n_valid(f"F:\质谱不鉴定数据\源数据\合并后joblib数据/{names}.joblib", task_name=names,
        #              save_dir=f'F:\MassLinker_workspace\save_fold5_results/{names}_origin')
        # ret, models = shap_ana(r"F:\质谱不鉴定数据\源数据\合并后joblib数据\8764_reduced_enhanced.joblib",
        #                        save_dir=r'F:\MassLinker_workspace\8764\shap')

        # addition_roc = ML_tools.addition_roc(paths=[
        #     f'D:\git\MassLinker\pseudoMS-image-predictor/trained_models/{names}_predicted_224x224_fold5.csv',
        #     f'F:\MetImage\save_model/{names}/roc_data_200通道.csv',
        #     f'F:\MassLinker_workspace/transformer_single_dataset/{names}/transformer_single_result.csv',
        #     f'F:\MassLinker_workspace/transformer_single_dataset/{names}/transformer_enhanced_result.csv'
        # ],
        #     model_names=['D-PMSI', 'MetImage', 'Transformer', 'Transformer_Enhanced'], )
        # models = joblib.load(f'F:\MassLinker_workspace\saved_MLmodels/{names}_single.joblib')
        # visualization_foldn_valid(f'F:\MassLinker_workspace\save_fold5_results/{names}_origin',
        #                           f'F:\MassLinker_workspace\save_fold5_results',
        #                           ['SVM', 'XGB', 'RF', 'LGB'],
        #                           f'{names}_fold5_valid.pdf',
        #                           addition_ROCs=addition_roc, title=f'{names}_validation')
        # models.plot_combined_roc(addition_ROCs=addition_roc,
        #                          save_path=f'F:\MassLinker_workspace\save_fold5_results/{names}_fold5_valid_roc.pdf')
    ###单独数据集测试
    # with torch.no_grad():
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     transformer = transformer_language(class_count=8, embed_dim=512, n_heads=8,
    #                                        depth=8)
    #     checkpoint = torch.load(
    #         os.path.join(r'F:\MassLinker_workspace\transformer_totaldataset', 'best_model_enhanced.pth'),
    #         map_location=device)
    #     transformer.load_state_dict(checkpoint)
    #     del checkpoint
    #     torch.cuda.empty_cache()
    #     transformer.to(device)
    #     for dataset_name in os.listdir(r'F:\MassLinker_workspace\transformer_single_dataset'):
    #         tp = joblib.load(
    #             os.path.join(r'F:\MassLinker_workspace\transformer_single_dataset', dataset_name,
    #                          'train_test_index.joblib'))
    #         train_idx, test_idx = tp[0], tp[1]
    #         dataset = joblib.load(r'F:\质谱不鉴定数据\源数据\合并后joblib数据\\' + dataset_name + '.joblib')
    #         x = dataset.samples
    #         x = torch.reshape(x, (x.shape[0], x.shape[1], 60))
    #         y = dataset.is_positive
    #         y = torch.tensor(y, dtype=torch.long)
    #         x_test = x[[i for i in test_idx]]
    #         y_test = torch.tensor(y[[i for i in test_idx]], dtype=torch.long)
    #         batch_size = 32
    #             y_out[1].append(y_prob.cpu().numpy())
    #         y_out = [np.concatenate(y_out[i]) for i in range(2)]
    #         y_out = np.column_stack(y_out)
    #         column_names = ['y_true', 'y_prob']
    #         results_df = pd.DataFrame(y_out, columns=column_names)
    #         results_df.to_csv(os.path.join(r'F:\MassLinker_workspace\transformer_single_dataset', dataset_name,
    #                                        'transformer_enhanced_result.csv'))
    #         test_dataset = TensorDataset(x_test, y_test)
    #         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #         y_out = [[] for i in range(2)]
    #         for batch_x, batch_y in test_loader:
    #             batch_x = batch_x.to(device)
    #             outputs = transformer(batch_x)
    #             y_pred = torch.argmax(outputs, dim=1)
    #             y_prob = 1 - torch.nn.functional.softmax(outputs, dim=1)[:, 0]
    #             y_out[0].append(batch_y.cpu().numpy())
    #####相似性热图
    # dataset = joblib.load(r'F:\质谱不鉴定数据\源数据\合并后joblib数据\1122.joblib')
    # x = dataset.samples
    # group = np.array(dataset.is_positive)
    # was = joblib.load('1122was.joblib')
    # was_arr = np.array(was)[0]
    # p_values, was_diff = get_diff(was_arr, group)
    # idx = np.argsort(p_values)[:100]
    # x = x[:, idx, :, :].cpu().numpy()
    # x = x.reshape(x.shape[0], -1)
    # x_scaled_all = StandardScaler().fit_transform(x)
    # #pca = PCA(n_components=min(20, x.shape[0]))
    # #x_reduced = pca.fit_transform(x_scaled_all)
    # final_sort_idx = []
    # unique_groups = np.unique(group)
    # for g_val in unique_groups:
    #     this_group_indices = np.where(group == g_val)[0]
    #     this_group_data = x_scaled_all[this_group_indices]
    #     if len(this_group_indices) > 1:
    #         res_linkage = linkage(this_group_data, method='ward')
    #         local_order = leaves_list(res_linkage)
    #         final_sort_idx.extend(this_group_indices[local_order])
    #     else:
    #         final_sort_idx.extend(this_group_indices)
    # sort_indices = np.array(final_sort_idx)
    # x_sorted_pca = x_scaled_all[sort_indices]
    # group_sorted = group[sort_indices]
    # corr_matrix = np.maximum(cosine_similarity(x_sorted_pca),0)
    # unique_sorted_groups = np.unique(group_sorted)
    # color_palette = sns.color_palette("husl", len(unique_sorted_groups))
    # group_color_map = dict(zip(unique_sorted_groups, color_palette))
    # row_colors = pd.Series(group_sorted).map(group_color_map).values
    # g = sns.clustermap(
    #     corr_matrix,
    #     row_cluster=False,
    #     col_cluster=False,
    #     row_colors=row_colors,
    #     col_colors=row_colors,
    #     cmap='RdBu_r',
    #     vmin=0, vmax=1,
    #     center=0,
    #     cbar_kws={'label': 'Pearson Correlation'},
    #     figsize=(12, 10)
    # )
    # g.fig.suptitle('Sample Heatmap: Group Partitioned with Intra-group Clustering', y=1.02)
    # legend_elements = [Patch(facecolor=color_palette[i], label=f'Group {g_val}')
    #                    for i, g_val in enumerate(unique_sorted_groups)]
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.4, 1), loc='upper left', title="Groups")
    # plt.savefig(r'D:\质谱不鉴定\ai_space\相似性heatmap.pdf', format='pdf')
    #
    #
    #
    # df = pd.read_csv(r'F:\质谱不鉴定数据\源数据\1122传统代谢组分析Result\Peak_table.csv')
    # data_cols = df.columns[9:]
    # x_raw = df[data_cols].T
    # x_raw = x_raw.fillna(0)
    # group = np.array([1 if ('LCC' in col or 'RCC' in col) else 0 for col in data_cols])
    # x_np = x_raw.values
    # p_values = []
    # for i in range(x_np.shape[1]):
    #     g0_values = x_np[group == 0, i]
    #     g1_values = x_np[group == 1, i]
    #     if np.std(g0_values) == 0 and np.std(g1_values) == 0:
    #         p_values.append(1.0)
    #     else:
    #         _, p = stats.ttest_ind(g0_values, g1_values, equal_var=False)
    #         p_values.append(p)
    # idx_diff = [i for i, p in enumerate(p_values) if p <= 0.1]
    # x_filtered = x_np[:, idx_diff]
    # x_filtered_scaled = StandardScaler().fit_transform(x_filtered)
    # print(f"原始峰数量: {x_np.shape[1]}, 筛选后差异峰数量: {len(idx_diff)}")
    # final_sort_idx = []
    # unique_groups = np.unique(group)
    # for g_val in unique_groups:
    #     this_group_indices = np.where(group == g_val)[0]
    #     this_group_data = x_filtered_scaled[this_group_indices]
    #     if len(this_group_indices) > 1:
    #         res_linkage = linkage(this_group_data, method='ward')
    #         local_order = leaves_list(res_linkage)
    #         final_sort_idx.extend(this_group_indices[local_order])
    #     else:
    #         final_sort_idx.extend(this_group_indices)
    # sort_indices = np.array(final_sort_idx)
    # x_sorted = x_filtered_scaled[sort_indices]
    # group_sorted = group[sort_indices]
    # corr_matrix = np.maximum(np.corrcoef(x_sorted),0)
    # color_palette = sns.color_palette("husl", len(unique_groups))
    # group_color_map = dict(zip(unique_groups, color_palette))
    # row_colors = pd.Series(group_sorted).map(group_color_map).values
    # g = sns.clustermap(
    #     corr_matrix,
    #     row_cluster=False,
    #     col_cluster=False,
    #     row_colors=row_colors,
    #     col_colors=row_colors,
    #     cmap='RdBu_r',
    #     vmin=0, vmax=1,
    #     center=0,
    #     figsize=(10, 10)
    # )
    # legend_elements = [
    #     Patch(facecolor=color_palette[i], label=f'Group {g_val} ({"LCC/RCC" if g_val == 1 else "Normal"})')
    #     for i, g_val in enumerate(unique_groups)]
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.4, 1), loc='upper left', title="Sample Groups")
    # plt.suptitle("Standardized Peak Similarity Heatmap (Z-Score)")
    # plt.savefig(r'D:\质谱不鉴定\ai_space\相似性heatmap_传统代谢组.pdf', format='pdf')
    #
    #
    # dataset = joblib.load(r'F:\质谱不鉴定数据\源数据\合并后joblib数据\1122.joblib')
    # group = np.array(dataset.is_positive)
    # was = joblib.load('1122was.joblib')
    # was_arr = np.array(was)[0]
    # p_values, _ = get_diff(was_arr, group)
    # idx = np.argsort(p_values)[:500]
    # x_dist = was_arr[:, idx]
    # final_sort_idx = []
    # unique_groups = np.unique(group)
    # for g_val in unique_groups:
    #     this_group_indices = np.where(group == g_val)[0]
    #     this_group_data = x_dist[this_group_indices]
    #     if len(this_group_indices) > 1:
    #         res_linkage = linkage(this_group_data, method='ward')
    #         local_order = leaves_list(res_linkage)
    #         final_sort_idx.extend(this_group_indices[local_order])
    #     else:
    #         final_sort_idx.extend(this_group_indices)
    # sort_indices = np.array(final_sort_idx)
    # x_sorted_dist = x_dist[sort_indices]
    # group_sorted = group[sort_indices]
    # corr_matrix = cosine_similarity(x_sorted_dist)
    # unique_sorted_groups = np.unique(group_sorted)
    # color_palette = sns.color_palette("husl", len(unique_sorted_groups))
    # group_color_map = dict(zip(unique_sorted_groups, color_palette))
    # row_colors = pd.Series(group_sorted).map(group_color_map).values
    # plt.figure(figsize=(12, 10))
    # g = sns.clustermap(
    #     corr_matrix,
    #     row_cluster=False,
    #     col_cluster=False,
    #     row_colors=row_colors,
    #     col_colors=row_colors,
    #     cmap='RdBu_r',
    #     vmin=np.percentile(corr_matrix, 5),
    #     vmax=1,
    #     center=np.median(corr_matrix),
    #     cbar_kws={'label': 'Distance-based Cosine Similarity'}
    # )
    # g.fig.suptitle('Sample Similarity Heatmap: Based on WAS Distance Patterns', y=1.02)
    # legend_elements = [Patch(facecolor=color_palette[i], label=f'Group {g_val}')
    #                    for i, g_val in enumerate(unique_sorted_groups)]
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.4, 1), loc='upper left', title="Groups")
    # plt.savefig(r'D:\质谱不鉴定\ai_space\基于距离的相似性heatmap.pdf', format='pdf')
    # plt.show()
    # for dataset in ['1705', "3444", "1122", "2521", "6039", '3838']:
    #     morandi_colors = [
    #         '#3584FF',
    #         'white',
    #         '#FF5559'
    #     ]
    #     custom_cmap = LinearSegmentedColormap.from_list("Morandi_Diverging", morandi_colors, N=256)
    #     feature_names = pd.read_csv(r'pathway_compound_detail.csv')['compound_names'].tolist()
    #     shap_data=joblib.load(f'F:\MassLinker_workspace\ML_shap/{dataset}sample_feature_improtances.joblib')
    #     for i in range(3):
    #             plot_top_n_feature_shap_2d(shap_data[i], 1, 10, 1, feature_names,
    #                                        save_path=f"F:\MassLinker_workspace\ML_shap/shap_ML/{dataset}\shap_2d_{['RF','XGB','LGB'][i]}.pdf", mode='ML',
    #                                        cmap=custom_cmap)
    # for i in range(3):
    #     plot_top_n_feature_shap_2d(shap_data[0], 20, 15, i, feature_names,
    #                                save_path=f"F:\MassLinker_workspace\8764\shap\Transformer\shap_2d_class{i}.pdf",
    #                                mode='T',
    #                                cmap=custom_cmap)