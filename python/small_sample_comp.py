from data import ExcelDataset
from utils import metabo_dis, get_diff, kegg_enrichment, plot_enrichment, plot_peak_comp, plot_2d, plot_p_value
import joblib
import numpy as np
import pandas as pd

dataset = joblib.load(r"1122.joblib")
sele_sample = [i for i in range(len(dataset))]
# sele_sample = [i for i in [1, 232]]
group = [dataset.is_positive[i].numpy().item() for i in sele_sample]
params = [dataset.samples[i] for i in sele_sample]
# JS, was = metabo_dis(params)
# joblib.dump(JS, filename="1122JS.joblib")
# joblib.dump(was, filename="1122was.joblib")
was = joblib.load('1122was.joblib')
JS = joblib.load('1122JS.joblib')
was_arr = np.array(was)[0]
JS_arr = np.array(JS)[0]
p_values, was_diff = get_diff(was_arr, group)
met_name = pd.read_csv("pathway_compound_detail.csv")["compound_names"].tolist()
mzs = pd.read_csv("pathway_compound_detail.csv")["mz"].tolist()
plot_p_value(p_values, met_name, mzs=mzs, n=20)
plot_peak_comp(p_values, group, params, met_name, mzs, save_path=r'D:\git\MassLinker\差异峰可视化', top_n=2000)
pathway_df_full = pd.read_csv("pathway_compound_detail.csv", index_col=0)
kegg_enriched = kegg_enrichment(pathway_df_full, p_values, was_diff)
plot_df = kegg_enriched.copy()
plot_enrichment(plot_df, save_name="KEGG_Enriched1.pdf")
was1 = [[] for i in range(len(was[0]))]
JS1 = [[] for i in range(len(was[0]))]
for i in range(len(p_values)):
    if p_values[i] <= 0.01:
        for j in range(len(was[0])):
            was1[j].append(was[0][j][i])
            JS1[j].append(JS[0][j][i])
plot_2d([JS1], [was1], group, save_path=r'D:\git\MassLinker\1122')
