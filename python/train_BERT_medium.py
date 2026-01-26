import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from data import ExcelDataset
import seaborn as sns
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "CSV_PATH": "D:/质谱不鉴定/pathway_compound_detail.csv",
    "DATA_PATH": r"F:\质谱不鉴定数据\源数据\merged_dataset_enhanced.joblib",
    "MODEL_PATH": r"D:\git\MassLinker\gpt2_medium_local/",
    "SIGNATURE_DIR": "F:/MassLinker_workspace/shap_ana/",
    "CHECKPOINT_DIR": r"F:\Mass Linker_BERT\medium_from_4080S",
    "EPOCHS": 500,
    "BATCH_SIZE": 12,
    "LR": 1e-6,
    "OUTPUT_VALID_FILE": r'F:\Mass Linker_BERT\medium_from_4080S\validation.txt',
}
DRUG_LIBRARY = {
    "Cisplatin": "Platinum",
    "Carboplatin": "Platinum",
    "Oxaliplatin": "Platinum",
    "5-Fluorouracil": "Antimetabolite",
    "Capecitabine": "Antimetabolite",
    "Gemcitabine": "Antimetabolite",
    "Methotrexate": "Antimetabolite",
    "Pemetrexed": "Antimetabolite",
    "Paclitaxel": "Plant Alkaloid",
    "Docetaxel": "Plant Alkaloid",
    "Irinotecan": "Plant Alkaloid",
    "Etoposide": "Plant Alkaloid",
    "Vincristine": "Plant Alkaloid",
    "Doxorubicin": "Antibiotic",
    "Epirubicin": "Antibiotic",
    "Bleomycin": "Antibiotic",
    "Cyclophosphamide": "Alkylating",
    "Ifosfamide": "Alkylating",
    "Abiraterone": "Hormone",
    "Enzalutamide": "Hormone",
    "Tamoxifen": "Hormone",
    "Megestrol": "Hormone",
    "Sunitinib": "Targeted",
    "Sorafenib": "Targeted",
    "Bevacizumab": "Targeted",
    "Olaparib": "Targeted",
    "Pembrolizumab": "Immunotherapy"
}
SORTED_DRUG_NAMES = sorted(DRUG_LIBRARY.keys(), key=lambda x: DRUG_LIBRARY[x])
STANDARD_TREATMENTS = {
    "Descending colon cancer": ["5-Fluorouracil", "Oxaliplatin", "Capecitabine", "Irinotecan", "Bevacizumab"],
    "Ascending colon cancer":  ["5-Fluorouracil", "Oxaliplatin", "Capecitabine", "Irinotecan", "Bevacizumab"],
    "RCC":                     ["Sunitinib", "Sorafenib", "Bevacizumab", "Pembrolizumab"], # 肾癌主要是靶向和免疫
    "Cervical cancer":         ["Cisplatin", "Paclitaxel", "Bevacizumab", "Ifosfamide"],
    "Ovarian Cancer":          ["Paclitaxel", "Carboplatin", "Cisplatin", "Doxorubicin", "Olaparib", "Bevacizumab"],
    "endometrial cancer":      ["Carboplatin", "Paclitaxel", "Doxorubicin", "Cisplatin", "Megestrol"],
    "Prostate cancer":         ["Docetaxel", "Abiraterone", "Enzalutamide"]
}
DISEASE_MAPPING = {
    1: 'Descending colon cancer',
    2: 'Ascending colon cancer',
    3: 'RCC',
    4: 'Cervical cancer',
    5: 'Ovarian Cancer',
    6: 'endometrial cancer',
    7: 'Prostate cancer'
}
ENDOGENOUS_PATHWAY_IDS = [
    "hsa00010", "hsa00020", "hsa00030", "hsa00040", "hsa00051", "hsa00052", "hsa00053",
    "hsa00500", "hsa00620", "hsa00630", "hsa00640", "hsa00650", "hsa00660", "hsa00562",
    "hsa00190", "hsa00910", "hsa00920", "hsa00061", "hsa00062", "hsa00071", "hsa00100",
    "hsa00120", "hsa00140", "hsa00561", "hsa00564", "hsa00565", "hsa00600", "hsa00590",
    "hsa00591", "hsa00592", "hsa01040",
    "hsa00230", "hsa00240", "hsa00250", "hsa00260", "hsa00270", "hsa00280", "hsa00290",
    "hsa00310", "hsa00220", "hsa00330", "hsa00340", "hsa00350", "hsa00360", "hsa00380",
    "hsa00410", "hsa00430", "hsa00440", "hsa00450", "hsa00460", "hsa00470", "hsa00480",
    "hsa00520", "hsa00510", "hsa00531", "hsa00563", "hsa00601", "hsa00730", "hsa00740",
    "hsa00750", "hsa00760", "hsa00770", "hsa00780", "hsa00785", "hsa00790", "hsa00670",
    "hsa00830", "hsa00860", "hsa00130", "hsa00900",
    "hsa04973", "hsa04974", "hsa04975", "hsa04976", "hsa04724", "hsa04727", "hsa04216"
]
def create_pathway_mask(csv_path, whitelist_ids=None, num_metabolites=5194):
    print(f"Loading pathway map from {csv_path}...")
    df = pd.read_csv(csv_path)
    if whitelist_ids is not None:
        print(f"应用白名单过滤: 原始通路数 {df['kegg_id'].nunique()}", end=" -> ")
        df = df[df['kegg_id'].isin(whitelist_ids)]
        print(f"过滤后通路数 {df['kegg_id'].nunique()}")
        if df.empty:
            raise ValueError("错误：白名单过滤后没有剩余通路！")
    unique_pathways = sorted(df['kegg_id'].unique())
    id_to_name = dict(zip(df['kegg_id'], df['pathway_name']))
    pathway_to_idx = {p_id: i for i, p_id in enumerate(unique_pathways)}
    num_pathways = len(unique_pathways)
    mask = torch.zeros(num_pathways, num_metabolites)
    for meta_idx, row in df.iterrows():
        if meta_idx >= num_metabolites: continue
        p_id = row['kegg_id']
        path_idx = pathway_to_idx[p_id]
        mask[path_idx, meta_idx] = 1.0
    row_sums = mask.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    mask = mask / row_sums
    pathway_names_list = [id_to_name[pid] for pid in unique_pathways]
    return mask, pathway_names_list
class ReportSynthesizer:
    def __init__(self, pathway_names):
        self.pathway_names = pathway_names

    def generate_report(self, is_positive, group, top_active_indices):
        is_pos = bool(is_positive)

        if is_pos:
            active_pathways = [self.pathway_names[i] for i in top_active_indices]
            clean_pathways = [p.split(" - ")[0] for p in active_pathways]
            pathway_str = ", ".join(clean_pathways)
            text = (f"Diagnosis: Positive. Group: {group}. "
                    f"Significant metabolic dysregulation observed in: {pathway_str}.")
        else:
            text = f"Diagnosis: Negative. Group: {group}. Metabolic profile is normal."

        return text
class MetabolomicsDataset(Dataset):
    def __init__(self, num_samples, rbf_data, groups, is_positive, synthesizer, tokenizer, pathway_mask,
                 signature_map=None):
        self.num_samples = num_samples
        self.data = rbf_data
        self.groups = groups
        self.is_positive = is_positive
        self.synthesizer = synthesizer
        self.tokenizer = tokenizer
        self.pathway_mask = pathway_mask
        self.signature_map = signature_map

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        group_label = self.groups[idx]
        is_pos = bool(self.is_positive[idx])

        top_indices = []
        if is_pos:
            if self.signature_map is not None and group_label in self.signature_map:
                target_names = self.signature_map[group_label]
                all_names = self.synthesizer.pathway_names
                for target in target_names:
                    for i, name in enumerate(all_names):
                        if target in name:
                            top_indices.append(i)
                            break
                if not top_indices:
                    top_indices = self._calculate_realtime_top_k(x)
            else:
                top_indices = self._calculate_realtime_top_k(x)
            if len(top_indices) > 3:
                top_indices = random.sample(top_indices, 3)
        else:
            top_indices = []
        report_text = self.synthesizer.generate_report(is_pos, group_label, top_indices)
        text_input = report_text + self.tokenizer.eos_token
        encodings = self.tokenizer(
            text_input,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'metabolite_tensor': x,
            'input_ids': encodings['input_ids'].squeeze(0),
            'raw_text': report_text
        }

    def _calculate_realtime_top_k(self, x, k=3):
        metabolite_intensity = x.mean(dim=(1, 2))
        real_pathway_scores = torch.matmul(self.pathway_mask, metabolite_intensity)
        return torch.topk(real_pathway_scores, k=k).indices.tolist()

class MassLinkerBioLLM(nn.Module):
    def __init__(self, mapping_mask, model_path, rbf_dim=60, pathway_dim=512, use_res_projector=True):
        super().__init__()
        self.register_buffer('pathway_mask', mapping_mask)
        self.num_pathways = mapping_mask.size(0)

        self.metabolite_projection = nn.Sequential(
            nn.Linear(rbf_dim, pathway_dim),
            nn.LayerNorm(pathway_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        print(f"Loading Pretrained GPT-2 from: {model_path} ...")
        self.llm = GPT2LMHeadModel.from_pretrained(model_path, attn_implementation='eager')
        self.llm_hidden_size = self.llm.config.n_embd

        if use_res_projector:
            print("Initializing Stacked Deep Residual Projector (3 Blocks)...")
            self.bridge_projector = DeepResidualProjector(
                input_dim=pathway_dim,
                output_dim=self.llm_hidden_size,
                num_blocks=3
            )
        else:
            self.bridge_projector = nn.Sequential(
                nn.Linear(pathway_dim, self.llm_hidden_size),
                nn.LayerNorm(self.llm_hidden_size)
            )

    def forward(self, metabolite_tensor, input_ids=None, attention_mask=None):
        batch_size = metabolite_tensor.size(0)
        device = metabolite_tensor.device

        x_flat = metabolite_tensor.view(batch_size, 5194, -1)
        meta_embeds = self.metabolite_projection(x_flat)
        pathway_features = torch.einsum('pm, bmc -> bpc', self.pathway_mask, meta_embeds)
        soft_prompts = self.bridge_projector(pathway_features)

        if input_ids is not None:
            text_embeds = self.llm.transformer.wte(input_ids)
            inputs_embeds = torch.cat((soft_prompts, text_embeds), dim=1)
            ignore_labels = torch.full((batch_size, self.num_pathways), -100, dtype=torch.long, device=device)
            labels = torch.cat((ignore_labels, input_ids), dim=1)
            outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
            return outputs
        else:
            return soft_prompts

    def generate(self, metabolite_tensor, tokenizer, **kwargs):
        if metabolite_tensor.dim() == 3:
            metabolite_tensor = metabolite_tensor.unsqueeze(0)
        batch_size = metabolite_tensor.size(0)
        x_flat = metabolite_tensor.view(batch_size, 5194, -1)
        meta_embeds = self.metabolite_projection(x_flat)
        pathway_features = torch.einsum('pm, bmc -> bpc', self.pathway_mask, meta_embeds)
        soft_prompts = self.bridge_projector(pathway_features)
        generate_kwargs = {
            "max_new_tokens": 180,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": True,
            "top_k": 150,
            "top_p": 0.95,
            "temperature": 0.1,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 2
        }
        generate_kwargs.update(kwargs)
        soft_prompt_len = soft_prompts.shape[1]
        attention_mask = torch.ones((batch_size, soft_prompt_len), device=metabolite_tensor.device)
        generated_ids = self.llm.generate(
            inputs_embeds=soft_prompts,
            attention_mask=attention_mask,  # 传入 mask 消除警告
            **generate_kwargs
        )
        return generated_ids

class MetabolicTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, device, lr, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = save_dir
        self.optimizer = optim.AdamW([
            {'params': model.metabolite_projection.parameters(), 'lr': lr * 10},
            {'params': model.bridge_projector.parameters(), 'lr': lr * 10},
            {'params': model.llm.parameters(), 'lr': lr}
        ], lr=lr)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            metabolic_input = batch['metabolite_tensor'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(metabolic_input, input_ids=input_ids)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                metabolic_input = batch['metabolite_tensor'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                outputs = self.model(metabolic_input, input_ids=input_ids)
                total_loss += outputs.loss.item()
        return total_loss / len(self.val_loader)

    def generate_monitor(self, epoch, num_samples=5):

        self.model.eval()
        dataset = self.val_loader.dataset
        total_len = len(dataset)
        indices = random.sample(range(total_len), min(num_samples, total_len))
        print(f"\n=== [Epoch {epoch} Monitor: Random {len(indices)} Samples] ===")
        for idx in indices:
            data_item = dataset[idx]
            x = data_item['metabolite_tensor'].to(self.device).unsqueeze(0)
            truth_ids = data_item['input_ids'].to(self.device)

            with torch.no_grad():
                pred_ids = self.model.generate(x, self.tokenizer)
                pred_str = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                true_str = self.tokenizer.decode(truth_ids, skip_special_tokens=True)

                print(f"Target: {true_str}")
                print(f"Model : {pred_str}")
                print("." * 30)
        print("================================================================\n")

    def run(self, epochs):
        print(f"Start training for {epochs} epochs...")
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            start = time.time()
            t_loss = self.train_epoch(epoch)
            v_loss = self.evaluate()
            self.scheduler.step()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)

            save_msg = ""
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
                save_msg = "(*Best Saved)"
            if epoch % 20 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model{epoch}.pth"))
                save_msg = "(*New Saved)"
            print(
                f"Epoch {epoch}/{epochs} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | Time: {time.time() - start:.1f}s {save_msg}")

            if epoch % 5 == 0 or epoch == 1:
                self.generate_monitor(epoch)

        self.plot_history()

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Val')
        plt.title('Training Curve')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, expansion_factor=2, dropout=0.1):
        super().__init__()
        intermediate_size = hidden_size * expansion_factor

        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # f(x) + x
        return x + self.block(x)


class DeepResidualProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(output_dim, expansion_factor=2)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class BioLLMValidator:
    def __init__(self, model, tokenizer, device, disease_mapping, valid_pathway_names):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.disease_mapping = disease_mapping
        all_labels = list(disease_mapping.values())
        control_keywords = {'Control', 'Healthy', 'CT', 'health control', 'benign', 'Normal'}
        self.disease_list = [g for g in all_labels if g not in control_keywords]
        self.disease_lookup = {g.lower(): g for g in self.disease_list}
        self.sorted_disease_keys = sorted(self.disease_lookup.keys(), key=len, reverse=True)
        self.control_list = list(control_keywords)
        self.control_lookup = {g.lower(): g for g in self.control_list}
        self.sorted_control_keys = sorted(self.control_lookup.keys(), key=len, reverse=True)
        clean_names = [p.split(" - ")[0] for p in valid_pathway_names]
        self.pathway_lookup = {p.lower(): p for p in clean_names}
        self.sorted_pathway_keys = sorted(self.pathway_lookup.keys(), key=len, reverse=True)

    def parse_generated_text(self, text):
        text_lower = text.lower()

        diag_match = re.search(r"Diagnosis:\s*(Positive|Negative)", text, re.IGNORECASE)
        if diag_match:
            raw_diag = diag_match.group(1).lower()
            diagnosis = "Positive" if raw_diag == "positive" else "Negative"
        else:
            diagnosis = "Unknown"
        group = "Unknown"
        for search_key in self.sorted_disease_keys:
            if search_key in text_lower:
                group = self.disease_lookup[search_key]
                break

        if group == "Unknown":
            for search_key in self.sorted_control_keys:
                if search_key in text_lower:
                    group = self.control_lookup[search_key]
                    break

        if group == "Unknown":
            group_match = re.search(r"Group:?\s*(.*?)[.,]", text, re.IGNORECASE)
            if group_match:
                raw_capture = group_match.group(1).strip()
                clean_capture = re.sub(r"(observed in|is|belongs to).*", "", raw_capture, flags=re.IGNORECASE).strip()
                if len(clean_capture) < 50:
                    group = clean_capture

        found_pathways = []
        if diagnosis == "Positive":
            path_part_start = 0
            separator_match = re.search(r"(observed in|dysregulation)[:\s]+", text, re.IGNORECASE)
            if separator_match:
                path_part_start = separator_match.end()

            target_text_lower = text_lower[path_part_start:]

            for p_key in self.sorted_pathway_keys:
                if p_key in target_text_lower:
                    found_pathways.append(self.pathway_lookup[p_key])

        return diagnosis, group, found_pathways

    def calculate_jaccard(self, list1, list2):
        set1 = set([x.lower() for x in list1])
        set2 = set([x.lower() for x in list2])
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        elif len(set1) == 0 or len(set2) == 0:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def run_evaluation(self, dataloader, num_batches=None, output_file="validation_report.txt"):
        self.model.eval()
        true_diagnoses, pred_diagnoses = [], []
        true_groups, pred_groups = [], []
        jaccard_scores = []

        mismatch_logs = []

        print(f"\n>>> Starting Evaluation (Priority Scanning Mode)...")
        print(f">>> Results will be saved to: {output_file}")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches: break

                metabolic_input = batch['metabolite_tensor'].to(self.device)
                ground_truth_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in
                                      batch['input_ids']]

                generated_ids = self.model.generate(
                    metabolic_input,
                    self.tokenizer,
                    max_new_tokens=128,  # 限制生成长度
                    temperature=0.3,  # 降温
                    repetition_penalty=1.2  # 惩罚重复
                )
                generated_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

                for gt_text, pred_text in zip(ground_truth_texts, generated_texts):
                    gt_diag, gt_group, gt_paths = self.parse_generated_text(gt_text)
                    pred_diag, pred_group, pred_paths = self.parse_generated_text(pred_text)

                    true_diagnoses.append(gt_diag)
                    pred_diagnoses.append(pred_diag)
                    true_groups.append(gt_group)
                    pred_groups.append(pred_group)
                    jaccard_scores.append(self.calculate_jaccard(gt_paths, pred_paths))

                    if gt_group != pred_group:
                        mismatch_logs.append({
                            "GT_Group": gt_group,
                            "Pred_Group": pred_group,
                            "Raw_Pred_Text": pred_text
                        })

        valid_indices = [i for i, x in enumerate(pred_diagnoses) if x != 'Unknown']
        acc = accuracy_score([true_diagnoses[i] for i in valid_indices],
                             [pred_diagnoses[i] for i in valid_indices]) if valid_indices else 0
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

        with open(output_file, "w", encoding="utf-8") as f:
            def log(msg=""):
                f.write(str(msg) + "\n")
            log("=" * 50)
            log("MassLinker-BioLLM Validation Report")
            log("=" * 50)
            valid_labels = sorted(list(set(true_groups + pred_groups)))
            if 'Unknown' in valid_labels: valid_labels.remove('Unknown')
            if valid_labels:
                log(f"\n[Confusion Matrix]")
                try:
                    cm = confusion_matrix(true_groups, pred_groups, labels=valid_labels)
                    cm_df = pd.DataFrame(cm, index=valid_labels, columns=valid_labels)
                    log(cm_df.to_string())
                except Exception as e:
                    log(f"Skipping matrix viz due to error: {e}")
            log("\n[Task 2: Disease Subtyping Report]")
            log(classification_report(true_groups, pred_groups, zero_division=0))

            log(f"\nSummary Metrics:")
            log(f"  Diagnosis Accuracy: {acc:.4f}")
            log(f"  Pathway Jaccard:    {avg_jaccard:.4f}")

            if mismatch_logs:
                log(f"\n[All Mismatches: {len(mismatch_logs)}]")

                for idx, log_item in enumerate(mismatch_logs):
                    log(f"Mismatch {idx + 1}: GT='{log_item['GT_Group']}' vs Pred='{log_item['Pred_Group']}'")
                    log(f"   Context: {log_item['Raw_Pred_Text']}")
                    log("-" * 50)

        print("=" * 50)
        print(f"Done! Full validation report saved to: {output_file}")
        print(f"Summary -> Acc: {acc:.4f} | Jaccard: {avg_jaccard:.4f} | Mismatches: {len(mismatch_logs)}")
        print("=" * 50)

        return {"accuracy": acc, "jaccard": avg_jaccard}
