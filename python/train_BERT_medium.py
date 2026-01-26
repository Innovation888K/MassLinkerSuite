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
# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# ==============================================================================
# Module 1: 配置与白名单 (Configuration & Whitelist)
# ==============================================================================

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
# ==========================================
# 1. 药物列表库 (28种)
# ==========================================
DRUG_LIBRARY = {
    # --- A. 铂类 (Platinum-based) ---
    "Cisplatin": "Platinum",       # 宫颈、卵巢、肺
    "Carboplatin": "Platinum",     # 卵巢、子宫内膜、肺
    "Oxaliplatin": "Platinum",     # 结直肠癌

    # --- B. 抗代谢药 (Antimetabolites) ---
    "5-Fluorouracil": "Antimetabolite", # 结直肠癌 (5-FU)
    "Capecitabine": "Antimetabolite",   # 结直肠癌
    "Gemcitabine": "Antimetabolite",    # 卵巢、膀胱
    "Methotrexate": "Antimetabolite",   # 广泛使用
    "Pemetrexed": "Antimetabolite",     # 肺、胸膜

    # --- C. 植物生物碱 (Plant Alkaloids / Taxanes / Topo inhibitors) ---
    "Paclitaxel": "Plant Alkaloid",     # 卵巢、宫颈、子宫内膜
    "Docetaxel": "Plant Alkaloid",      # 前列腺
    "Irinotecan": "Plant Alkaloid",     # 结直肠癌
    "Etoposide": "Plant Alkaloid",      # 广泛
    "Vincristine": "Plant Alkaloid",    # 血液瘤、肉瘤

    # --- D. 抗生素类 (Antitumor Antibiotics / Anthracyclines) ---
    "Doxorubicin": "Antibiotic",        # 卵巢、子宫内膜
    "Epirubicin": "Antibiotic",         # 乳腺、胃
    "Bleomycin": "Antibiotic",          # 宫颈、生殖细胞

    # --- E. 烷化剂 (Alkylating Agents) ---
    "Cyclophosphamide": "Alkylating",   # 卵巢、广泛
    "Ifosfamide": "Alkylating",         # 宫颈、肉瘤

    # --- F. 激素/内分泌类 (Hormonal Agents) ---
    "Abiraterone": "Hormone",       # 前列腺癌 (关键)
    "Enzalutamide": "Hormone",      # 前列腺癌
    "Tamoxifen": "Hormone",         # 乳腺 (作为对照)
    "Megestrol": "Hormone",         # 子宫内膜

    # --- G. 其他/靶向/免疫 (Others / Targeted / Immuno) ---
    "Sunitinib": "Targeted",        # 肾细胞癌 (RCC 标准)
    "Sorafenib": "Targeted",        # 肾细胞癌 (RCC)
    "Bevacizumab": "Targeted",      # 结直肠、宫颈、卵巢 (抗血管生成)
    "Olaparib": "Targeted",         # 卵巢 (PARP抑制剂)
    "Pembrolizumab": "Immunotherapy" # 广泛 (MSI-H)
}

# 按照类别排序，方便画图时X轴整齐
SORTED_DRUG_NAMES = sorted(DRUG_LIBRARY.keys(), key=lambda x: DRUG_LIBRARY[x])

# ==========================================
# 2. 临床基准 (Ground Truth) - 红色空心圆圈
# ==========================================
# 定义每种癌症的“标准/常用”药物
STANDARD_TREATMENTS = {
    "Descending colon cancer": ["5-Fluorouracil", "Oxaliplatin", "Capecitabine", "Irinotecan", "Bevacizumab"],
    "Ascending colon cancer":  ["5-Fluorouracil", "Oxaliplatin", "Capecitabine", "Irinotecan", "Bevacizumab"],
    "RCC":                     ["Sunitinib", "Sorafenib", "Bevacizumab", "Pembrolizumab"], # 肾癌主要是靶向和免疫
    "Cervical cancer":         ["Cisplatin", "Paclitaxel", "Bevacizumab", "Ifosfamide"],
    "Ovarian Cancer":          ["Paclitaxel", "Carboplatin", "Cisplatin", "Doxorubicin", "Olaparib", "Bevacizumab"],
    "endometrial cancer":      ["Carboplatin", "Paclitaxel", "Doxorubicin", "Cisplatin", "Megestrol"],
    "Prostate cancer":         ["Docetaxel", "Abiraterone", "Enzalutamide"]
}
# 疾病名称映射
DISEASE_MAPPING = {
    1: 'Descending colon cancer',
    2: 'Ascending colon cancer',
    3: 'RCC',
    4: 'Cervical cancer',
    5: 'Ovarian Cancer',
    6: 'endometrial cancer',
    7: 'Prostate cancer'
}

# 人类内源性代谢通路白名单
ENDOGENOUS_PATHWAY_IDS = [
    # 1.1 碳水化合物
    "hsa00010", "hsa00020", "hsa00030", "hsa00040", "hsa00051", "hsa00052", "hsa00053",
    "hsa00500", "hsa00620", "hsa00630", "hsa00640", "hsa00650", "hsa00660", "hsa00562",
    # 1.2 能量 & 1.3 脂质
    "hsa00190", "hsa00910", "hsa00920", "hsa00061", "hsa00062", "hsa00071", "hsa00100",
    "hsa00120", "hsa00140", "hsa00561", "hsa00564", "hsa00565", "hsa00600", "hsa00590",
    "hsa00591", "hsa00592", "hsa01040",
    # 1.4 核苷酸 & 1.5 氨基酸
    "hsa00230", "hsa00240", "hsa00250", "hsa00260", "hsa00270", "hsa00280", "hsa00290",
    "hsa00310", "hsa00220", "hsa00330", "hsa00340", "hsa00350", "hsa00360", "hsa00380",
    # 1.6 其他氨基酸
    "hsa00410", "hsa00430", "hsa00440", "hsa00450", "hsa00460", "hsa00470", "hsa00480",
    # 1.7 糖链 & 1.8 辅因子
    "hsa00520", "hsa00510", "hsa00531", "hsa00563", "hsa00601", "hsa00730", "hsa00740",
    "hsa00750", "hsa00760", "hsa00770", "hsa00780", "hsa00785", "hsa00790", "hsa00670",
    "hsa00830", "hsa00860", "hsa00130", "hsa00900",
    # 补充：消化与神经递质
    "hsa04973", "hsa04974", "hsa04975", "hsa04976", "hsa04724", "hsa04727", "hsa04216"
]


# ==============================================================================
# Module 2: 工具函数 (Utils)
# ==============================================================================

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
            # === Positive (Cancer) ===
            # 列出具体的特征通路
            active_pathways = [self.pathway_names[i] for i in top_active_indices]
            clean_pathways = [p.split(" - ")[0] for p in active_pathways]
            pathway_str = ", ".join(clean_pathways)
            text = (f"Diagnosis: Positive. Group: {group}. "
                    f"Significant metabolic dysregulation observed in: {pathway_str}.")
        else:
            # === Negative (Healthy/Benign) ===
            # 【修改点】 更加简洁的报告，不需要列出 stable pathways
            text = f"Diagnosis: Negative. Group: {group}. Metabolic profile is normal."

        return text


# ==============================================================================
# Module 3: 数据集 (Dataset)
# ==============================================================================

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

        # 【修改点】 只有阳性样本才需要计算/查找异常通路
        if is_pos:
            # 策略 A: 查表 (Signature Map)
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
                # 策略 B: 阳性但未定义特征，实时计算
                top_indices = self._calculate_realtime_top_k(x)

            # 截取前 3 个
            if len(top_indices) > 3:
                top_indices = random.sample(top_indices, 3)
        else:
            # 【修改点】 阴性样本不需要异常通路，直接传空列表即可
            # ReportSynthesizer 会处理成 "Metabolic profile is normal"
            top_indices = []

        # 生成报告
        report_text = self.synthesizer.generate_report(is_pos, group_label, top_indices)

        # Tokenize
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


# ==============================================================================
# Module 4: 模型 (Model)
# ==============================================================================

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
        """
        Args:
            metabolic_input: 代谢物张量
            tokenizer: 分词器
            **kwargs: 接收 Validator 传来的 temperature, repetition_penalty 等参数
        """
        # 确保输入有 batch 维度 [1, 5194, 3, 20]
        if metabolite_tensor.dim() == 3:
            metabolite_tensor = metabolite_tensor.unsqueeze(0)

        batch_size = metabolite_tensor.size(0)

        # 1. 编码与映射
        x_flat = metabolite_tensor.view(batch_size, 5194, -1)
        meta_embeds = self.metabolite_projection(x_flat)
        pathway_features = torch.einsum('pm, bmc -> bpc', self.pathway_mask, meta_embeds)
        soft_prompts = self.bridge_projector(pathway_features)

        # 2. 设置默认生成参数
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

        # 3. 更新参数
        generate_kwargs.update(kwargs)

        # 4. 【修改点】构造 Attention Mask
        # 因为输入全是有效的 embedding (soft prompts)，所以 mask 全为 1
        soft_prompt_len = soft_prompts.shape[1]
        attention_mask = torch.ones((batch_size, soft_prompt_len), device=metabolite_tensor.device)

        # 5. 调用底层 LLM 生成
        generated_ids = self.llm.generate(
            inputs_embeds=soft_prompts,
            attention_mask=attention_mask,  # 传入 mask 消除警告
            **generate_kwargs
        )

        return generated_ids


# ==============================================================================
# Module 5: 训练器与主程序 (Trainer & Main)
# ==============================================================================

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
        """
        【修改点】: 随机从验证集中抽取 5 个样本进行展示
        """
        self.model.eval()
        dataset = self.val_loader.dataset
        total_len = len(dataset)

        # 随机抽取索引
        indices = random.sample(range(total_len), min(num_samples, total_len))

        print(f"\n=== [Epoch {epoch} Monitor: Random {len(indices)} Samples] ===")

        for idx in indices:
            # 获取单个样本
            data_item = dataset[idx]

            # 转为 Tensor 并增加 Batch 维度 [1, 5194, 3, 20]
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
    """单个残差块：Norm -> Linear -> GELU -> Linear -> Add"""

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
        """
        Args:
            input_dim: 通路维度 (例如 1024)
            output_dim: LLM维度 (例如 1024)
            num_blocks: 堆叠多少个残差块 (建议 2 或 3)
        """
        super().__init__()

        # 1. 入口对齐层 (Layer 0)
        # 先把维度对齐到 LLM 的维度，之后才能做残差相加
        self.input_proj = nn.Linear(input_dim, output_dim)

        # 2. 堆叠残差块 (Layers 1...N)
        self.blocks = nn.ModuleList([
            ResidualBlock(output_dim, expansion_factor=2)
            for _ in range(num_blocks)
        ])

        # 3. 最终归一化
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # 维度对齐
        x = self.input_proj(x)

        # 穿过多个残差块
        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)


class BioLLMValidator:
    def __init__(self, model, tokenizer, device, disease_mapping, valid_pathway_names):
        """
        Args:
            model: 训练好的模型
            tokenizer: 分词器
            device: GPU/CPU
            disease_mapping: 原始的 id2label 映射表
            valid_pathway_names: 合法通路白名单列表
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 保存映射以便 Debug 使用
        self.disease_mapping = disease_mapping

        # === 1. 组别 (Group) 查找表构建 ===
        all_labels = list(disease_mapping.values())

        # 定义对照组/良性的关键词集合
        control_keywords = {'Control', 'Healthy', 'CT', 'health control', 'benign', 'Normal'}

        # 1.1 优先列表 (Priority 1): 具体的癌症/疾病名称
        self.disease_list = [g for g in all_labels if g not in control_keywords]
        self.disease_lookup = {g.lower(): g for g in self.disease_list}
        self.sorted_disease_keys = sorted(self.disease_lookup.keys(), key=len, reverse=True)

        # 1.2 次要列表 (Priority 2): 对照/健康
        self.control_list = list(control_keywords)
        self.control_lookup = {g.lower(): g for g in self.control_list}
        self.sorted_control_keys = sorted(self.control_lookup.keys(), key=len, reverse=True)

        # === 2. 通路 (Pathway) 查找表构建 ===
        clean_names = [p.split(" - ")[0] for p in valid_pathway_names]
        self.pathway_lookup = {p.lower(): p for p in clean_names}
        self.sorted_pathway_keys = sorted(self.pathway_lookup.keys(), key=len, reverse=True)

    def parse_generated_text(self, text):
        """
        全白名单双重扫描模式解析
        """
        text_lower = text.lower()

        # --- 1. Diagnosis 解析 ---
        diag_match = re.search(r"Diagnosis:\s*(Positive|Negative)", text, re.IGNORECASE)
        if diag_match:
            raw_diag = diag_match.group(1).lower()
            diagnosis = "Positive" if raw_diag == "positive" else "Negative"
        else:
            diagnosis = "Unknown"

        # --- 2. Group 解析 ---
        group = "Unknown"

        # 第一轮：Priority 1
        for search_key in self.sorted_disease_keys:
            if search_key in text_lower:
                group = self.disease_lookup[search_key]
                break

        # 第二轮：Priority 2
        if group == "Unknown":
            for search_key in self.sorted_control_keys:
                if search_key in text_lower:
                    group = self.control_lookup[search_key]
                    break

        # 第三轮：Regex 兜底
        if group == "Unknown":
            group_match = re.search(r"Group:?\s*(.*?)[.,]", text, re.IGNORECASE)
            if group_match:
                raw_capture = group_match.group(1).strip()
                clean_capture = re.sub(r"(observed in|is|belongs to).*", "", raw_capture, flags=re.IGNORECASE).strip()
                if len(clean_capture) < 50:
                    group = clean_capture

        # --- 3. Pathway 解析 ---
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

                # === 生成 ===
                # 【修改点】使用 max_new_tokens 替代 max_length，避免冲突警告
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

        # ==========================================
        # Report Output to File
        # ==========================================

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

            # 1. Confusion Matrix
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

            # 2. Classification Report
            log("\n[Task 2: Disease Subtyping Report]")
            log(classification_report(true_groups, pred_groups, zero_division=0))

            # 3. Metrics
            log(f"\nSummary Metrics:")
            log(f"  Diagnosis Accuracy: {acc:.4f}")
            log(f"  Pathway Jaccard:    {avg_jaccard:.4f}")

            # 4. Mismatches
            if mismatch_logs:
                log(f"\n[All Mismatches: {len(mismatch_logs)}]")
                # 【关键修复】这里把遍历变量名从 log 改为了 log_item
                for idx, log_item in enumerate(mismatch_logs):
                    log(f"Mismatch {idx + 1}: GT='{log_item['GT_Group']}' vs Pred='{log_item['Pred_Group']}'")
                    log(f"   Context: {log_item['Raw_Pred_Text']}")
                    log("-" * 50)

        print("=" * 50)
        print(f"Done! Full validation report saved to: {output_file}")
        print(f"Summary -> Acc: {acc:.4f} | Jaccard: {avg_jaccard:.4f} | Mismatches: {len(mismatch_logs)}")
        print("=" * 50)

        return {"accuracy": acc, "jaccard": avg_jaccard}

        # =========================================================
        # New Method 4: Hierarchical Confusion Matrices (3-Level)
        # =========================================================



# ==============================================================================
# Module 6 Extension: 可视化工具 (Visualizer)
# ==============================================================================


class BioLLMVisualizer:
    def __init__(self, model, tokenizer, device, save_dir, pathway_names, csv_path, whitelist_ids):
        """
        Args:
            csv_path (str): 原始CSV路径，用于独立构建分类映射，不污染数据加载器
            whitelist_ids (list): 通路白名单，用于保证顺序一致
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = save_dir
        self.pathway_names = pathway_names  # 这是模型使用的名字列表（已排序）

        # === 核心修正：在类内部独立构建分类元数据，不改变外部接口 ===
        self.category_list = self._build_category_list(csv_path, whitelist_ids)

        self.model.eval()
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'

    def _build_category_list(self, csv_path, whitelist_ids):
        """
        内部辅助函数：读取CSV并生成与模型Mask顺序完全一致的功能分类列表。
        """
        import pandas as pd
        df = pd.read_csv(csv_path)

        # 1. 过滤
        if whitelist_ids:
            df = df[df['kegg_id'].isin(whitelist_ids)]

        # 2. 排序 (关键！必须与 create_pathway_mask 中的 sorted(unique) 逻辑一致)
        unique_pathways = sorted(df['kegg_id'].unique())

        # 3. 定义映射逻辑 (Hardcoded rules based on your provided text)
        def get_category(pid):
            # 碳水化合物
            if pid in ["hsa00010", "hsa00020", "hsa00030", "hsa00040", "hsa00051", "hsa00052", "hsa00053",
                       "hsa00500", "hsa00620", "hsa00630", "hsa00640", "hsa00650", "hsa00660", "hsa00562"]:
                return "Carbohydrates"
            # 能量与脂质
            if pid in ["hsa00190", "hsa00910", "hsa00920", "hsa00061", "hsa00062", "hsa00071", "hsa00100",
                       "hsa00120", "hsa00140", "hsa00561", "hsa00564", "hsa00565", "hsa00600", "hsa00590",
                       "hsa00591", "hsa00592", "hsa01040"]:
                return "Energy & Lipids"
            # 核苷酸与氨基酸
            if pid in ["hsa00230", "hsa00240", "hsa00250", "hsa00260", "hsa00270", "hsa00280", "hsa00290",
                       "hsa00310", "hsa00220", "hsa00330", "hsa00340", "hsa00350", "hsa00360", "hsa00380",
                       "hsa00410", "hsa00430", "hsa00440", "hsa00450", "hsa00460", "hsa00470", "hsa00480"]:
                return "Nucleotides & Amino Acids"
            # 糖链与辅因子
            if pid in ["hsa00520", "hsa00510", "hsa00531", "hsa00563", "hsa00601", "hsa00730", "hsa00740",
                       "hsa00750", "hsa00760", "hsa00770", "hsa00780", "hsa00785", "hsa00790", "hsa00670",
                       "hsa00830", "hsa00860", "hsa00130", "hsa00900"]:
                return "Glycans & Cofactors"
            # 其他
            return "Others"

        return [get_category(pid) for pid in unique_pathways]
    # =========================================================
    # Panel A: 混淆矩阵
    # =========================================================
    def plot_confusion_matrix(self, true_groups, pred_groups, labels=None, filename="Fig5A_ConfusionMatrix.pdf"):
        if labels is None:
            labels = sorted(list(set(true_groups + pred_groups)))
            if 'Unknown' in labels: labels.remove('Unknown')

        if not labels: return

        cm = confusion_matrix(true_groups, pred_groups, labels=labels)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / row_sums
            cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(11, 9))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Recall'})
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted Diagnosis')
        plt.title('Diagnostic Precision Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    # =========================================================
    # Panel B (Option 1): 生物学一致性 (Hit Rate)
    # =========================================================
    def plot_biological_concordance(self, dataloader, filename="Fig5B_Concordance.pdf"):
        print("Calculating Biological Concordance...")
        hits_model = 0
        total_targets = 0

        with torch.no_grad():
            for batch in dataloader:
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)

                # 计算 Input 信号强度 (Top 3)
                x_flat = metabolite_tensor.view(metabolite_tensor.size(0), 5194, -1).mean(dim=-1)
                scores = torch.matmul(x_flat, self.model.pathway_mask.t())
                _, topk_indices = torch.topk(scores, k=3, dim=1)

                # 获取 Model Output 文本
                # 这里使用 batch['input_ids'] (Ground Truth) 作为近似
                # 如果想看模型生成的，可以用 model.generate (速度较慢)
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

                for b in range(len(texts)):
                    text = texts[b]
                    target_indices = topk_indices[b].cpu().numpy()

                    # 检查 Top 3 强度的通路是否出现在文本中
                    for t_idx in target_indices:
                        t_name = self.pathway_names[t_idx].split(" - ")[0].lower()
                        if t_name in text:
                            hits_model += 1
                        total_targets += 1

        model_hit_rate = hits_model / total_targets if total_targets > 0 else 0
        # 随机基线：3 / 总通路数
        random_baseline = 3.0 / len(self.pathway_names)

        plt.figure(figsize=(6, 6))
        bars = plt.bar(["Random Chance", "MassLinker"], [random_baseline, model_hit_rate],
                       color=['gray', '#e74c3c'], alpha=0.8, width=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.ylabel("Pathway Hit Rate (Recall @ Top 3 Intensity)")
        plt.title("Biological Concordance Analysis")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()
        print(f"Saved Concordance Plot: {filename}")

    # =========================================================
    # Panel B (Option 2): 数字敲除实验 (单个通路)
    # =========================================================
    def plot_perturbation_analysis(self, dataloader, target_pathway_idx, target_pathway_name, filename=None):
        """
        执行 In Silico Knockout 并绘图
        """
        # 清洗文件名
        safe_name = re.sub(r'[\\/*?:"<>| ]', '_', target_pathway_name)
        if filename is None:
            filename = f"Fig5B_Perturbation_{safe_name}.pdf"

        # 1. 寻找合适的样本 (Best Sample Selection)
        target_sample = None
        max_signal_strength = -1.0

        # 遍历 dataloader 找一个该通路信号最强的样本
        # 这样敲除的效果才最明显
        for batch in dataloader:
            metabolite_tensor = batch['metabolite_tensor'].to(self.device)
            texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

            # 计算该 batch 中所有样本在该通路上的强度
            batch_size = metabolite_tensor.size(0)
            x_flat = metabolite_tensor.view(batch_size, 5194, -1).mean(dim=-1)
            # [B, 5194] * [5194] -> [B]
            pathway_vec = self.model.pathway_mask[target_pathway_idx]
            intensities = torch.matmul(x_flat, pathway_vec)

            for i in range(batch_size):
                # 条件：文本里提到它 AND 信号强度不错
                if target_pathway_name.lower() in texts[i]:
                    curr_strength = intensities[i].item()
                    if curr_strength > max_signal_strength:
                        max_signal_strength = curr_strength
                        target_sample = metabolite_tensor[i].unsqueeze(0)  # [1, 5194, 3, 20]

        if target_sample is None:
            # print(f"Skipping {target_pathway_name}: No valid sample found.")
            return False  # 跳过

        # 2. 对比生成 (Original vs Knockout)
        self.model.eval()
        N_trials = 50  # 增加次数以稳定概率

        # A. Original
        count_orig = 0
        for _ in range(N_trials):
            ids = self.model.generate(target_sample, self.tokenizer, max_new_tokens=100, do_sample=True,
                                      temperature=0.7)
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True).lower()
            if target_pathway_name.lower() in text:
                count_orig += 1

        # B. Knockout
        original_mask_val = self.model.pathway_mask[target_pathway_idx].clone()
        self.model.pathway_mask[target_pathway_idx] = 0.0  # 阻断信号

        count_ko = 0
        for _ in range(N_trials):
            ids = self.model.generate(target_sample, self.tokenizer, max_new_tokens=100, do_sample=True,
                                      temperature=0.7)
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True).lower()
            if target_pathway_name.lower() in text:
                count_ko += 1

        # 恢复 Mask
        self.model.pathway_mask[target_pathway_idx] = original_mask_val

        prob_orig = count_orig / N_trials
        prob_ko = count_ko / N_trials

        # 【过滤机制】如果原始概率太低 (<0.1)，说明模型本来就不太想说这个词，敲除没意义
        if prob_orig < 0.05:
            return False

        # 3. 绘图
        plt.figure(figsize=(6, 6))
        colors = ['#2ecc71', '#e74c3c']  # Green, Red

        plt.bar(['Original Signal', 'Signal Knockout'], [prob_orig, prob_ko], color=colors, alpha=0.8, width=0.5)
        plt.ylabel(f"Generation Probability")
        plt.title(f"Causal Validation: {target_pathway_name}\n(In Silico Knockout)")
        plt.ylim(0, 1.05)

        # 标注下降幅度
        diff = prob_orig - prob_ko
        if diff > 0:
            plt.arrow(0.5, prob_orig, 0.5, -diff,
                      head_width=0.05, head_length=0.05, fc='k', ec='k', linestyle='--', length_includes_head=True)
            plt.text(0.5, (prob_orig + prob_ko) / 2, f"-{diff:.0%} Drop", ha='center', va='bottom', fontweight='bold',
                     color='red')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Perturbation: {filename} (Drop: {diff:.2f})")
        return True

    # =========================================================
    # 批量遍历工具
    # =========================================================
    def run_all_perturbations(self, dataloader):
        """
        遍历所有通路，自动生成 Perturbation 图
        """
        print(f"\n>>> Batch Processing {len(self.pathway_names)} Pathways for Perturbation Analysis...")
        success_count = 0

        for i, p_name in enumerate(self.pathway_names):
            clean_name = p_name.split(" - ")[0]
            # 只有当成功生成图片时，才计数
            if self.plot_perturbation_analysis(dataloader, i, clean_name,
                                               filename=self.pathway_names[i] + "_perturbations.pdf"):
                success_count += 1

        print(f"\n>>> Batch Finished. Generated {success_count} valid perturbation plots.")

    # ... (Attention Map 和 Case Study 保持不变，可以照抄之前的) ...
    def plot_attention_map(self, metabolite_tensor, target_text, layer_idx=-1, head_idx=0,
                           custom_colors=None, filename="Fig5C_Attention.pdf"):
        if metabolite_tensor.dim() == 3: metabolite_tensor = metabolite_tensor.unsqueeze(0)
        metabolite_tensor = metabolite_tensor.to(self.device)
        inputs = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        self.model.llm.config.output_attentions = True
        try:
            with torch.no_grad():
                batch_size = metabolite_tensor.size(0)
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)
                text_embeds = self.model.llm.transformer.wte(input_ids)
                inputs_embeds = torch.cat((soft_prompts, text_embeds), dim=1)
                total_len = inputs_embeds.shape[1]
                att_mask = torch.ones((batch_size, total_len), device=self.device)

                outputs = self.model.llm(inputs_embeds=inputs_embeds, attention_mask=att_mask, return_dict=True)
                attention = outputs.attentions[layer_idx][0]
                attn_map = attention[head_idx].cpu().numpy()
        finally:
            self.model.llm.config.output_attentions = False

        n_prompts = soft_prompts.shape[1]
        y_tokens = [self.tokenizer.decode([tid]).strip() for tid in input_ids[0]]
        if self.pathway_names and len(self.pathway_names) == n_prompts:
            x_labels = [name.split(" - ")[0] for name in self.pathway_names]
        else:
            x_labels = [f"Pathway_{i}" for i in range(n_prompts)]

        interest_map = attn_map[n_prompts:, :n_prompts]
        if custom_colors:
            my_cmap = LinearSegmentedColormap.from_list("custom_bio", custom_colors)
        else:
            my_cmap = "viridis"

        plt.figure(figsize=(24, 10))
        sns.heatmap(interest_map, cmap=my_cmap, yticklabels=y_tokens, xticklabels=x_labels)
        plt.title(f"Cross-Modal Attention (Layer {layer_idx}, Head {head_idx})")
        plt.xlabel("Metabolic Pathway Tokens (Soft Prompts)")
        plt.ylabel("Generated Clinical Tokens")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_case_study(self, gt_text, pred_text, case_id="Sample_1", filename="Fig5D_CaseStudy.png"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        props_gt = dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.5, edgecolor='green')
        props_pred = dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.5, edgecolor='red')
        ax.text(0.02, 0.9, f"CASE ID: {case_id}", fontsize=14, fontweight='bold')
        ax.text(0.02, 0.75, "Ground Truth (Expert Annotation):", fontsize=11, fontweight='bold', color='green')
        ax.text(0.02, 0.55, self._wrap_text(gt_text), fontsize=10, bbox=props_gt, va='top')
        ax.text(0.02, 0.35, "MassLinker Generated Diagnosis:", fontsize=11, fontweight='bold', color='red')
        ax.text(0.02, 0.15, self._wrap_text(pred_text), fontsize=10, bbox=props_pred, va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def _wrap_text(self, text, width=100):
        import textwrap
        return textwrap.fill(text, width)

    def plot_semantic_alignment(self, dataloader, num_batches=20,
                                use_attention_as_x=True,  # 【新功能】使用注意力作为X轴
                                filter_zeros=True,  # 【新功能】过滤掉没被提及的通路
                                filename="Fig5B_SemanticAlignment.pdf"):
        """
        Args:
            use_attention_as_x: True=使用Attention权重(语义), False=使用RBF强度(物理)
            filter_zeros: True=隐藏Y轴为0的点
        """
        print(
            f"Calculating Semantic Alignment (X={'Attention' if use_attention_as_x else 'Intensity'}, Filter0={filter_zeros})...")

        p_names = self.pathway_names
        # 记录累积值
        pathway_x_metrics = {name: [] for name in p_names}  # X轴数据容器
        pathway_mentions = {name: [] for name in p_names}  # Y轴数据容器

        # 临时开启 Attention 用于计算
        if use_attention_as_x:
            self.model.llm.config.output_attentions = True

        try:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_batches: break

                    metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                    batch_size = metabolite_tensor.shape[0]

                    # 1. 获取 Output (Y轴数据: 提及频率)
                    texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

                    # 2. 获取 Input/Internal (X轴数据)
                    if use_attention_as_x:
                        # === 方案二：计算语义注意力 (Semantic Attention) ===
                        x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                        meta_embeds = self.model.metabolite_projection(x_flat)
                        pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                        soft_prompts = self.model.bridge_projector(pathway_features)

                        input_ids = batch['input_ids'].to(self.device)
                        text_embeds = self.model.llm.transformer.wte(input_ids)
                        inputs_embeds = torch.cat((soft_prompts, text_embeds), dim=1)

                        # 构造 Mask
                        att_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=self.device)

                        outputs = self.model.llm(
                            inputs_embeds=inputs_embeds,
                            attention_mask=att_mask,
                            return_dict=True
                        )

                        # 取最后一层的平均 Attention [B, Heads, Seq, Seq] -> [B, Seq, Seq]
                        # 重点看：文本Token 对 SoftPrompts 的注意力
                        # SoftPrompts 长度
                        n_prompts = soft_prompts.shape[1]
                        # 取最后一层注意力
                        last_layer_attn = outputs.attentions[-1].mean(dim=1)  # [B, Seq, Seq] (平均所有头)

                        # 切片：[B, Text_Tokens, Pathway_Tokens]
                        # 对 Text 维度求和 -> 得到这一个样本中，每个 Pathway 被看了多少眼
                        # [B, Pathway_Tokens]
                        batch_attn_scores = last_layer_attn[:, n_prompts:, :n_prompts].sum(dim=1)  # [B, P]

                    else:
                        # === 方案一：计算物理强度 (Physical Intensity) ===
                        x_flat = metabolite_tensor.view(batch_size, 5194, -1).mean(dim=-1)
                        batch_attn_scores = torch.matmul(x_flat, self.model.pathway_mask.t())  # [B, P] (其实是 intensity)

                    # 3. 存入字典
                    for b in range(batch_size):
                        scores = batch_attn_scores[b].cpu().numpy()
                        text = texts[b]

                        for p_idx, p_name in enumerate(p_names):
                            clean_name = p_name.split(" - ")[0]

                            # X轴: 归一化一下，避免量级差异太大
                            val = scores[p_idx]
                            pathway_x_metrics[p_name].append(val)

                            # Y轴: 是否提及
                            is_mentioned = 1 if clean_name.lower() in text else 0
                            pathway_mentions[p_name].append(is_mentioned)
        finally:
            if use_attention_as_x:
                self.model.llm.config.output_attentions = False

        # 4. 汇总统计
        x_vals = []
        y_vals = []
        labels = []

        for p_name in p_names:
            xs = np.array(pathway_x_metrics[p_name])
            ys = np.array(pathway_mentions[p_name])

            mean_x = np.mean(xs)
            mean_y = np.mean(ys)

            # 【方案一：过滤零值】
            if filter_zeros and mean_y == 0:
                continue

            # 基础过滤（防止完全没激活的）
            if np.mean(xs) > 1e-6:
                x_vals.append(mean_x)
                y_vals.append(mean_y)
                labels.append(p_name.split(" - ")[0])

        # 5. 绘图
        plt.figure(figsize=(10, 6))

        # 散点
        sns.scatterplot(x=x_vals, y=y_vals, color="#34495e", alpha=0.7, s=100, edgecolor='w')

        # 拟合线
        if len(x_vals) > 1:
            sns.regplot(x=x_vals, y=y_vals, scatter=False, color="#e74c3c", line_kws={"linewidth": 2.5})
            corr, _ = pearsonr(x_vals, y_vals)
        else:
            corr = 0

        # 标注 Top 5 和 Bottom 3
        # 找出 Y 值最高的点标注
        sorted_indices = np.argsort(y_vals)
        indices_to_label = sorted_indices[-5:]  # Top 5
        # 如果不是太拥挤，也可以标几个异常点

        for idx in indices_to_label:
            plt.text(x_vals[idx], y_vals[idx] + 0.005, labels[idx], fontsize=9, fontweight='bold', color='#2c3e50')

        # 坐标轴标签
        if use_attention_as_x:
            plt.xlabel("Semantic Attention Score (Internal Model Focus)")
            title_prefix = "Semantic Consistency"
        else:
            plt.xlabel("Biological Signal Intensity (Physical Input)")
            title_prefix = "Signal-to-Text Mapping"

        plt.ylabel("Generation Frequency (Model Output)")
        plt.title(f"{title_prefix}: Attention vs. Generation (r={corr:.2f})")
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Alignment Chart to {save_path}")

        # =========================================================
        # Figure 6: 通用语义偏差分析 (Generalized Semantic Deviation)
        # =========================================================
        # =========================================================
        # Figure 6: 通用语义偏差分析 (支持多基准组合)
        # =========================================================

    def plot_semantic_deviation(self, dataloader,
                                baseline_group=["Control"],  # 支持列表
                                target_group="RCC",
                                filename="Fig6_SemanticTypo.pdf"):
        """
        计算 [目标组] 相对于 [基准组集合] 的语义偏差。

        Args:
            baseline_group (str or List[str]): 基准组关键词列表。只要匹配其中任意一个，就算作基准样本。
                                               例如: ["Healthy", "Control", "Benign"]
            target_group (str): 目标组关键词。例如: "RCC"
        """
        # 1. 规范化输入：如果是字符串，转为列表
        if isinstance(baseline_group, str):
            baseline_group = [baseline_group]

        # 2. 自动生成文件名
        if filename is None:
            safe_target = re.sub(r'[\\/*?:"<>| ]', '_', target_group)
            if len(baseline_group) == 1:
                safe_base = re.sub(r'[\\/*?:"<>| ]', '_', baseline_group[0])
            else:
                safe_base = "Combined_Baseline"  # 组合基准
            filename = f"Fig6_Deviation_{safe_target}_vs_{safe_base}.pdf"

        print(f"Running Semantic Deviation Analysis...")
        print(f"  Target: '{target_group}'")
        print(f"  Baseline (Combined): {baseline_group}")

        # 容器
        baseline_embeds = []
        target_embeds = []

        # 3. 提取语义向量
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

                # 计算 Soft Prompts [B, P, D]
                batch_size = metabolite_tensor.size(0)
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)

                # 分组收集
                for i, text in enumerate(texts):
                    # A. 检查是否为目标组 (优先级最高)
                    if target_group.lower() in text:
                        target_embeds.append(soft_prompts[i].cpu())

                    # B. 检查是否为基准组 (只要包含列表中的任意一个词)
                    # 使用 any() 函数进行多词匹配
                    elif any(bg.lower() in text for bg in baseline_group):
                        baseline_embeds.append(soft_prompts[i].cpu())

        # 检查样本量
        if not baseline_embeds:
            print(f"Error: No samples found for Baseline Group {baseline_group}.")
            return
        if not target_embeds:
            print(f"Error: No samples found for Target Group '{target_group}'.")
            return

        print(f"  Samples found -> Combined Baseline: {len(baseline_embeds)}, Target: {len(target_embeds)}")

        # 堆叠
        base_stack = torch.stack(baseline_embeds)
        target_stack = torch.stack(target_embeds)

        # 4. 计算基准质心 (The Robust Standard Reference)
        # 这里实际上是把 Healthy, Benign, Control 的所有样本混在一起求了一个平均人
        base_centroid = torch.mean(base_stack, dim=0)  # [P, D]

        # 5. 计算目标质心
        target_centroid = torch.mean(target_stack, dim=0)  # [P, D]

        # 6. 计算偏差 (L2 Distance in Latent Space)
        deviation_scores = torch.norm(target_centroid - base_centroid, p=2, dim=1)

        # 7. 排序与绘图 (Top 20)
        scores_np = deviation_scores.numpy()
        top_indices = np.argsort(scores_np)[-20:]

        top_names = [self.pathway_names[i].split(" - ")[0] for i in top_indices]
        top_scores = scores_np[top_indices]

        plt.figure(figsize=(10, 8))
        colors = plt.cm.magma(np.linspace(0.4, 0.9, len(top_scores)))  # 使用 Magma 配色，更有压迫感

        bars = plt.barh(top_names, top_scores, color=colors)

        # 动态标题
        base_title = baseline_group[0] if len(baseline_group) == 1 else "Combined Healthy/Control"
        plt.xlabel(f"Semantic Deviation (Latent Distance from {base_title})")
        plt.title(
            f"Metabolic 'Typo' Detection: {target_group} vs {base_title}\n(Top 20 Dysregulated Pathways in Latent Space)")
        plt.grid(axis='x', linestyle='--', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                     f'{width:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Semantic Deviation Plot: {save_path}")

    def plot_logit_shift(self, dataloader,
                         baseline_group=["Healthy", "Control", "Benign"],
                         target_group="RCC",
                         filename=None):
        """
        通过对比 Next Token Logits，寻找模型最想"纠正"（生成）的通路词。
        """
        # 1. 自动生成文件名
        if filename is None:
            safe_target = re.sub(r'[\\/*?:"<>| ]', '_', target_group)
            filename = f"Fig6_LogitShift_{safe_target}.pdf"

        print(f"Running LLM Spell-Checker (Logit Shift): {target_group} vs Baseline...")

        # 2. 准备"诱导提示" (Trigger Prompt)
        # 我们强迫模型去接这句话，看它最想接哪个通路的名字
        trigger_text = "Significant metabolic dysregulation observed in:"
        trigger_ids = self.tokenizer(trigger_text, return_tensors='pt')['input_ids'].to(self.device)

        # 3. 预计算所有通路的 Token ID (我们只看通路名字的第一个核心词)
        # 例如 "Tyrosine metabolism" -> 关注 "Tyrosine" 这个词的 Logit
        pathway_token_map = {}
        for p_name in self.pathway_names:
            clean_name = p_name.split(" - ")[0]
            # 为了准确，我们取名字的第一个Token作为代理
            # 注意：要在前面加空格，因为GPT2分词对空格敏感
            first_word = " " + clean_name.split()[0]
            token_ids = self.tokenizer(first_word)['input_ids']
            if len(token_ids) > 0:
                target_token_id = token_ids[0]  # 取第一个token ID
                pathway_token_map[p_name] = target_token_id

        # 容器：存储每个通路在不同组别下的 Logit 值
        baseline_logits = {p: [] for p in self.pathway_names}
        target_logits = {p: [] for p in self.pathway_names}

        self.model.eval()

        # 4. 批量推理
        with torch.no_grad():
            for batch in dataloader:
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

                batch_size = metabolite_tensor.size(0)

                # A. 计算 Soft Prompts
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)

                # B. 拼接 诱导提示 (Force Feeding)
                # Input = [Soft Prompts] + [Trigger Text Embeds]
                trigger_embeds = self.model.llm.transformer.wte(trigger_ids).expand(batch_size, -1, -1)
                inputs_embeds = torch.cat((soft_prompts, trigger_embeds), dim=1)

                # C. 前向传播 (只拿 Logits，不生成)
                outputs = self.model.llm(inputs_embeds=inputs_embeds)
                # 取最后一个 token 的 logits (即预测"冒号"后面该接什么词)
                next_token_logits = outputs.logits[:, -1, :]  # [B, Vocab_Size]

                # D. 提取特定通路的 Logits
                for i, text in enumerate(texts):
                    # 判断当前样本属于哪一组
                    is_target = target_group.lower() in text
                    is_baseline = any(bg.lower() in text for bg in baseline_group)

                    if not is_target and not is_baseline: continue

                    for p_name, t_id in pathway_token_map.items():
                        score = next_token_logits[i, t_id].item()

                        if is_target:
                            target_logits[p_name].append(score)
                        elif is_baseline:
                            baseline_logits[p_name].append(score)

        # 5. 计算 Shift (Diff)
        pathway_shifts = {}
        for p_name in self.pathway_names:
            b_vals = baseline_logits[p_name]
            t_vals = target_logits[p_name]

            if len(b_vals) > 0 and len(t_vals) > 0:
                # Shift = Target平均Logit - Baseline平均Logit
                # 正值越大，说明模型在癌症样本上越"渴望"说出这个词
                shift = np.mean(t_vals) - np.mean(b_vals)
                pathway_shifts[p_name] = shift
            else:
                pathway_shifts[p_name] = -999  # 无效数据

        # 6. 排序与绘图 (Top 20 "Typo")
        sorted_items = sorted(pathway_shifts.items(), key=lambda x: x[1], reverse=True)[:20]
        if not sorted_items:
            print("Error: No valid logits data collected.")
            return

        top_names = [k.split(" - ")[0] for k, v in sorted_items]
        top_scores = [v for k, v in sorted_items]

        plt.figure(figsize=(10, 8))
        # 配色：使用 Coolwarm，红色代表强烈的"纠错"信号
        colors = plt.cm.coolwarm(np.linspace(0.6, 1.0, len(top_scores)))

        bars = plt.barh(top_names[::-1], top_scores[::-1], color=colors)  # 倒序画，让最大的在上面

        plt.xlabel("Logit Shift (Prediction Confidence Gain vs Healthy)")
        plt.title(f"Metabolic 'Spell-Checker': {target_group}\n(What the LLM wants to say most compared to Healthy)")
        plt.grid(axis='x', linestyle='--', alpha=0.3)

        # 标注
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'+{width:.2f}', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Logit Shift Plot: {save_path}")

        # =========================================================
        # Figure 6/7: AI 药剂师 (Therapeutic Suggestion)
        # 检测模型是否能根据代谢特征，联想到潜在的治疗药物
        # =========================================================

    def plot_comprehensive_pharmacist(self, dataloader, filename="Fig6_Comprehensive_Pharmacist.pdf",topn=5):
        """
        全景药物推荐图：
        X轴：药物 (按机制分类)
        Y轴：癌症类型
        Marker 1 (红色空心圆): 临床标准治疗 (Ground Truth)
        Marker 2 (蓝色数字): 模型基于代谢特征的推荐排名 (Top 5)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. 定义数据 (使用上面定义的字典)
        # 注意：这里需要在类外部定义 DRUG_LIBRARY, SORTED_DRUG_NAMES, STANDARD_TREATMENTS
        # 或者直接把上面的定义复制到这里面

        # 为了代码独立性，我把关键列表再次放在这里，或者你可以作为参数传入
        # (此处复用上文定义的全局变量 SORTED_DRUG_NAMES 和 STANDARD_TREATMENTS)
        all_drugs = SORTED_DRUG_NAMES
        cancer_types = [
            "Descending colon cancer", "Ascending colon cancer", "RCC",
            "Cervical cancer", "Ovarian Cancer", "endometrial cancer", "Prostate cancer"
        ]

        # 2. 准备 Token IDs
        trigger_text = "Suggested therapeutic intervention or drug:"
        trigger_ids = self.tokenizer(trigger_text, return_tensors='pt')['input_ids'].to(self.device)

        drug_token_map = {}
        for d_name in all_drugs:
            # 同样取第一个 subword token, 加空格处理
            first_word = " " + d_name
            token_ids = self.tokenizer(first_word)['input_ids']
            if len(token_ids) > 0:
                drug_token_map[d_name] = token_ids[0]
            else:
                print(f"Warning: Drug '{d_name}' not found in vocabulary.")

        # 3. 开始遍历癌症进行计算
        print(f"Running Comprehensive AI Pharmacist Analysis...")

        # 存储模型预测结果： {cancer: {drug: rank}}
        model_predictions = {}

        # 预计算 Baseline (Healthy/Control) 的 Logits
        # 为了节省时间，我们先扫一遍数据算出 Baseline Logits
        baseline_group = ["Healthy", "Control", "Benign", "Normal"]
        baseline_logits_accum = {d: [] for d in all_drugs}

        # 还要把癌症的数据先按组别存好，避免重复过 DataLoader
        cancer_data_logits = {c: {d: [] for d in all_drugs} for c in cancer_types}

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]
                batch_size = metabolite_tensor.size(0)

                # 计算 Soft Prompts + Trigger
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)
                trigger_embeds = self.model.llm.transformer.wte(trigger_ids).expand(batch_size, -1, -1)
                inputs_embeds = torch.cat((soft_prompts, trigger_embeds), dim=1)

                outputs = self.model.llm(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]

                for i, text in enumerate(texts):
                    # 检查是否是 Baseline
                    is_baseline = any(bg.lower() in text for bg in baseline_group)
                    if is_baseline:
                        for d_name, t_id in drug_token_map.items():
                            baseline_logits_accum[d_name].append(next_token_logits[i, t_id].item())
                        continue

                    # 检查是否是目标癌症
                    for c_type in cancer_types:
                        if c_type.lower() in text:
                            for d_name, t_id in drug_token_map.items():
                                cancer_data_logits[c_type][d_name].append(next_token_logits[i, t_id].item())
                            break  # 一个样本只归为一类

        # 4. 计算 Shift 并排名
        # 计算 Baseline 平均值
        baseline_means = {d: np.mean(vals) if vals else -999 for d, vals in baseline_logits_accum.items()}

        for c_type in cancer_types:
            shifts = {}
            for d_name in all_drugs:
                c_vals = cancer_data_logits[c_type][d_name]
                if not c_vals or baseline_means[d_name] == -999:
                    shifts[d_name] = -9999
                else:
                    shifts[d_name] = np.mean(c_vals) - baseline_means[d_name]

            # 排序取出 Top 5
            sorted_drugs = sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:topn]

            # 存入结果字典: {drug_name: rank_number}
            model_predictions[c_type] = {item[0]: rank + 1 for rank, item in enumerate(sorted_drugs)}
            print(f"  {c_type} Top 3: {[k for k, v in sorted_drugs[:3]]}")

        # 5. 绘图 (Grid Plot)
        fig, ax = plt.subplots(figsize=(16, 8))

        # 建立网格坐标
        x_coords = range(len(all_drugs))
        y_coords = range(len(cancer_types))

        # 画背景网格
        ax.set_xticks(x_coords)
        ax.set_xticklabels(all_drugs, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(y_coords)
        ax.set_yticklabels(cancer_types, fontsize=11, fontweight='bold')

        ax.grid(True, linestyle='--', alpha=0.3)

        # 标记颜色区分药物类别 (可选，为了美观)
        category_colors = {
            "Platinum": "#95a5a6", "Antimetabolite": "#3498db",
            "Plant Alkaloid": "#2ecc71", "Antibiotic": "#e74c3c",
            "Alkylating": "#9b59b6", "Hormone": "#f1c40f",
            "Targeted": "#e67e22", "Immunotherapy": "#34495e"
        }
        # 在X轴标签上应用颜色
        for xtick, drug in zip(ax.get_xticklabels(), all_drugs):
            cat = DRUG_LIBRARY.get(drug, "Others")
            xtick.set_color(category_colors.get(cat, "black"))

        # --- Layer 1: Plot Ground Truth (Red Hollow Circles) ---
        for y_idx, c_type in enumerate(cancer_types):
            std_drugs = STANDARD_TREATMENTS.get(c_type, [])
            for d_name in std_drugs:
                if d_name in all_drugs:
                    x_idx = all_drugs.index(d_name)
                    # 画红色空心圆
                    ax.scatter(x_idx, y_idx, s=500, facecolors='none', edgecolors='#e74c3c', linewidth=2, zorder=1)

        # --- Layer 2: Plot Model Predictions (Blue Numbers) ---
        for y_idx, c_type in enumerate(cancer_types):
            preds = model_predictions.get(c_type, {})  # {drug: rank}
            for d_name, rank in preds.items():
                if d_name in all_drugs:
                    x_idx = all_drugs.index(d_name)
                    # 画蓝色数字
                    ax.text(x_idx, y_idx, str(rank), color='#2980b9',
                            ha='center', va='center', fontsize=12, fontweight='bold', zorder=2)

        # 添加 Legend (手动构造)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markeredgecolor='#e74c3c', markerfacecolor='none', markersize=15,
                   markeredgewidth=2, label='Clinical Standard (Ground Truth)'),
            Line2D([0], [0], marker='$1$', color='w', markeredgecolor='#2980b9', markerfacecolor='#2980b9',
                   markersize=15, label='Model Prediction Rank (1=Top)')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        plt.title("AI Pharmacist: Metabolic-Inferred Drug Recommendation vs Clinical Standards", y=1.16, fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Comprehensive Pharmacist Plot: {save_path}")

    # =========================================================
    # Figure 8: 潜在空间亚型分析 (Latent Subtyping)
    # =========================================================
    def plot_latent_subtyping(self, dataloader, target_group="Ovarian Cancer", filename=None):
        """
        使用 Soft Prompts 进行 T-SNE 降维，查看同一疾病内部是否有亚型分化
        """
        from sklearn.manifold import TSNE
        if filename is None:
            filename = f"Fig8_Subtyping_{target_group}.pdf"

        print(f"Running Latent Subtyping for {target_group}...")

        embeds = []
        labels = []  # 这里如果有临床亚型数据最好，没有的话我们尝试无监督聚类

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True).lower() for ids in batch['input_ids']]

                batch_size = metabolite_tensor.size(0)
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)

                # 展平 Soft Prompts: [B, P, D] -> [B, P*D]
                flat_prompts = soft_prompts.view(batch_size, -1).cpu().numpy()

                for i, text in enumerate(texts):
                    if target_group.lower() in text:
                        embeds.append(flat_prompts[i])
                    # 也可以把 Healthy 放进来做对比
                    elif "healthy" in text or "control" in text:
                        pass  # 暂时只看疾病内部

        if len(embeds) < 10:
            print("Not enough samples for T-SNE.")
            return

        X = np.array(embeds)

        # T-SNE 降维
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeds) - 1), random_state=42)
        X_embedded = tsne.fit_transform(X)

        # K-Means 聚类 (尝试找两个亚型)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X)

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.8,
                              edgecolor='w')
        plt.title(f"Latent Space Subtyping: {target_group}\n(Unsupervised Discovery of Metabolic Sub-phenotypes)")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.legend(*scatter.legend_elements(), title="AI-Discovered Subtypes")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Subtyping Plot: {save_path}")

        # =========================================================
        # New Method 1: 向量空间分析 (验证"物以类聚")
        # =========================================================

    def plot_vector_space_analysis(self, dataloader, filename="Fig7A_VectorSpace_Comparison.pdf"):
        """
        只计算一次 t-SNE 坐标，然后在两个子图上分别上色：
        Left:  Model-Learned Clustering (K-Means)
        Right: Biological Ground Truth (Manual Categories)
        """
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        import matplotlib.patches as mpatches

        print(f"Running Vector Space Comparison (Side-by-Side)...")

        # 1. 获取 Soft Prompts (与之前一致)
        accumulated_prompts = []
        num_batches = 1000
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                batch_size = metabolite_tensor.size(0)
                x_flat = metabolite_tensor.view(batch_size, 5194, -1)
                meta_embeds = self.model.metabolite_projection(x_flat)
                pathway_features = torch.einsum('pm, bmc -> bpc', self.model.pathway_mask, meta_embeds)
                soft_prompts = self.model.bridge_projector(pathway_features)
                batch_avg_prompt = soft_prompts.mean(dim=0).cpu()
                accumulated_prompts.append(batch_avg_prompt)

        all_prompts = torch.stack(accumulated_prompts).mean(dim=0).numpy()  # [P, D]

        # 2. t-SNE 降维 (只做一次！)
        print(f"  Projecting {len(self.pathway_names)} pathways to 2D...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(all_prompts)

        # 3. 准备两套标签
        # Set A: Model Auto Clustering
        n_clusters = 5  # 根据你的生物学类别数量大概设定
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_auto = kmeans.fit_predict(all_prompts)

        # Set B: Biological Ground Truth
        labels_manual = self.category_list
        unique_cats = sorted(list(set(labels_manual)))
        cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
        labels_manual_ids = [cat_to_id[cat] for cat in labels_manual]  # 转为数字用于配色

        # 4. 绘图 (Side-by-Side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # --- Left Plot: Model Perception (Auto) ---
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_auto, cmap='viridis', s=100, alpha=0.8, edgecolors='w')
        ax1.set_title("A. Model's Internal Representation\n(Unsupervised K-Means Clustering)", fontsize=14,
                      fontweight='bold')
        ax1.set_xlabel("t-SNE Dim 1")
        ax1.set_ylabel("t-SNE Dim 2")

        # 标注文字 (Model View)
        for i, name in enumerate(self.pathway_names):
            clean_name = name.split(" - ")[0]
            ax1.text(X_2d[i, 0] + 0.2, X_2d[i, 1] + 0.2, clean_name, fontsize=7, alpha=0.6)

        # --- Right Plot: Biological Reality (Ground Truth) ---
        # 使用离散调色板
        palette = sns.color_palette("Set2", len(unique_cats))
        # 手动画 Scatter 以便控制 Legend
        for cat_idx, cat_name in enumerate(unique_cats):
            indices = [i for i, x in enumerate(labels_manual) if x == cat_name]
            ax2.scatter(X_2d[indices, 0], X_2d[indices, 1],
                        color=palette[cat_idx], label=cat_name,
                        s=100, alpha=0.9, edgecolors='k', linewidth=0.5)

        ax2.set_title("B. Biological Ground Truth\n(Colored by Functional Category)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("t-SNE Dim 1")
        ax2.set_yticks([])  # 省略中间的Y轴刻度
        ax2.legend(title="Functional Category", loc='upper right', fontsize=10)

        # 标注文字 (Bio View)
        for i, name in enumerate(self.pathway_names):
            clean_name = name.split(" - ")[0]
            ax2.text(X_2d[i, 0] + 0.2, X_2d[i, 1] + 0.2, clean_name, fontsize=7, alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()

        print(f"Saved Comparison Plot: {save_path}")
        print(
            "Interpretation Guide: Compare the spatial grouping in A vs B. If distinct colors in B correspond to grouped clusters in A, the model has learned biology.")


    # =========================================================
    # Refactored Method 2: Shuffling Test (Structure Disruption)
    # =========================================================

    def run_shuffling_test(self, dataloader, disease_mapping, filename="Fig7B_Shuffling_Breakdown.pdf"):
        """
        [主入口] 执行扰动测试：对比 "原始输入" vs "打乱输入" 的分类准确率。
        流程：
        1. 调用计算核心，获取按癌症类型分组的统计数据。
        2. 调用绘图核心，生成对比柱状图。
        """
        print(f"\n>>> [START] Ablation Study: Data Shuffling Test...")

        # 1. 计算数据
        df_grouped = self._calculate_shuffling_stats(dataloader, disease_mapping, num_batches=100000)

        if df_grouped is None or df_grouped.empty:
            print("Thinking: Calculation failed or returned no data. Aborting plot.")
            return

        # 打印表格供检查
        print("\n>>> [DATA SUCCESS] Aggregated Results (Normalized):")
        print(df_grouped)

        # 2. 执行绘图
        self._plot_shuffling_results(df_grouped, filename)
        print(f">>> [FINISHED] Shuffling test complete.\n")

    def _calculate_shuffling_stats(self, dataloader, disease_mapping, num_batches=100):
        """
        [计算核心 - 私有方法] 负责模型推理、文本解析和数据聚合。
        Returns: pd.DataFrame (按 Group 聚合后的均值数据) 或 None
        """
        import pandas as pd
        import torch

        print("Thinking: Initializing temporary validator for strict parsing...")

        # 定义局部归一化函数
        def _normalize_group(name):
            control_aliases = {'health control', 'healthy', 'benign', 'normal', 'control', 'ct'}
            if name.lower() in control_aliases: return "Control"
            return name

        # 初始化验证器
        temp_validator = BioLLMValidator(
            model=self.model, tokenizer=self.tokenizer, device=self.device,
            disease_mapping=disease_mapping, valid_pathway_names=self.pathway_names
        )

        results = []
        self.model.eval()


        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                if i >= num_batches: break

                metabolite_tensor = batch['metabolite_tensor'].to(self.device)
                batch_size = metabolite_tensor.size(0)

                # 1. 解析 Ground Truth
                gt_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
                parsed_gts = [temp_validator.parse_generated_text(t) for t in gt_texts]

                # 2. 原始生成 (Original)
                gen_ids_orig = self.model.generate(
                    metabolite_tensor, self.tokenizer, max_new_tokens=100, do_sample=False
                )
                texts_orig = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids_orig]

                # 3. 打乱生成 (Shuffled)
                B, N, C, T = metabolite_tensor.shape
                perm_indices = torch.randperm(N)
                metabolite_shuffled = metabolite_tensor[:, perm_indices, :, :]
                gen_ids_shuf = self.model.generate(
                    metabolite_shuffled, self.tokenizer, max_new_tokens=100, do_sample=False
                )
                texts_shuf = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids_shuf]

                # 4. 逐样本统计
                for j in range(batch_size):
                    _, raw_gt_group, _ = parsed_gts[j]
                    gt_group = _normalize_group(raw_gt_group)
                    if gt_group == 'Unknown': continue

                    # --- Original Stats ---
                    _, raw_pred_o, _ = temp_validator.parse_generated_text(texts_orig[j])
                    acc_o = 1.0 if _normalize_group(raw_pred_o) == gt_group else 0.0

                    # --- Shuffled Stats ---
                    _, raw_pred_s, _ = temp_validator.parse_generated_text(texts_shuf[j])
                    acc_s = 1.0 if _normalize_group(raw_pred_s) == gt_group else 0.0

                    results.append({"Group": gt_group, "Acc_Orig": acc_o, "Acc_Shuf": acc_s})

        # 5. 数据聚合
        if not results:
            print("Thinking: No valid samples collected.")
            return None

        df = pd.DataFrame(results)
        # 按组求平均，并按原始准确率降序排列
        df_grouped = df.groupby("Group")[["Acc_Orig", "Acc_Shuf"]].mean().sort_values(by="Acc_Orig", ascending=False)
        return df_grouped

    def _plot_shuffling_results(self, df_grouped, filename):
        """
        [绘图核心 - 私有方法] 接收干净的数据，进行防御性绘图。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        print(f"Thinking: Starting plot generation for {filename}...")

        # --- Step 1: 数据准备与过滤 ---
        # 这里的 df_grouped 已经是聚合好的了，我们需要计算样本量来过滤
        # 为了简单，这里假设传入的数据已经是过滤好的，或者我们只画前N个
        plot_df = df_grouped.copy()  # 使用全部数据进行绘制

        if plot_df.empty:
            print("Thinking: Plot DataFrame is empty. Skipping.")
            return

        n_groups = len(plot_df)
        indices = np.arange(n_groups)
        width = 0.35

        # --- Step 2: 防御性绘图初始化 ---
        # 【关键】清除之前的任何全局状态，防止残留图像干扰
        plt.close('all')
        # 显式创建独立的 Figure 和 Axes
        fig, ax = plt.subplots(figsize=(14, 7))

        # --- Step 3: 绘制柱状图 ---
        # 使用 ax.bar 确保画在当前 axes 上
        rects1 = ax.bar(indices - width / 2, plot_df["Acc_Orig"], width,
                        label='Original Input (Structured)', color='#2980b9', alpha=0.9, edgecolor='k', linewidth=0.5)
        rects2 = ax.bar(indices + width / 2, plot_df["Acc_Shuf"], width,
                        label='Shuffled Input (Unstructured)', color='#c0392b', alpha=0.9, edgecolor='k', linewidth=0.5)

        # --- Step 4: 设置图表属性 ---
        ax.set_ylabel("Subtype Classification Accuracy", fontsize=12, fontweight='bold')
        ax.set_title("Impact of Metabolic Structure Disruption on Diagnostic Accuracy\n(Labels Normalized)",
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(indices)
        ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=11)

        # 设置 Y 轴范围，留出顶部空间给标注
        ax.set_ylim(0, 1.25)

        ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)  # 让网格线在柱子后面

        # --- Step 5: 添加数值标注 ---
        def add_labels(rects, is_original=True):
            for rect, (idx, row) in zip(rects, plot_df.iterrows()):
                height = rect.get_height()
                val = row['Acc_Orig'] if is_original else row['Acc_Shuf']

                # 基础数值标注
                ax.text(rect.get_x() + rect.get_width() / 2., height + 0.02,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold' if is_original else 'normal')

        add_labels(rects1, is_original=True)
        add_labels(rects2, is_original=False)

        # 添加显著下降的红色标注
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            drop = row['Acc_Orig'] - row['Acc_Shuf']
            if drop > 0.15:  # 阈值可调
                # 标在两个柱子中间的上方
                ax.text(i, max(row['Acc_Orig'], row['Acc_Shuf']) + 0.08,
                        f"▼ -{drop:.2f}",
                        ha='center', va='bottom', color='#d63031', fontsize=10, fontweight='bold')

        # --- Step 6: 保存并清理 ---
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        # 使用 fig.savefig 确保只保存当前图窗
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # 【关键】显式关闭当前图窗，释放内存，防止污染下一次绘图
        plt.close(fig)
        print(f"Thinking: Plot saved successfully to: {save_path}")

    # =========================================================
    # New Method 3: 类比推理测试 (Metabolic Arithmetic)
    # =========================================================
    # 定义功能分类映射 (基于你的白名单注释)
    def get_pathway_category(kegg_id):
        # 1.1 碳水化合物
        if kegg_id in ["hsa00010", "hsa00020", "hsa00030", "hsa00040", "hsa00051", "hsa00052", "hsa00053",
                       "hsa00500", "hsa00620", "hsa00630", "hsa00640", "hsa00650", "hsa00660", "hsa00562"]:
            return "Carbohydrates"

        # 1.2 能量 & 1.3 脂质 (合并或分开均可，这里按你的注释合并)
        if kegg_id in ["hsa00190", "hsa00910", "hsa00920", "hsa00061", "hsa00062", "hsa00071", "hsa00100",
                       "hsa00120", "hsa00140", "hsa00561", "hsa00564", "hsa00565", "hsa00600", "hsa00590",
                       "hsa00591", "hsa00592", "hsa01040"]:
            return "Energy & Lipids"

        # 1.4 核苷酸 & 1.5 氨基酸
        if kegg_id in ["hsa00230", "hsa00240", "hsa00250", "hsa00260", "hsa00270", "hsa00280", "hsa00290",
                       "hsa00310", "hsa00220", "hsa00330", "hsa00340", "hsa00350", "hsa00360", "hsa00380"]:
            return "Nucleotides & Amino Acids"

        # 1.6 其他氨基酸
        if kegg_id in ["hsa00410", "hsa00430", "hsa00440", "hsa00450", "hsa00460", "hsa00470", "hsa00480"]:
            return "Other Amino Acids"

        # 1.7 糖链 & 1.8 辅因子
        if kegg_id in ["hsa00520", "hsa00510", "hsa00531", "hsa00563", "hsa00601", "hsa00730", "hsa00740",
                       "hsa00750", "hsa00760", "hsa00770", "hsa00780", "hsa00785", "hsa00790", "hsa00670",
                       "hsa00830", "hsa00860", "hsa00130", "hsa00900"]:
            return "Glycans & Cofactors"

        # 补充：消化与神经递质
        if kegg_id in ["hsa04973", "hsa04974", "hsa04975", "hsa04976", "hsa04724", "hsa04727", "hsa04216"]:
            return "Digestion & Synapse"

        return "Others"

        # =========================================================
        # Updated Method 3: 批量类比推理验证 (Batch Analogical Reasoning)
        # =========================================================
    def run_analogical_reasoning(self, dataloader, target_group, suppress_list=None, boost_list=None,
                                 filename_suffix=""):
        """
        全量样本验证：
        1. 找出验证集中所有属于 target_group 的样本。
        2. 对每个样本进行两次生成：Original vs Perturbed (Suppress/Boost)。
        3. 统计：
            - Diagnostic Accuracy: 目标疾病是否还在？
            - Pathway Jaccard: 生成的通路与GT的一致性。
        4. 输出：全量样本对照报告 + 统计图表。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import re

        if suppress_list is None: suppress_list = []
        if boost_list is None: boost_list = []

        print(f"\n>>> Running Batch Analogical Reasoning on '{target_group}'...")
        print(f"    Target: Suppress {suppress_list} | Boost {boost_list}")

        # --- 1. 准备索引 ---
        def find_indices(keywords):
            indices = []
            names = []
            for kw in keywords:
                for i, p_name in enumerate(self.pathway_names):
                    if kw.lower() in p_name.lower():
                        indices.append(i)
                        names.append(p_name.split(" - ")[0])
                        break
            return indices, names

        suppress_idxs, suppress_names = find_indices(suppress_list)
        boost_idxs, boost_names = find_indices(boost_list)

        # 辅助函数：提取文本中提到的所有通路名称集合
        def extract_pathways_from_text(text):
            found_pathways = set()
            text_lower = text.lower()
            for pname in self.pathway_names:
                clean_name = pname.split(" - ")[0]
                if clean_name.lower() in text_lower:
                    found_pathways.add(clean_name)
            return found_pathways

        # 辅助函数：计算 Jaccard Index
        def calc_set_jaccard(set_a, set_b):
            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))
            return intersection / union if union > 0 else 0.0

        # --- 2. 收集目标样本 (Batch Collection) ---
        target_data = []  # List of dict: {'tensor': t, 'gt_text': txt}

        # 遍历整个 DataLoader
        for batch in dataloader:
            metabolite_tensor = batch['metabolite_tensor']
            gt_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]

            for i, text in enumerate(gt_texts):
                # 只要 Ground Truth 里包含目标疾病，就算作测试样本
                if target_group.lower() in text.lower():
                    target_data.append({
                        'tensor': metabolite_tensor[i].clone(),
                        'gt_text': text
                    })

        num_samples = len(target_data)
        print(f"    Found {num_samples} samples matching '{target_group}'. Starting inference...")

        if num_samples == 0:
            print("    No samples found. Skipping.")
            return

        # --- 3. 批量推理对比 (Original vs Perturbed) ---
        self.model.eval()

        # 统计容器
        correct_diag_orig = 0
        correct_diag_pert = 0

        jaccard_path_orig_sum = 0.0
        jaccard_path_pert_sum = 0.0

        # 完整记录用于输出文本
        all_results = []

        # 开始循环推理
        with torch.no_grad():
            for idx, item in enumerate(target_data):
                input_tensor = item['tensor'].unsqueeze(0).to(self.device)
                gt_text = item['gt_text']
                gt_pathways = extract_pathways_from_text(gt_text)  # Ground Truth 中的通路集合

                # A. 原始生成 (Baseline)
                gen_ids_base = self.model.generate(input_tensor, self.tokenizer, max_new_tokens=80, do_sample=False)
                text_base = self.tokenizer.decode(gen_ids_base[0], skip_special_tokens=True)

                # B. 扰动生成 (Perturbed)
                original_mask_row = self.model.pathway_mask.clone()
                try:
                    # 修改 Mask
                    for m_idx in suppress_idxs: self.model.pathway_mask[m_idx] = 0.0  # Mask (抑制)
                    for m_idx in boost_idxs:    self.model.pathway_mask[m_idx] = 10.0  # Emphasize (强调)

                    # 生成
                    gen_ids_pert = self.model.generate(input_tensor, self.tokenizer,
                                                       max_new_tokens=80,
                                                       temperature=0.6,
                                                       do_sample=True)
                    text_pert = self.tokenizer.decode(gen_ids_pert[0], skip_special_tokens=True)
                finally:
                    # 恢复 Mask
                    self.model.pathway_mask = original_mask_row

                # --- C. 计算指标 ---

                # 1. 诊断准确率 (Diagnostic Accuracy)
                # 检查生成的文本是否还包含目标疾病名称
                is_correct_orig = 1 if target_group.lower() in text_base.lower() else 0
                is_correct_pert = 1 if target_group.lower() in text_pert.lower() else 0

                correct_diag_orig += is_correct_orig
                correct_diag_pert += is_correct_pert

                # 2. 通路 Jaccard (Generated Pathways vs Ground Truth Pathways)
                pred_pathways_orig = extract_pathways_from_text(text_base)
                pred_pathways_pert = extract_pathways_from_text(text_pert)

                jc_orig = calc_set_jaccard(gt_pathways, pred_pathways_orig)
                jc_pert = calc_set_jaccard(gt_pathways, pred_pathways_pert)

                jaccard_path_orig_sum += jc_orig
                jaccard_path_pert_sum += jc_pert

                # 收集记录
                all_results.append({
                    "id": idx + 1,
                    "gt": gt_text,
                    "orig": text_base,
                    "pert": text_pert,
                    "acc_orig": is_correct_orig,
                    "acc_pert": is_correct_pert,
                    "jc_orig": jc_orig,
                    "jc_pert": jc_pert
                })

        # --- 4. 统计结果 ---
        avg_acc_orig = correct_diag_orig / num_samples
        avg_acc_pert = correct_diag_pert / num_samples

        avg_jc_orig = jaccard_path_orig_sum / num_samples
        avg_jc_pert = jaccard_path_pert_sum / num_samples

        print(f"    [Diagnostic Acc] Original: {avg_acc_orig:.2%} -> Perturbed: {avg_acc_pert:.2%}")
        print(f"    [Pathway Jaccard] Original: {avg_jc_orig:.4f} -> Perturbed: {avg_jc_pert:.4f}")

        # --- 5. 绘图 (Double Bar Chart) ---
        plt.figure(figsize=(10, 6))

        categories = ['Diagnostic Accuracy', 'Pathway Jaccard (vs GT)']
        orig_vals = [avg_acc_orig, avg_jc_orig]
        pert_vals = [avg_acc_pert, avg_jc_pert]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        rects1 = ax.bar(x - width / 2, orig_vals, width, label='Original', color='#95a5a6')
        rects2 = ax.bar(x + width / 2, pert_vals, width, label='Perturbed (Mask/Boost)', color='#e74c3c')

        ax.set_ylabel('Score')
        ax.set_title(
            f'Impact of Semantic Perturbation: {target_group}\n(Suppress: {len(suppress_list)}, Boost: {len(boost_list)})')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.1)
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        save_plot_name = f"Fig7C_PerturbationStats_{target_group}_{filename_suffix}.pdf"
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_plot_name))
        plt.close()
        print(f"    Saved Plot: {save_plot_name}")

        # --- 6. 生成文本报告 (TXT - Full Dump) ---
        txt_name = f"Fig7C_Perturbation_FullReport_{target_group}_{filename_suffix}.txt"
        with open(os.path.join(self.save_dir, txt_name), "w", encoding='utf-8') as f:
            f.write(f"MassLinker Semantic Perturbation Report (N={num_samples})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Target Group:     {target_group}\n")
            f.write(f"Masked (Suppress): {suppress_names}\n")
            f.write(f"Emphasized (Boost): {boost_names}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Summary Metrics:\n")
            f.write(
                f"  Diagnostic Accuracy: {avg_acc_orig:.2%} (Orig) -> {avg_acc_pert:.2%} (Pert) | Diff: {avg_acc_pert - avg_acc_orig:.2%}\n")
            f.write(
                f"  Pathway Jaccard:     {avg_jc_orig:.4f} (Orig) -> {avg_jc_pert:.4f} (Pert) | Diff: {avg_jc_pert - avg_jc_orig:.4f}\n")
            f.write("=" * 80 + "\n\n")
            f.write(">>> FULL EXAMPLES (Ground Truth included) <<<\n")

            for res in all_results:
                f.write(f"Sample ID: {res['id']}\n")
                f.write(f"  [Ground Truth]: {res['gt']}\n")
                f.write(f"  [Original]    : {res['orig']}\n")
                f.write(f"      -> Diag Correct: {res['acc_orig']} | Jaccard: {res['jc_orig']:.4f}\n")
                f.write(f"  [Perturbed]   : {res['pert']}\n")
                f.write(f"      -> Diag Correct: {res['acc_pert']} | Jaccard: {res['jc_pert']:.4f}\n")
                f.write("-" * 50 + "\n")

        print(f"    Saved Full Text Report: {txt_name}")

        # =========================================================
        # Method 4 (Revised): Hierarchical Diagnosis (With Benign Check & Filter)
        # =========================================================
    def run_hierarchical_diagnosis(self, dataloader, disease_mapping, filename_prefix="Fig_Matrix"):
        """
        生成三个层级的混淆矩阵报告（带异常值过滤）：
        1. Binary Diagnosis: 良性 vs 恶性 (基于全量数据)
        2. Benign Discrimination: 在正确识别为阴性(True Negative)的样本中，区分 Benign vs Control
        3. Cancer Subtyping: 在正确识别为阳性(True Positive)的样本中，区分具体癌症类型
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        print(f"\n>>> [START] Hierarchical Diagnostic Analysis (Strict Filtering)...")

        # --- 1. 定义动态白名单与归一化逻辑 ---

        # 获取所有已知的癌症名称
        known_cancers = list(disease_mapping.values())

        # 定义合法的组别白名单 (用于过滤幻觉)
        # 注意：这里我们显式加入 'Benign'，不再将其视为 Control 的同义词
        VALID_GROUPS = set([g.lower() for g in known_cancers] + ['control', 'healthy', 'normal', 'benign'])

        # 归一化函数：核心是把 Benign 独立出来
        def _normalize_group(name):
            name_lower = name.lower()

            # A. 处理健康对照
            if name_lower in ['healthy', 'normal', 'control', 'health control', 'ct']:
                return "Control"

            # B. 处理良性/增生 (关键修改：独立返回 Benign)
            if 'benign' in name_lower:
                return "Benign"

            # C. 处理癌症 (从 Mapping 中匹配标准名)
            for standard_name in known_cancers:
                if standard_name.lower() in name_lower:
                    return standard_name

            return "Unknown"

        # 初始化验证器
        validator = BioLLMValidator(
            model=self.model, tokenizer=self.tokenizer, device=self.device,
            disease_mapping=disease_mapping, valid_pathway_names=self.pathway_names
        )

        data_records = []
        self.model.eval()

        print("Thinking: Collecting and filtering predictions...")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                metabolite_tensor = batch['metabolite_tensor'].to(self.device)

                # --- Ground Truth 解析 ---
                gt_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
                parsed_gts = [validator.parse_generated_text(t) for t in gt_texts]

                # --- Model Prediction 解析 ---
                gen_ids = self.model.generate(
                    metabolite_tensor, self.tokenizer, max_new_tokens=80, do_sample=False
                )
                pred_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_ids]
                parsed_preds = [validator.parse_generated_text(t) for t in pred_texts]

                for j in range(len(gt_texts)):
                    gt_diag, raw_gt_group, _ = parsed_gts[j]
                    pred_diag, raw_pred_group, _ = parsed_preds[j]

                    # 1. 归一化
                    gt_group = _normalize_group(raw_gt_group)
                    pred_group = _normalize_group(raw_pred_group)

                    # 2. 严格过滤 (Filter Logic)
                    # 如果 GT 无效（比如解析失败），跳过
                    if gt_group == 'Unknown':
                        continue

                    # 如果 预测出的组别 不在白名单里（幻觉），跳过
                    # 或者 预测出的 Diagnosis 不是 Positive/Negative，跳过
                    if pred_group == 'Unknown':
                        # print(f"Skipping hallucination: {raw_pred_group}") # Debug用
                        continue
                    if pred_diag not in ['Positive', 'Negative']:
                        continue

                    data_records.append({
                        "GT_Diag": gt_diag,
                        "Pred_Diag": pred_diag,
                        "GT_Group": gt_group,
                        "Pred_Group": pred_group
                    })

        df = pd.DataFrame(data_records)
        if df.empty:
            print("Error: No valid data after filtering.")
            return

        # =========================================================
        # Matrix 1: Binary Diagnosis (Detection Capability)
        # =========================================================
        print("Thinking: Generating Matrix 1 (Binary Diagnosis)...")
        labels_bin = ["Positive", "Negative"]

        # 确保数据中包含这些标签，否则 confusion_matrix 会报错或缺列
        cm_bin = confusion_matrix(df["GT_Diag"], df["Pred_Diag"], labels=labels_bin)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels_bin, yticklabels=labels_bin, annot_kws={"size": 14})
        plt.title("Matrix 1: Binary Diagnosis\n(Detection Capability)", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Diagnosis")
        plt.ylabel("Ground Truth Diagnosis")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{filename_prefix}_1_Binary.pdf"))
        plt.close()

        # =========================================================
        # Matrix 2: Benign Discrimination (True Negatives Only)
        # 逻辑：在 GT=Neg 且 Pred=Neg 的样本中，看 Control vs Benign
        # =========================================================
        print("Thinking: Generating Matrix 2 (Benign Discrimination in True Negatives)...")

        # 筛选：诊断正确地判断为“阴性”的样本
        df_tn = df[
            (df["GT_Diag"] == "Negative") &
            (df["Pred_Diag"] == "Negative")
            ]

        if not df_tn.empty:
            # 这里的标签只关注 Control 和 Benign
            # 即使 GT 数据集里全是 Control，如果模型预测出了 Benign，这里也能体现出来
            labels_neg = ["Control", "Benign"]

            # 为了防止 labels 在数据中完全不存在导致报错，做个检查
            # 如果数据里全是 Control，我们强制用这个 labels 列表，这样 Benign 一列会是全 0，正是我们想看到的
            cm_neg = confusion_matrix(df_tn["GT_Group"], df_tn["Pred_Group"], labels=labels_neg)

            plt.figure(figsize=(6, 5))
            # 使用绿色系区分阴性分析
            sns.heatmap(cm_neg, annot=True, fmt="d", cmap="Greens", cbar=False,
                        xticklabels=labels_neg, yticklabels=labels_neg, annot_kws={"size": 14})
            plt.title("Matrix 2: Differentiation of Negatives\n(True Negatives Only: Control vs Benign)",
                      fontsize=12, fontweight='bold')
            plt.xlabel("Predicted Condition")
            plt.ylabel("Ground Truth Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{filename_prefix}_2_BenignDiscrimination.pdf"))
            plt.close()
        else:
            print("Warning: No True Negative samples found.")

        # =========================================================
        # Matrix 3: Cancer Subtyping (True Positives Only)
        # 逻辑：在 GT=Pos 且 Pred=Pos 的样本中，看癌症分类准确度
        # =========================================================
        print("Thinking: Generating Matrix 3 (Cancer Subtypes in True Positives)...")

        # 筛选：诊断正确地判断为“阳性”的样本
        df_tp = df[
            (df["GT_Diag"] == "Positive") &
            (df["Pred_Diag"] == "Positive")
            ]

        if not df_tp.empty:
            # 动态获取出现的癌症标签，并排序
            present_cancers = sorted(list(set(df_tp["GT_Group"].unique()) | set(df_tp["Pred_Group"].unique())))

            # 移除 Control/Benign (理论上 True Positive 不应该包含这些，但为了稳健性)
            present_cancers = [c for c in present_cancers if c not in ['Control', 'Benign']]

            if present_cancers:
                cm_subtype = confusion_matrix(df_tp["GT_Group"], df_tp["Pred_Group"], labels=present_cancers)

                plt.figure(figsize=(10, 8))
                # 使用橙色系区分阳性分析
                sns.heatmap(cm_subtype, annot=True, fmt="d", cmap="Oranges", cbar=False,
                            xticklabels=present_cancers, yticklabels=present_cancers)
                plt.title("Matrix 3: Cancer Subtype Classification\n(True Positives Only)", fontsize=12,
                          fontweight='bold')
                plt.xlabel("Predicted Subtype")
                plt.ylabel("Ground Truth Subtype")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{filename_prefix}_3_PositiveSubtype.pdf"))
                plt.close()
            else:
                print("Warning: No valid cancer labels found in True Positives.")
        else:
            print("Warning: No True Positive samples found.")

        print(f">>> [FINISHED] Hierarchical diagnosis plots saved.")

if __name__ == "__main__":
    print(f"Running on: {CONFIG['DEVICE']}")

    # 1. 构建特征通路字典 (Signature Map)
    print("Building Signature Map from SHAP analysis...")
    signature_map = {}

    for i in range(1, 8):
        disease_name = DISEASE_MAPPING[i]
        file_path = os.path.join(CONFIG["SIGNATURE_DIR"], f'KEGG_Enriched{i}.csv')

        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}, skipping {disease_name}")
            continue

        df = pd.read_csv(file_path)

        df_enrich = df[df['P_Value'] <= 0.05]
        df_enrich = df_enrich[df_enrich['Pathway_ID'].isin(ENDOGENOUS_PATHWAY_IDS)]

        df_PPS = df.nlargest(10, ['PPS'])
        df_PPS = df_PPS[df_PPS['Pathway_ID'].isin(ENDOGENOUS_PATHWAY_IDS)]

        sigs = list(set(df_enrich['Pathway_Name'].tolist() + df_PPS['Pathway_Name'].tolist()))
        signature_map[disease_name] = sigs
        print(f"{disease_name}: {len(sigs)} signature pathways.")

    # 2. 加载数据
    print("Loading Tensor Data...")
    data_origin = joblib.load(CONFIG["DATA_PATH"])
    full_data = data_origin.samples

    full_groups = [i.item() if hasattr(i, 'item') else i for i in data_origin.label]
    full_status = [int(i.item()) if hasattr(i, 'item') else int(i) for i in data_origin.is_positive]

    # 3. 初始化 Tokenizer & Mask
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(CONFIG["MODEL_PATH"])
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Error: Local GPT-2 weights not found.")
        exit()

    mask, pathway_names = create_pathway_mask(CONFIG["CSV_PATH"], whitelist_ids=ENDOGENOUS_PATHWAY_IDS)
    print(f"Mask Shape: {mask.shape}")

    # 4. Dataset & DataLoader
    synthesizer = ReportSynthesizer(pathway_names)
    dataset = MetabolomicsDataset(
        num_samples=len(full_data),
        rbf_data=full_data,
        groups=full_groups,
        is_positive=full_status,
        synthesizer=synthesizer,
        tokenizer=tokenizer,
        pathway_mask=mask,
        signature_map=signature_map
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    # joblib.dump(train_loader,r'F:\Mass Linker_BERT\medium_from_4080S\train_loader.joblib')
    # joblib.dump(val_loader, r'F:\Mass Linker_BERT\medium_from_4080S\val_loader.joblib')
    train_loader=joblib.load(r'F:\Mass Linker_BERT\medium_from_4080S\train_loader.joblib')
    val_loader=joblib.load(r'F:\Mass Linker_BERT\medium_from_4080S\val_loader.joblib')
    model = MassLinkerBioLLM(mapping_mask=mask, model_path=CONFIG["MODEL_PATH"], pathway_dim=1024)

    trainer = MetabolicTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=CONFIG["DEVICE"],
        lr=CONFIG["LR"],
        save_dir=CONFIG["CHECKPOINT_DIR"]
    )
    # trainer.run(epochs=CONFIG["EPOCHS"])
    best_model_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "model260.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for validation...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found, validating with current weights.")
    # trainer.run(epochs=CONFIG["EPOCHS"])
    # validator = BioLLMValidator(model,
    #                             tokenizer,
    #                             CONFIG["DEVICE"],
    #                             disease_mapping=DISEASE_MAPPING,
    #                             valid_pathway_names=pathway_names)
    # metrics = validator.run_evaluation(val_loader,output_file=CONFIG["OUTPUT_VALID_FILE"])
    # ... (Validator 运行之后) ...
    # ==========================================
    # 7. 生成 Figure 5 可视化结果
    # ==========================================
    # print("\n>>> Generating Figure 5 Visualizations...")
    # print(sum(p.numel() for p in model.parameters()))
    visualizer = BioLLMVisualizer(
        model,
        tokenizer,
        CONFIG["DEVICE"],
        #CONFIG["CHECKPOINT_DIR"],
        r'F:\Mass Linker_BERT\BERTplot',
        pathway_names=pathway_names,
        csv_path=CONFIG["CSV_PATH"],  # <--- 新增
        whitelist_ids=ENDOGENOUS_PATHWAY_IDS  # <--- 新增
    )
    # visualizer.run_hierarchical_diagnosis(val_loader, disease_mapping=DISEASE_MAPPING)
    # 1. 语义一致性
    # visualizer.plot_semantic_alignment(val_loader, num_batches=200, use_attention_as_x=True, filter_zeros=True,
    #                                    filename=r"F:\Mass Linker_BERT\medium_from_4080S\Fig5B_SemanticAlignment.pdf")

    # 2. 混淆矩阵 (如果有现成数据最好，没有的话可以略过或用 validator 的结果)
    # visualizer.plot_confusion_matrix(true_groups, pred_groups)

    # 3. Attention Map (带名字版)
    # 取一个样本jixia
    model.eval()
    total_scanned = 0
    found_count = 0
    SCAN_LIMIT = 200
    print(f"Scanning validation set for 'Endometrial Cancer' samples...")
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        metabolite_tensors = batch['metabolite_tensor']
        input_ids_batch = batch['input_ids']
        current_batch_size = metabolite_tensors.size(0)
        for i in range(current_batch_size):
            total_scanned += 1
            gt_ids = input_ids_batch[i]
            gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
            if "positive" in gt_text.lower():
                sample_tensor = metabolite_tensors[i].unsqueeze(0).to(CONFIG["DEVICE"])
                with torch.no_grad():
                    gen_ids = model.generate(sample_tensor, tokenizer)
                    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                # print(f"\n[Match #{found_count + 1} | Sample {total_scanned}]")
                # print(f"  GT  : {gt_text}")
                # print(f"  Pred: {gen_text}")
                visualizer.plot_attention_map(sample_tensor, gen_text,
                                              filename=f"Fig5C_Attention{found_count}.pdf",
                                              custom_colors=['#FCEED4', '#D15D73'])
                visualizer.plot_case_study(
                    gt_text,
                    gen_text,
                    case_id=f"Val_Sample_{total_scanned}",
                    filename=f"Fig5D_Case{found_count}.png"
                )

                found_count += 1

    # visualizer.run_all_perturbations(val_loader)
    # visualizer.plot_biological_concordance(val_loader)
    # visualizer.plot_semantic_deviation(
    #     val_loader,
    #     baseline_group=["health"],
    #     target_group="RCC",
    #     filename="Fig6_RCC_vs_GrandBaseline.pdf"
    # )
    # visualizer.plot_logit_shift(
    #     val_loader,
    #     baseline_group=["health"],
    #     target_group="RCC"
    # )
    # rcc_drugs = ["Sunitinib", "Sorafenib", "Everolimus", "Bevacizumab", "Metformin", "Statin"]
    # visualizer.plot_therapeutic_suggestion(val_loader, drug_names=rcc_drugs, target_group="RCC")
    #
    # visualizer.plot_comprehensive_pharmacist(val_loader,topn=3)
    # # 针对 Ovarian Cancer，Paclitaxel (紫杉醇) 和 Cisplatin (顺铂) 是常用的
    # ov_drugs = ["Paclitaxel", "Cisplatin", "Carboplatin", "Olaparib", "Bevacizumab"]
    # visualizer.plot_therapeutic_suggestion(val_loader, drug_names=ov_drugs, target_group="Ovarian Cancer")
    #
    # # 2. 尝试无监督亚型发现
    # visualizer.plot_latent_subtyping(val_loader, target_group="RCC")
    # ... (之前的 visualizer 代码) ...

    # ==========================================
    # 8. Figure 7: 代谢语义深度验证 (Deep Semantic Validation)
    # ==========================================
    # print("\n>>> Generating Figure 7: Semantic Validation...")
    #
    # # 7.1 向量空间 (生成全标注图)
    # visualizer.plot_vector_space_analysis(val_loader)
    #
    # # 7.2 扰动测试 (保持不变)
    # visualizer.run_shuffling_test(train_loader, disease_mapping=DISEASE_MAPPING)
    #
    # # 7.3 类比推理 (用户自定义参数)
    # target_disease = "endometrial cancer"
    # visualizer.run_analogical_reasoning(
    #     val_loader,
    #     target_group=target_disease,
    #     suppress_list=['Glycolysis / Gluconeogenesis - Homo sapiens (human)'],  # 关键词列表
    #     boost_list=[],  # 关键词列表
    #     filename_suffix="ExpA_Ether"
    # )
    # === 实验 B: 抑制 TCA，增强 Retinol ===
    # visualizer.run_analogical_reasoning(
    #     val_loader,
    #     target_group='Cervical Cancer',
    #     suppress_list=[],  # 关键词列表
    #     boost_list=["Ether lipid metabolism - Homo sapiens (human)"],  # 关键词列表
    #     filename_suffix="ExpB_Ether"
    # )
    # visualizer.run_analogical_reasoning(
    #     val_loader,
    #     target_group='Ovarian Cancer',
    #     suppress_list=["Ether lipid metabolism - Homo sapiens (human)",
    #                 'Glycolysis / Gluconeogenesis - Homo sapiens (human)'],  # 关键词列表
    #     boost_list=[],  # 关键词列表
    #     filename_suffix="ExpD_Ether"
    # )
    print("\nAll Semantic Validations Completed!")
