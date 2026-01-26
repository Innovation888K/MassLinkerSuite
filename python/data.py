import os
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import utils
import numpy as np


class ExcelDataset(Dataset):
    '''
        Used for constructing the datasets
        data_path should contain all xlsx files converted from MassLinker
        target_path should be a xlsx file contain filename、label and is_positive(1 or 0)
    '''

    def __init__(self, data_path, target_path="target.xlsx", mode='origin'):
        self.data_path = data_path
        self.target_path = target_path
        self.samples, self.label, self.name, self.is_positive = self.load_data(data_path, target_path, mode=mode)

    def load_data(self,
                  data_path: str,
                  target_path: str,
                  mode='origin'
                  ):
        ret = []
        target_df = pd.read_excel(target_path)
        labels = []
        name = []
        is_positives = []
        for filename in tqdm(os.listdir(data_path)):
            if mode == "enhance":
                split = filename.split('-')
                add_name = ""
                for i in range(1, len(split)):
                    add_name += split[i]
                    add_name += "-"
                add_name = add_name[:-1]
            else:
                add_name = filename
            ret.append([])
            if add_name not in target_df.iloc[:, 0].tolist():
                raise Exception(f"{add_name} not in target xlsx")
            condition = target_df['filename'] == add_name
            label = target_df["label"][condition]
            is_positive = torch.tensor(target_df["is_positive"][condition].item())
            is_positives.append(is_positive)
            labels.append(label)
            name.append(add_name)
            f_name = os.path.join(data_path, filename, filename + ".xlsx")
            sheets = pd.read_excel(f_name, sheet_name=None)
            valid_sheets = []
            for sheet_name, temp_df in sheets.items():
                if not temp_df.empty:
                    temp = torch.tensor(temp_df.to_numpy(), dtype=torch.float32)
                    temp = temp.unsqueeze(0).unsqueeze(0)
                    valid_sheets.append(temp)
            if valid_sheets:
                for sheets in valid_sheets:
                    sheets = sheets.permute(0, 1, 3, 2)
                    for wordcount in range(int(sheets.shape[2] / 3)):
                        ret[-1].append(sheets[0][0][3 * (wordcount):3 * (wordcount + 1)])
                ret[-1] = torch.stack(ret[-1])
        return torch.stack(ret), labels, name, torch.stack(is_positives)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], [self.label[i] for i in idx], [self.name[i] for i in idx], self.is_positive[idx], [
            self.classes[i] for i in idx]

    def save_dataset(self, save_path):
        joblib.dump(self, save_path)

    def gen_classes(self):
        classes_list = []
        unique_classes = []
        for i in range(len(self.is_positive)):
            if self.is_positive[i] == 0:
                classes_list.append(0)
            else:
                if self.label[i].item() in unique_classes:
                    classes_list.append(unique_classes.index(self.label[i].item()) + 1)
                else:
                    unique_classes.append(self.label[i].item())
                    classes_list.append(len(unique_classes))
        self.classes = classes_list


def load_data(filename, n_part=5):
    x = joblib.load(filename)
    x.gen_classes()
    split_idx = split_dataset(len(x), n_part)
    return x, split_idx


def split_dataset(n: int, m: int) -> list[list[int]]:
    numbers = list(range(n))
    random.shuffle(numbers)
    base_size = n // m
    remainder = n % m
    result = [[] for _ in range(m)]
    current_index = 0
    for i in range(m):
        current_partition_size = base_size + (1 if i < remainder else 0)
        result[i] = numbers[current_index: current_index + current_partition_size]
        current_index += current_partition_size
    return result


def data_transform(data):
    return data.reshape(data.shape[0], -1)


def data_rev_transform(data):
    return data.reshape(1, 5194, 3, 20)[0]


def gen_feature_names():
    names = torch.load('names.pth')
    feature_names = list()
    for i in range(20):
        for j in names:
            temp = j.split("_")
            feature_names.append(temp[1] + '_' + temp[2] + str(i))
    return feature_names


def data_center_rev(data):
    data.samples[:, :, 1, :] = 1800 - data.samples[:, :, 1, :]
    return data


def concat_dataset(data_dir, init_polarity_dir):
    init_polarity = pd.read_excel(init_polarity_dir, dtype='string')
    dataset_list = []
    for file in os.listdir(data_dir):
        if file.split('.')[0] in list(init_polarity["dataset"]):
            
            if (init_polarity["init_polarity"][init_polarity["dataset"] == file.split('.')[0]] == 'low').iloc[0]:
                dataset_list.append(data_center_rev(joblib.load(os.path.join(data_dir, file))))
            else:
                dataset_list.append(joblib.load(os.path.join(data_dir, file)))
    init_dataset = dataset_list[0]
    for i in range(1, len(dataset_list)):
        init_dataset.label += dataset_list[i].label
        init_dataset.name += dataset_list[i].name
        init_dataset.is_positive = torch.concat((init_dataset.is_positive, dataset_list[i].is_positive))
        init_dataset.samples = torch.concat((init_dataset.samples, dataset_list[i].samples))
    init_dataset.gen_classes()
    return init_dataset


if __name__ == "__main__":
    dataset = ExcelDataset(r'F:\质谱不鉴定数据\8764\4', r"F:\质谱不鉴定数据\源数据\targets\8764_reduced.xlsx",mode='enhance')
    dataset.save_dataset(r"F:\质谱不鉴定数据\源数据\合并后joblib数据\8764_reduced_enhanced.joblib")

    # total_dataset = concat_dataset(r'F:\质谱不鉴定数据\源数据\数据增强后的joblib',
    #                                r'D:\git\MassLinker\init_polarity.xlsx')
    # joblib.dump(total_dataset, filename=r"F:\质谱不鉴定数据\源数据\merged_dataset_enhanced_total.joblib")
    # dataset = joblib.load(r"F:\质谱不鉴定数据\源数据\merged_dataset_origin.joblib")
    # plot1 = [i.reshape(1, i.shape[0] * i.shape[1] * i.shape[2])[0].numpy() for i in dataset.samples]
    # group = dataset.classes
    # utils.low_dim_plots(plot1, group, dim=2)
