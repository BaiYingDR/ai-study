import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from NLP.input_method.src import config


class InputMethodDataset(Dataset):
    """
    负责定义数据的读取逻辑（比如从文件 / 内存中加载样本、做基础预处理、标签映射等）

    """

    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    负责批量加载、打乱、并行读取数据
    是 “数据搬运工”，基于Dataset提供可迭代的批量数据，直接喂给模型训练
    :param train:
    :return:
    """
    datasets = InputMethodDataset(config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl"))
    """
        batch_size: 
            每个批次包含的样本数量，即DataLoader每次迭代返回的样本数。
            比如batch_size=32，则每次迭代会返回 32 个样本的特征和标签（维度为(32, ...)）
            
        shuffle：
            是否在每个 epoch（训练轮次）开始时打乱数据集的样本顺序
    """
    return DataLoader(datasets, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(False)

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)   # [BATCH_SIZE,SEQ_LEN]
        print(target_tensor.shape)  # [BATCH_SIZE]
        break

    print(len(train_dataloader))
    print(len(test_dataloader))
