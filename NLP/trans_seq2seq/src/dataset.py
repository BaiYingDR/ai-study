import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from NLP.trans_seq2seq.src import config


class TranslationDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        @params idx: 由dataset自己随机生成的index
        """
        input = torch.tensor(self.data[idx]['zh'], dtype=torch.long)
        target = torch.tensor(self.data[idx]['en'], dtype=torch.long)
        return input, target


def collate_fn(batch):
    """
    批处理数据整理函数，用于将一批数据样本整理成批次张量

    参数:
        batch: 包含多个数据样本的列表，每个样本是一个元组 (input_tensor, target_tensor)

    返回:
        tuple: 包含两个元素的元组
            - 第一个元素：填充后的输入张量序列，形状为 (batch_size, max_seq_len, *)
            - 第二个元素：填充后的目标张量序列，形状为 (batch_size, max_seq_len, *)
    """

    # print(batch)
    # 提取批次中的所有输入张量和目标张量
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    # 对输入张量和目标张量分别进行填充，使它们具有相同的序列长度
    return (pad_sequence(input_tensors, batch_first=True, padding_value=0),
            pad_sequence(target_tensors, batch_first=True, padding_value=0))


def get_dataloader(train=True):
    """
    获取数据加载器

    Args:
        train (bool): 是否使用训练数据，默认为True。如果为True则加载train.json，
                     否则加载test.json

    Returns:
        DataLoader: 配置好的数据加载器对象，用于批量加载翻译数据集
    """
    # 创建翻译数据集实例，根据train参数决定加载训练集或测试集
    dataset = TranslationDataset(config.PROCESSED_DATA_DIR / ('train.json' if train else 'test.json'))

    # 创建数据加载器，设置批次大小、是否打乱数据以及自定义的数据合并函数
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    # 简单测试数据加载器
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(False)

    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape, target_tensor.shape)
        break
