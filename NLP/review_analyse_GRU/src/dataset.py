import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from NLP.review_analyse_GRU.src import config


class ReviewAnalyzeDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['label'], dtype=torch.float)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    dataset = ReviewAnalyzeDataset(config.PROCESSED_DATA_DIR / ('train.json' if train else 'test.json'))
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
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
