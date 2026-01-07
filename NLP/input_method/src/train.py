import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from NLP.input_method.src import config
from NLP.input_method.src.dataset import get_dataloader
from NLP.input_method.src.model import InputMethodModule
from NLP.input_method.src.tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
        训练一个 epoch。

        :param model: 输入法模型。
        :param dataloader: 数据加载器。
        :param loss_function: 损失函数。
        :param optimizer: 优化器。
        :param device: 设备。
        :return: 平均损失。
    """

    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc="train"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_dataloader()

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    model = InputMethodModule(vocab_size=tokenizer.vocab_size).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))
    best_loss = float('inf')

    for epoch in range(1, config.EPOCH_SIZE + 1):
        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')


if __name__ == '__main__':
    train()
