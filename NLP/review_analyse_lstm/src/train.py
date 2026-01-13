import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from NLP.review_analyse_lstm.src import config
from NLP.review_analyse_lstm.src.dataset import get_dataloader
from NLP.review_analyse_lstm.src.model import ReviewAnalyzeModel
from NLP.review_analyse_lstm.src.tokenizer import JiebaTokenizer


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
    """

    :return:
    """
    # 选择训练设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据加载器
    dataloader = get_dataloader()

    # 从词汇表文件加载分词器
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 初始化模型，使用分词器的词汇表大小和填充值索引
    model = ReviewAnalyzeModel(tokenizer.vocab_size, padding_idx=0).to(device)

    # 定义损失函数：带logits的二元交叉熵损失
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 定义优化器：Adam优化器，学习率从配置文件读取
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 创建TensorBoard写入器，用于记录训练日志
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 初始化最佳损失值为无穷大
    best_loss = float('inf')

    # 开始训练循环，迭代指定的轮数
    for epoch in range(1, config.EPOCH_SIZE + 1):
        # 训练一个epoch，获取平均损失
        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)

        # 将当前epoch的平均损失写入TensorBoard日志
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 如果当前损失是历史最佳（最小），则保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')


if __name__ == '__main__':
    train()
