import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from NLP.trans_seq2seq.src import config
from NLP.trans_seq2seq.src.dataset import get_dataloader
from NLP.trans_seq2seq.src.model import TranslationModel, TranslationEncoder, TranslationDecoder
from NLP.trans_seq2seq.src.tokenizer import EnglishTokenizer, ChineseTokenizer


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    训练模型一个epoch
    Args:
        model: 待训练的神经网络模型
        dataloader: 训练数据的数据加载器
        loss_function: 损失函数
        optimizer: 优化器
        device: 训练设备（CPU或GPU）
    Returns:
        float: 当前epoch的平均损失值
    """
    total_loss = 0
    model.train()

    # 遍历训练数据进行前向传播、反向传播和参数更新
    for inputs, targets in tqdm(dataloader, desc="train"):
        # encoder_inputs.shape: [batch_size, src_seq_len]
        # targets.shape: [batch_size, tgt_seq_len]
        encoder_inputs, targets = inputs.to(device), targets.to(device)
        decoder_inputs = targets[:, :-1] # shape: [batch_size, tgt_seq_len - 1]
        decoder_targets = targets[:, 1:] # shape: [batch_size, tgt_seq_len - 1]

        # context_vector.shape: [batch_size, hidden_size]
        context_vector = model.encoder(encoder_inputs)

        # [1, batch_size, hidden_size
        decoder_hidden = context_vector.unsqueeze(0)

        decoder_outputs = []

        sql_len = decoder_inputs.shape[1]

        for i in range(sql_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1) # shape: [batch_size, 1]
            decoder_output, hidden = model.decoder(decoder_input, decoder_hidden)

            decoder_outputs.append(decoder_output)

        # decoder_outputs.shape: [batch_size, seq_len, hidden_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        decoder_targets = decoder_targets.reshape(-1)

        loss_fn = loss_function(decoder_outputs, decoder_targets)

        loss_fn.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss_fn.item()

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
    en_tokenizer = EnglishTokenizer.from_vocab(config.RAW_DATA_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.RAW_DATA_DIR / 'zh_vocab.txt')

    # 初始化模型，使用分词器的词汇表大小和填充值索引
    model = TranslationModel(zh_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.vocab_size,
                             en_tokenizer.pad_token_index).to(device)

    # 定义损失函数：带logits的二元交叉熵损失
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

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
