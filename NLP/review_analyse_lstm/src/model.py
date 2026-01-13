import torch
from tensorboard import summary
from torch import nn

from NLP.review_analyse_lstm.src import config


class ReviewAnalyzeModel(nn.Module):
    """
    评论分析模块，使用LSTM进行情感分析或评论分类
    """

    def __init__(self, vocab_size, padding_idx):
        """
        初始化评论分析模块
        
        Args:
            vocab_size: 词汇表大小
            padding_idx: 填充token的索引
        """
        super().__init__()
        # 词嵌入层，将词汇索引转换为向量表示
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        # LSTM层，用于处理序列信息
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        # 线性层，将LSTM的输出转换为预测结果
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的token序列 [batch_size, seq_len]
            
        Returns:
            output: 预测结果 [batch_size]
        """
        # 将输入序列转换为词嵌入 [batch_size, seq_len, embedding_size]
        embed = self.embedding(x)
        # 通过LSTM获取序列输出 [batch_size, hidden_size]
        output, (hx, cx) = self.lstm(embed)
        # 取最后一个时间步的隐藏状态作为序列的表示 [batch_size, hidden_size]
        last_hidden_status = output[:, -1, :]
        # 通过线性层得到预测分数，并去除最后一个维度 [batch_size]
        output = self.linear(last_hidden_status).squeeze(dim=-1)
        return output


if __name__ == '__main__':
    model = ReviewAnalyzeModel(vocab_size=1000, padding_idx=0)
    dummy_input = torch.randint(low=0, high=1000, size=(config.BATCH_SIZE, config.SEQ_LEN), dtype=torch.long)
    summary(model, input_data=dummy_input)
