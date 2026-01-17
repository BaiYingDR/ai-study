import torch
from tensorboard import summary
from torch import nn

from NLP.review_analyse_GRU.src import config


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
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        # LSTM层，用于处理序列信息
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                            hidden_size=config.HIDDEN_SIZE,
                            batch_first=True)
        # 线性层，将LSTM的输出转换为预测结果
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1)

    def forward(self, x: torch.Tensor):
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
        output, _ = self.gru(embed)
        # 获取批次索引，用于后续的张量索引操作
        batch_indexes = torch.arange(0, output.shape[0])
        # 计算每个序列的实际长度（非零元素的个数）
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        # 根据每个序列的实际长度，提取对应位置的隐藏状态（最后一个有效token的表示）
        last_hidden = output[batch_indexes, lengths - 1]
        # 通过线性层将最后的隐藏状态映射到输出空间
        output = self.linear(last_hidden).squeeze(-1)

        return output


if __name__ == '__main__':
    model = ReviewAnalyzeModel(vocab_size=1000, padding_idx=0)
    dummy_input = torch.randint(low=0, high=1000, size=(config.BATCH_SIZE, config.SEQ_LEN), dtype=torch.long)
    summary(model, input_data=dummy_input)
