from torch import nn

from NLP.input_method.src import config


class InputMethodModule(nn.Module):
    """
        输入法模型
    """
    def __init__(self, vocab_size):
        """
        :param vocab_size: 嵌入层的 “词汇表大小”，即需要映射的唯一整数索引的总数
        embedding_dim=config.EMBEDDING_DIM: 嵌入向量的维度（即每个索引映射后的向量长度）
        :return:
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)            # 词潜入层
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)    # 模型定义
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=vocab_size)                        # 全连接层

    def forward(self, x):
        embed = self.embedding(x)                           # [batch_size, seq_len, embedding_size]
        output, _ = self.rnn(embed)                         # [batch_size, seq_len, hidden_size]
        last_hidden_status = output[:, -1, :]               # [batch_size, hidden_size]
        output = self.linear(last_hidden_status)            # [batch_size, vocab_size]
        return output
