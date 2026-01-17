import torch
from torch import nn
from torch.nn.functional import embedding

from NLP.trans_seq2seq.src import config


class TranslationEncoder(nn.Module):

    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

    def forward(self, x):
        # x:[batch_size, seq_len]
        """
        前向传播函数

        Args:
            x: 输入张量，形状为(batch_size, seq_len)，包含序列数据

        Returns:
            last_hidden_state: 最后一个时间步的隐藏状态，形状为(batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        # embedded.shape:[batch_size,seq_len, embedding_dim]

        # encoder step 的输出中也存在 hidden_status，也可以往外输出，但是没啥用，因为已经向量化了，看不懂
        output, _ = self.gru(embedded)
        # output.shape:[batch_size, seq_len, hidden_size]

        # 计算每个序列的实际长度（排除padding部分）
        # 这里是因为同一个batch下的所有token，长度被认为填充了，只有填充完了才能形成一个正确格式的输入[batch_size, seq_len]
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        # 根据实际长度提取每个序列最后一个有效时间步的输出作为最终隐藏状态
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        return last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

        self.linear = nn.Linear(config.HIDDEN_SIZE, vocab_size)

    def forward(self, x, hidden):
        """
            x：第一个输入，shape:[batch_size,1]
            hidden：encoding step 中最后一个隐藏状态，shape:[1,batch_size,hidden_size]
        """
        # embedded.shape:[batch_size,1,embedding_dim]
        embedded = self.embedding(x)
        # output.shape:[batch_size,1,hidden_size]
        # hidden.shape:[1,batch_size,hidden_size]
        output, hidden = self.gru(embedded, hidden)
        output = self.linear(output)

        return output, hidden


class TranslationModel(nn.Module):
    def __init__(self, encoder_vocab_size, encoder_padding_index,
                 decoder_vocab_size, decoder_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=encoder_vocab_size,
                                          padding_index=encoder_padding_index)
        self.decoder = TranslationDecoder(vocab_size=decoder_vocab_size,
                                          padding_index=decoder_padding_index)
