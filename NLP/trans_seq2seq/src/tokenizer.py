from abc import abstractmethod, ABC

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from sympy.logic.boolalg import Boolean
from tqdm import tqdm

from NLP.input_method.src import config


class BaseTokenizer(ABC):
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls, sentence) -> list[str]:
        """
        这里不能使用abstractmethod, 抽象方法需要实例化之后才能使用，
        """
        pass

    def encode(self, sentence, add_sos_eos: Boolean):
        """
        对输入的句子进行编码处理

        参数:
            sentence: 待编码的句子，通常为字符串或token序列
            seq_len: 序列的最大长度限制
            add_sos_eos (Boolean): 是否在序列开头和结尾添加开始(SOS)和结束(EOS)标记

        返回值:
            编码后的序列，通常是数字ID序列或张量格式
        """

        tokens = self.tokenize(sentence)
        # 将 token 转为索引，未知词用 unk 索引替代
        token_indices = [self.word2index.get(token, self.unk_token_index) for token in tokens]

        if add_sos_eos:
            # 添加开始和结束标记
            token_indices = [self.sos_token_index] + token_indices + [self.eos_token_index]

        return token_indices

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        构建词表并保存到文件

        :param sentences:
        :param vocab_file:
        :return:
        """
        vocab_set = set()
        for sentence in tqdm(sentences, desc="tokenize"):
            for word in cls.tokenize(sentence):
                vocab_set.add(word)

        # 构建词汇表列表，包含特殊标记和过滤后的词汇
        # 首先添加填充标记和未知标记，然后添加词汇集合中非空且非未知标记的词汇
        vocab_list = ([cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token]
                      + [token for token in vocab_set if
                         token is not None
                         and token != ''
                         and token != cls.unk_token])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_file):
        """
        从文件加载词表

        :param vocab_file:
        :return:
        """
        with open(vocab_file, "r", encoding="UTF-8") as f:
            vocal_list = [line.strip() for line in f.readlines()]
            # 创建并返回当前类的新实例，传入处理后的vocal_list作为参数
            return cls(vocal_list)


class EnglishTokenizer(BaseTokenizer):
    """
    英文分词器
    """
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, sentence):
        return cls.tokenizer.tokenize(sentence)

    def decode(self, indexes):
        return self.detokenizer.detokenize([self.index2word[index] for index in indexes])


class ChineseTokenizer(BaseTokenizer):
    """
    中文分词
    """

    @classmethod
    def tokenize(cls, sentence):
        """
        按照字符切分句子
        """
        return list(sentence)


if __name__ == '__main__':
    pass
