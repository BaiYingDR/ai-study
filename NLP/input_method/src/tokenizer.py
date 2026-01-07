import jieba
from tqdm import tqdm

from NLP.input_method.src import config


class JiebaTokenizer:
    unk_token = '<unk>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(sentence):
        """
        该方法没有用到该类中self下的的任何一个variable，所以设置为静态方法
        :param sentence:
        :return:
        """
        return jieba.lcut(sentence)

    def encode(self, sentence):
        """
                将句子编码为索引列表。

                :param sentence: 输入句子。
                :return: 索引列表。
                """
        tokens = self.tokenize(sentence)
        # 将 token 转为索引，未知词用 unk 索引替代
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

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

        vocab_list = [cls.unk_token] + [token for token in vocab_set if
                                        token is not None and token != '' and token != cls.unk_token]

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
            return cls(vocal_list)
