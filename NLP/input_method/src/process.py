import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from NLP.input_method.src import config


def build_dataset(word2index, sentences):
    indexed_sentences = [[word2index.get(token, 0) for token in jieba.lcut(sentence)] for sentence in
                         sentences]
    train_dateset = []
    for sentence in tqdm(indexed_sentences):
        for i in (range(len(sentence) - config.SEQ_LEN)):
            train_dateset.append({'input': sentence[i:i + config.SEQ_LEN],
                                  'target': sentence[i + config.SEQ_LEN]})
    return train_dateset


def process():
    """
    1. 获取json格式数据提取对话信息
    2. 划分训练集，测试集
    3. 训练集分词后去重落地
    4. 提取分词后的word 以及对应的index
    :return:
    """
    print("开始处理数据")
    df = pd.read_json(
        config.RAW_DATA_DIR / "synthesized_.jsonl",
        lines=True,
        orient="records"
    )
    print(df.head())

    sentences = []
    for dialog in tqdm(df['dialog']):
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])
    print(len(sentences))

    train_sentences, test_sentences = train_test_split(
        sentences,
        test_size=0.2
    )

    vocab_set = set()
    for train_sentence in train_sentences:
        vocab_set.update(jieba.lcut(train_sentence))

    vocab_list = ['<unk>'] + list(vocab_set)
    print(len(vocab_list))

    with open(config.MODELS_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(vocab_list))

    word2index = {word: index for index, word in enumerate(vocab_list)}

    train_dateset = build_dataset(word2index, train_sentences)
    pd.DataFrame(train_dateset).to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    test_dataset = build_dataset(word2index, test_sentences)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    process()
