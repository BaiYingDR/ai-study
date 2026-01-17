import pandas as pd
from sklearn.model_selection import train_test_split

from NLP.review_analyse_lstm.src.utils import safe_path_join
from NLP.trans_seq2seq.src.tokenizer import EnglishTokenizer, ChineseTokenizer
from NLP.trans_seq2seq.src import config


def process():
    print("数据处理开始")

    df = pd.read_csv(
        safe_path_join(config.RAW_DATA_DIR, "cmn.txt"),
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=['en', 'zh']
    )

    # 清理数据：移除包含空值的行
    df = df.dropna()
    # 进一步清理：移除英文和中文字段中包含空白字符串的行
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 将数据集分割为训练集和测试集，测试集占20%
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建词汇表：基于训练集数据创建英文和中文词汇表文件
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.RAW_DATA_DIR / 'en_vocab.txt')
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.RAW_DATA_DIR / 'zh_vocab.txt')

    # 从词汇表文件加载分词器实例
    en_tokenizer = EnglishTokenizer.from_vocab(config.RAW_DATA_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.RAW_DATA_DIR / 'zh_vocab.txt')

    # 对训练集数据进行编码处理，并保存为JSON格式
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.json', orient='records', lines=True)

    # 对测试集数据进行编码处理，并保存为JSON格式
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.json', orient='records', lines=True)


if __name__ == '__main__':
    process()
