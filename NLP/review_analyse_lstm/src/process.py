import pandas as pd
from sklearn.model_selection import train_test_split

from NLP.review_analyse_lstm.src import config
from NLP.review_analyse_lstm.src.tokenizer import JiebaTokenizer
from NLP.review_analyse_lstm.src.utils import safe_path_join


def process():
    print("数据处理开始")

    df = pd.read_csv(
        safe_path_join(config.RAW_DATA_DIR, "online_shopping_10_cats.csv"),
        usecols=['label', 'review'],
        encoding="UTF-8"
    ).dropna().sample(frac=0.1, random_state=42)

    # 验证数据是否为空
    if df.empty:
        raise ValueError("读取的数据为空")

    # 检查必需的列是否存在
    if not all(col in df.columns for col in ['label', 'review']):
        raise ValueError("数据中缺少必需的列: 'label' 或 'review'")

    df = df[df['review'].str.strip().ne('')]

    if df.empty:
        raise ValueError("数据清洗后结果为空")

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    if train_df.empty or test_df.empty:
        raise ValueError("数据分割后训练集或测试集为空")

    # 构建词汇表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 加载词汇表
    tokenizers = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 计算训练数据中每条评论的token长度，并获取第95百分位数作为最大序列长度的参考值
    train_df['review'].apply(lambda x: len(tokenizers.tokenize(x))).quantile(0.95)

    train_df['review'] = train_df['review'].apply(lambda x: tokenizers.encode(x, config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.json', orient='records', lines=True)

    test_df['review'] = test_df['review'].apply(lambda x: tokenizers.encode(x, config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.json', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    process()
