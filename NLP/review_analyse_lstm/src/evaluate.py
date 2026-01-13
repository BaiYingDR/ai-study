import torch

from NLP.review_analyse_lstm.src.dataset import get_dataloader
from NLP.review_analyse_lstm.src.model import ReviewAnalyzeModel
from NLP.review_analyse_lstm.src.tokenizer import JiebaTokenizer
from NLP.review_analyse_lstm.src import config
from NLP.review_analyse_lstm.src.predict import predict_batch


def evaluate(model, dataloader, device):
    """
    模型评估。

    :param model: 模型。
    :param dataloader: 数据加载器。
    :param device: 设备。
    :return: 准确率。
    """
    total_count = 0
    correct_count = 0


    for inputs, targets in dataloader:
        # 数据转移到设备
        inputs = inputs.to(device)
        targets = targets.tolist()

        # 获取预测概率
        probs = predict_batch(model, inputs)

        # 统计准确率
        for prob, target in zip(probs, targets):
            pred_label = 1 if prob > 0.5 else 0
            if pred_label == target:
                correct_count += 1
            total_count += 1

    return correct_count / total_count


def run_evaluate():
    """
    运行评估流程。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    dataloader = get_dataloader(train=False)

    acc = evaluate(model, dataloader, device)

    print("========== 评估结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("=============================")


if __name__ == '__main__':
    run_evaluate()
