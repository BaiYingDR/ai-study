import torch.cuda
from tqdm import tqdm

from NLP.input_method.src import config
from NLP.input_method.src.dataset import get_dataloader
from NLP.input_method.src.model import InputMethodModule
from NLP.input_method.src.predict import predict_batch
from NLP.input_method.src.tokenizer import JiebaTokenizer


def evaluate(model, dataloader, device):
    """
    模型评估。
    :param model: 模型。
    :param dataloader: 数据加载器。
    :param device: 设备。
    :return: 准确率。
    """

    total_count = 0
    top1_correct = 0
    topk_correct = 0

    model.eval()

    for inputs, targets in tqdm(dataloader, desc="eval"):
        inputs = inputs.to(device)
        targets = targets.tolist()

        predicted_ids = predict_batch(inputs, model)

        for pred, target in zip(predicted_ids, targets):
            if pred[0] == target:
                top1_correct += 1
            if target in pred:
                topk_correct += 1
            total_count += 1

    top1_acc = top1_correct / total_count
    topk_acc = topk_correct / total_count
    return top1_acc, topk_acc


def run_evaluate():
    """
    运行评估流程。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    model = InputMethodModule(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    dataloader = get_dataloader(train=False)

    top1_acc, topk_acc = evaluate(model, dataloader, device)

    print("======= 评估结果 =======")
    print(f"Top-1 准确率: {top1_acc:.4f}")
    print(f"Top-5 准确率: {topk_acc:.4f}")
    print("========================")


if __name__ == '__main__':
    run_evaluate()
