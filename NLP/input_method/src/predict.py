from pathlib import Path

import jieba
import torch.cuda

from NLP.input_method.src import config
from NLP.input_method.src.model import InputMethodModule
from NLP.input_method.src.tokenizer import JiebaTokenizer

def predict_batch(input_tensor, model):
    """

    :param input_tensor:
    :param model:
    :return:
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predict_ids = torch.topk(output, k=5, dim=-1).indices
    return predict_ids.tolist()


def predict(text, model, device, tokenizers):
    """

    :param text:
    :param model:
    :param device:
    :param tokenizers:
    :return:
    """
    input_ids = tokenizers.encode(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    topk_ids = predict_batch(input_tensor, model)[0]

    return [tokenizers.index2word[topk_id] for topk_id in topk_ids]


def run_predict():
    """

    :return:
    """
    # 初始化设备，优先使用CUDA GPU，如果不可用则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从词汇表文件加载jieba分词器
    tokenizers = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 创建输入法模块模型，设置词汇表大小并移动到指定设备
    model = InputMethodModule(vocab_size=tokenizers.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    input_history = ""
    max_history_length = 10000
    while True:
        input_text = input("请输入：")
        if input_text == "exit":
            break
        if input_text.strip() == "":
            continue
        input_history += input_text
        print(input_history)
        if len(input_history) > max_history_length:
            input_history = input_history[-max_history_length:]  # 保留最近的内容

        top5_tokens = predict(input_history, model, device, tokenizers)
        print("top5：", top5_tokens)


if __name__ == '__main__':
    run_predict()
