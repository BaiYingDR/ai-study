import jieba
import torch.cuda

from NLP.input_method.src import config
from NLP.input_method.src.model import InputMethodModule


def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]  # f.read().splitlines()

    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    model = InputMethodModule(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "model.pth"))

    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([indexes])

    # model进入evaluate模式
    model.eval()
    # 计算模式下不需要计算梯度，直接走前向传播
    with torch.no_grad():
        output = model(input_tensor)

    top5_indices = torch.topk(output, 5).indices


if __name__ == '__main__':
    predict()
