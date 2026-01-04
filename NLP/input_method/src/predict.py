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
    model.load_state_dict(torch.load(config.MODELS_DIR / "model.pt"))

    tokens = jieba.lcut(text)
    indexes = [word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([indexes])
    input_tensor = input_tensor.to(device)

    # model进入evaluate模式
    model.eval()
    # 计算模式下不需要计算梯度，直接走前向传播
    with torch.no_grad():
        output = model(input_tensor)

    top5_indices = torch.topk(output, 5).indices

    top5_indexes_list = top5_indices.tolist()
    top5_tokens = [index2word[index] for index in top5_indexes_list[0]]

    return top5_tokens


def run_predict():
    input_history = ""
    max_history_length = 10000
    while True:
        try:
            input_text = input("请输入：")
            if input_text == "exit":
                break
            if input_text.strip() == "":
                continue
            input_history += input_text
            print(input_history)
            if len(input_history) > max_history_length:
                input_history = input_history[-max_history_length:]  # 保留最近的内容

            top5_tokens = predict(input_history)
            print("top5：", top5_tokens)
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            continue


if __name__ == '__main__':
    run_predict()
