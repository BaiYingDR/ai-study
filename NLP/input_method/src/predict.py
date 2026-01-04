from pathlib import Path

import jieba
import torch.cuda

from NLP.input_method.src import config
from NLP.input_method.src.model import InputMethodModule
from NLP.input_method.src.utils import safe_path_join


def predict(text):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 安全路径验证
        vocab_file_path = safe_path_join(config.MODELS_DIR, 'vocab.txt')

        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]  # f.read().splitlines()

        # 检查词汇表是否为空
        if not vocab_list:
            raise ValueError("Vocabulary list is empty")

        word2index = {word: index for index, word in enumerate(vocab_list)}
        index2word = {index: word for index, word in enumerate(vocab_list)}

        model = InputMethodModule(vocab_size=len(vocab_list)).to(device)
        model_file_path = safe_path_join(config.MODELS_DIR, "model.pt")
        model.load_state_dict(torch.load(model_file_path, map_location=device))

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

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except torch.serialization.pickle.UnpicklingError:
        print("Model file is corrupted or unsafe")
        raise
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise


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
