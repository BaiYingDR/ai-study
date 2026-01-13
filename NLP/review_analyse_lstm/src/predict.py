import jieba
import torch

from NLP.review_analyse_lstm.src import config
from NLP.review_analyse_lstm.src.model import ReviewAnalyzeModel
from NLP.review_analyse_lstm.src.tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """

    :param model:
    :param inputs:
    :return:
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        batch_result = torch.sigmoid(output)
    return batch_result.tolist()


def predict(text, device, tokenizer, model):
    """

    :param text:
    :param device:
    :param tokenizer:
    :param model:
    :return:
    """
    indexes = tokenizer.encode(text, config.SEQ_LEN)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)

    batch_result = predict_batch(model, input_tensor)
    return batch_result[0]


def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    while True:
        user_input = input(">")
        if user_input == 'exit':
            break
        if user_input == '':
            continue
        result = predict(user_input,device,tokenizer,model)

        if result > 0.5:
            print("positive")
        else:
            print("negative")


if __name__ == '__main__':
    run_predict()
