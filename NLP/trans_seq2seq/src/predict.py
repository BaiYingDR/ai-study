from tokenize import generate_tokens

import torch

from NLP.trans_seq2seq.src import config
from NLP.trans_seq2seq.src.model import TranslationModel
from NLP.trans_seq2seq.src.tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, tgt_tokenizer):
    """

    :param model:
    :param inputs:
    :return:
    """
    model.eval()
    with torch.no_grad():
        batch_size = inputs.shape[0]
        device = inputs.device

        context_vector = model.encoder(inputs).unsqueeze(0)
        decoder_input = torch.full([batch_size, 1], tgt_tokenizer.sos_token_index, device=device)

        generated = []
        for i in range(config.MAX_SEQ_LENGTH):
            output, context_vector = model.decoder(decoder_input, context_vector)
            next_token_index = torch.argmax(output, dim=-1)
            generated.append(next_token_index)

            decoder_input = next_token_index

            if next_token_index.item() == tgt_tokenizer.eos_token_index:
                break
        generated_tensor = torch.cat(generated, dim=1)
        generated_list = generated_tensor.tolist()

        for index, sentence in enumerate(generated_list):
            if tgt_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(tgt_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        return generated_list


def predict(text, device, src_tokenizer, tgt_tokenizer, model):
    """

    :param text:
    :param device:
    :param src_tokenizer:
    :param tgt_tokenizer:
    :param model:
    :return:
    """
    indexes = src_tokenizer.encode(text, add_sos_eos=False)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)

    batch_result = predict_batch(model, input_tensor, tgt_tokenizer)
    return tgt_tokenizer.decode(batch_result[0])


def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    zh_tokenizer = ChineseTokenizer.from_vocab(config.RAW_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.RAW_DATA_DIR / 'en_vocab.txt')

    model = TranslationModel(zh_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                             en_tokenizer.vocab_size, en_tokenizer.pad_token_index) \
        .to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    while True:
        user_input = input(">")
        if user_input == 'q':
            break
        if user_input == '':
            continue
        result = predict(user_input, device, zh_tokenizer, en_tokenizer, model)
        print(result)


if __name__ == '__main__':
    run_predict()
