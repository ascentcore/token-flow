import torch
import numpy as np
from torch.nn import functional as F

from train import get_input
from src.net.models.gptS import GPT
from src.embeddings.embeddings import get_embeddings

from config import get_training_setup, get_model_name
from buildcontext import get_context_for_file

path = f'studies/single-context'


def load_embeddings(vocabulary, config):
    matrix_len = len(vocabulary.vocabulary)
    pretrained_embeddings = torch.zeros((matrix_len, config.n_embd))
    embeddings = get_embeddings()

    for i, word in enumerate(vocabulary.vocabulary):
        try: 
            pretrained_embeddings[i] = torch.tensor(embeddings[word])
        except KeyError:
            if word == '<null>':
                pretrained_embeddings[i] = torch.tensor(np.zeros(config.n_embd))
            else:
                pretrained_embeddings[i] = torch.tensor(np.random.normal(scale=0.6, size=(config.n_embd, )))

    config.pretrained_embeddings = pretrained_embeddings


@torch.no_grad()
def generate():
    contexts, vocabulary, config = get_training_setup()
    
    #################
    ##   Configs   ##
    #################
    
    last_epoch = True
    topk = 1


    epoch_end = '-epoch-end' if last_epoch else ''
    model_name = get_model_name(config)

    load_embeddings(vocabulary=vocabulary, config=config)

    print('model_name', model_name)
    model = GPT(config)
    model_name = f'studies/single-context/models/{model_name}{epoch_end}.pt'
    print(f'Loading model from {model_name}')
    model.load_state_dict(torch.load(model_name)['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    context = contexts['default']
    
    # context = get_context_for_file(vocabulary, 'city_mouse.txt')
    # context.render(f'.', force_text_rendering=True, dpi=600)
    
    
    
    context.decrease_stimulus(1)

    print('\n\n --- Test --- \n\n')
    sentence = 'This is a city mouse. He lives '
    # sentence = ''
    if sentence != '':
        _, sentences = context.vocabulary.get_token_sequence(
            sentence, append_to_vocab=False, skip_eol=True)
        for sent in sentences:
            for tokens in sent:
                for token in tokens:
                    context.stimulate(token)

    input_data = get_input(context)

    for i in range(150):
        x_data = [[d[0] for d in input_data]]
        stimulus_data = [[d[1] for d in input_data]]

        x_data = torch.tensor(x_data)
        stimulus_data = torch.tensor(stimulus_data)

        logits, _, acc = model(x_data.to(device), stimulus_data)
        probs = F.softmax(logits, dim=-1)
        if topk is None:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=topk, dim=-1)
        token = context.vocabulary.vocabulary[idx_next.item()]
        context.stimulate(token)

        if token == '<eol>' and len(sentence) > 0:
            sentence = sentence + token + ' '
            break

        sentence = sentence + token + ' '

        input_data = get_input(context)

    print(sentence)


if __name__ == '__main__':
    generate()
