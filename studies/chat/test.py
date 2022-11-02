import torch
import os
import numpy
from src.net.models.gpt2 import GPT
from config import vocabulary, config, contexts
from trainer import get_input
from torch.nn import functional as F

path = f'studies/chat'

@torch.no_grad()
def generate():
    
    last_epoch = False
    epoch_end = '-epoch-end' if last_epoch else ''
    model_name = f"gpt-2-{config.block_size}-{config.vocab_size}-{config.n_embd}-{config.n_layer}-{config.n_head}{epoch_end}"

    model = GPT(config)
    model.load_state_dict(torch.load(
        f'studies/chat/models/{model_name}.pt')['model_state_dict'])
    model.eval()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    talker = contexts['agent_1']
    listener = contexts['agent_2']

    talker.decrease_stimulus(1)
    listener.decrease_stimulus(1)

    print('\n\n --- Test --- \n\n')
    # sentence = 'Do you like the snakes?'
    sentence = ''
    if sentence != '':
        _, sentences = talker.vocabulary.get_token_sequence(
            sentence, append_to_vocab=False, skip_eol=True)
        for sent in sentences:
            for tokens in sent:
                for token in tokens:
                    talker.stimulate(token)
                

    input_data = get_input(talker)

    for j in range(20):
    
        for i in range(50):
            x = torch.tensor(numpy.array([input_data]), dtype=torch.int32)
            logits, loss = model(x.to(device))
            logits = logits[:, -1, :] / 1.0
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            token = talker.vocabulary.vocabulary[idx_next.item()]
            talker.stimulate(token)

            if token == '<eol>' and len(sentence) > 0:
                sentence = sentence + token + ' '
                break

            sentence = sentence + token + ' '

            input_data = get_input(talker)

        print(talker.name, ' >> ', sentence)
        listener.stimulate_sequence(sentence)
        buf = talker
        talker = listener
        listener = buf
        sentence = ''


if __name__ == '__main__':
    generate()
