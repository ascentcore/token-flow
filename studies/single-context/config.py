import os
import json

from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.models.utils import CfgNode

dir_path = f'studies/single-context/datasets/train_adap'
text_files = os.listdir(dir_path)

initial_weight = 0.2
weight_increase = 0
temp_decrease = 0.08
neuron_opening = 0.75

n_dim = 50

history = 20
next = 20
size = history + next + 1

def get_model_name(config):
    return f"gpt-2-{config.block_size}-{config.vocab_size}-{config.n_embd}-{config.n_layer}-{config.n_head}-{history}-{next}"

def get_training_setup():
    contexts = {}
    vocabulary = Vocabulary.from_file('studies/single-context/dataset')

    config = CfgNode()
    config.model_type = None
    config.vocab_size = len(vocabulary.vocabulary)
    config.block_size = size
    config.n_embd = n_dim
    config.n_layer = 1
    config.n_head = 1
    config.embd_pdrop = 0.1
    config.attn_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.betas = (0.9, 0.95)
    config.weight_decay = 0.1 # only applied on matmul weights
    config.grad_norm_clip = 1.0
    # config.learning_rate = 0.001
    config.learning_rate = 0.0001
    config.pretrained_embeddings = None

    settings = json.loads(open(f'studies/single-context/dataset/dataset.settings.json').read())

    contexts_list = None

    for context_name in settings['contexts']:
        if (contexts_list is None or context_name in contexts_list):
            context = Context.from_file('studies/single-context/dataset', context_name, vocabulary)
            contexts[context_name] = context

    return contexts, vocabulary, config
    