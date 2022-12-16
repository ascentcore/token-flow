import os
import json

from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.models.utils import CfgNode

dir_path = f'studies/text/datasets/train'
text_files = os.listdir(dir_path)

initial_weight = 0.2
weight_increase = 0.037
temp_decrease = 0.08
neuron_opening = 0.75

n_dim = 50
size = 21
history = 10
next = 10


def get_training_setup():
    contexts = {}
    vocabulary = Vocabulary.from_file('studies/text/dataset')

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
    config.learning_rate = 0.001
    config.pretrained_embeddings = None

    settings = json.loads(open(f'studies/text/dataset/dataset.settings.json').read())

    contexts_list = None

    for context_name in settings['contexts']:
        if context_name != 'default' and (contexts_list is None or context_name in contexts_list):
            context = Context.from_file('studies/text/dataset', context_name, vocabulary)
            contexts[context_name] = context

    return contexts, vocabulary, config
    