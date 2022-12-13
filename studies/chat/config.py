import json
import numpy as np
from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.models.gpt2 import GPT
from src.net.models.utils import CfgNode


chat_file = f'studies/chat/datasets/alexa/train_small.json'

initial_weight = 0.2
weight_increase = 0.037
temp_decrease = 0.08
neuron_opening = 0.75

n_dim = 50
size = 3
history = 1
next = 1

vocabulary = Vocabulary.from_file('studies/chat/dataset')

config = CfgNode()
config.model_type = None
config.vocab_size = len(vocabulary.vocabulary)
config.block_size = size
config.n_embd = n_dim
config.n_layer = 5
config.n_head = 5
config.embd_pdrop = 0.1
config.attn_pdrop = 0.1
config.resid_pdrop = 0.1
config.betas = (0.9, 0.95)
config.weight_decay = 0.1 # only applied on matmul weights
config.grad_norm_clip = 1.0
config.learning_rate = 3e-4
config.pretrained_embeddings = None

settings = json.loads(
    open(f'studies/chat/dataset/dataset.settings.json').read())

contexts_list = None
contexts = {}

for context_name in settings['contexts']:
    if context_name != 'default' and (contexts_list is None or context_name in contexts_list):
        context = Context.from_file(
            'studies/chat/dataset', context_name, vocabulary)
        contexts[context_name] = context
        