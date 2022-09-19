import os
import re
import json

from tqdm import tqdm
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary

from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.ds.transformer_runtime_dataset import TransformerRuntimeDataset
from src.net.models.rnntrans import RNNModel, TransformerModel
from src.net.models.vit import VisionTransformer
import torch.nn as nn
import torch

from src.net.trainer import Trainer


vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=False,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.0275
weight_increase = 0.00137
temp_decrease = 0.0175
neuron_opening = 0.55

count_lines = 8


def get_context(name,
                initial_weight=initial_weight,
                weight_increase=weight_increase,
                temp_decrease=temp_decrease,
                neuron_opening=neuron_opening):

    context = Context(name, vocabulary,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def build_dataset():
    dataset = Dataset(vocabulary)
    dataset.delete_context('default')

    chat_data = json.loads(
        open(f'studies/chat/datasets/alexa/train_small_medium.json').read())

    agents = []

    for id in tqdm(chat_data.keys()):
        content = chat_data[id]["content"]
        for data in content:
            agent = data["agent"]
            message = data["message"]

            if not dataset.has_context(agent):
                agents.append(agent)
                context = get_context(agent)
                dataset.add_context(context)
            else:
                context = dataset.get_context(agent)

            # print(f'adding to {agent} message: {message}')
            context.add_text(message)

    dataset.store('studies/chat/dataset')

    print('Starting training ....')
    num_patches = 24

    model = VisionTransformer(
        embed_dim=64,
        hidden_dim=128,
        num_channels=1,
        num_heads=4,
        num_layers=8,
        num_classes=vocabulary.size(),
        patch_size=vocabulary.size(),
        num_patches=num_patches,
        dropout=0.02)

    trainer = Trainer(model, vocabulary)

    stimulate_with = ["What music do you like?",
                      "I love cat videos!",
                      "The traffic today is really heavy."]
    try:
        os.mkdir('studies/chat/models/')
    except:
        pass

    for iteration in range(10):
        for id in tqdm(chat_data.keys()):
            content = chat_data[id]["content"]
            print(f'Content Length: {len(content)}')
            for data in tqdm(content):
                agent = data["agent"]
                message = data["message"]
                for context in dataset.contexts.keys():
                    if context != agent:
                        dataset.get_context(
                            context).stimulate_sequence(message)
                    else:
                        dataset.get_dataset(context, context).add_text(message)

            ds = TransformerRuntimeDataset(dataset, num_patches)

            print(f'Training on {id} with length {len(ds)}')
            trainer.train(ds, epochs=50, batch_size=16)

            dataset.reset_stimulus()

            contexts_list = list(dataset.contexts.values())

            print("\n\nTraining done, starting to generate ...\n")
            for start_stimulus in stimulate_with:
                speaker = contexts_list[1]
                listener = contexts_list[0]
                print('\nStarting conversation about: \n' + start_stimulus)
                pre = start_stimulus

                for i in range(count_lines):
                    input_data = Dataset.get_sample_data(speaker, pre)
                    text = trainer.get_sentence(
                        speaker,
                        input_data,
                        generate_length=50,
                        prevent_convergence_history=2,
                        num_patches=num_patches,
                        break_on_eol=True)
                    print(f'>[{i}] {speaker.name}: {text}')
                    # listener.stimulate_sequence(text)
                    pre = text
                    buf = speaker
                    speaker = listener
                    listener = buf

                dataset.reset_stimulus()
            dataset.clear_datasets()

            torch.save(model, f'studies/chat/models/model_{iteration}_{id}')


if __name__ == '__main__':
    build_dataset()
